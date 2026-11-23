// DCU device test for the page-based megakernel.
//
// It builds small instruction streams (NoOp + ElementwiseOp, GemmOp and
// EmbeddingOp) and verifies the results, the instruction pipeline reuse across
// more instructions than stages, partial tiles, blocks without work, and
// re-launching the same kernel over fresh buffers.
//
// Build/run through: python tests/csrc/run_dmk_lds_test.py
#include "demo_op.h"
#include "embedding.h"
#include "gemm.h"
#include "megakernel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

namespace config = dmk::config;
using dmk::mk::ElementwiseOp;
using dmk::mk::EmbeddingOp;
using dmk::mk::GemmOp;
using dmk::mk::kDynamicBytes;
using dmk::mk::kLaunchBytes;
using dmk::mk::kStaticBytes;
using dmk::mk::MkGlobals;
using dmk::tensor::HbmLayout;
using dmk::tensor::MemorySpace;
using dmk::tensor::Shape;
using dmk::tensor::Tensor;

constexpr int kInstructionWidth = config::kInstructionWidth;
constexpr int kTileElements = ElementwiseOp<MkGlobals>::kTileElements;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void check_hip(hipError_t status, const char* what) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(status));
    std::exit(1);
}

int float_bits(float value) {
    int bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void write_noop(int* words) {
    for (int i = 0; i < kInstructionWidth; ++i) words[i] = 0;
}

void write_elementwise(int* words, int tile, float scale, float bias,
                       int count) {
    write_noop(words);
    words[0] = ElementwiseOp<MkGlobals>::opcode;
    words[1] = tile;
    words[2] = float_bits(scale);
    words[3] = float_bits(bias);
    words[4] = count;
}

void run_case(const char* name, size_t elements, uint32_t blocks,
              bool all_noop) {
    constexpr float scale = 2.0f;
    constexpr float bias = 1.0f;
    constexpr float sentinel = -12345.0f;

    const int tiles = static_cast<int>(
        elements / kTileElements + (elements % kTileElements != 0));
    const int per_block =
        tiles == 0 ? 0 : (tiles + static_cast<int>(blocks) - 1) /
                              static_cast<int>(blocks);
    const int max_tiles = all_noop ? 0 : per_block;
    const int num_instructions = 2 + max_tiles;

    std::vector<float> input(elements);
    std::vector<float> expected(elements);
    std::vector<float> actual(elements, sentinel);
    for (size_t i = 0; i < elements; ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 17) - 8);
        expected[i] = all_noop ? sentinel : scale * input[i] + bias;
    }

    std::vector<int> words(static_cast<size_t>(blocks) * num_instructions *
                           kInstructionWidth);
    for (uint32_t block = 0; block < blocks; ++block) {
        int* row =
            words.data() + static_cast<size_t>(block) * num_instructions *
                               kInstructionWidth;
        write_noop(row);
        if (!all_noop) {
            const int first = static_cast<int>(block) * per_block;
            const int last =
                first + per_block < tiles ? first + per_block : tiles;
            for (int tile = first; tile < last; ++tile) {
                const size_t base = static_cast<size_t>(tile) * kTileElements;
                const size_t remaining = elements - base;
                const int count = static_cast<int>(
                    remaining < static_cast<size_t>(kTileElements)
                        ? remaining
                        : kTileElements);
                write_elementwise(row + (1 + tile - first) * kInstructionWidth,
                                  tile, scale, bias, count);
            }
        }
        write_noop(row + (num_instructions - 1) * kInstructionWidth);
    }

    int* device_instructions = nullptr;
    float* device_input = nullptr;
    float* device_output = nullptr;
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_instructions),
                        words.size() * sizeof(int)),
              "hipMalloc(instructions)");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_input),
                        elements * sizeof(float)),
              "hipMalloc(input)");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_output),
                        elements * sizeof(float)),
              "hipMalloc(output)");
    check_hip(hipMemcpy(device_instructions, words.data(),
                        words.size() * sizeof(int), hipMemcpyHostToDevice),
              "hipMemcpy(instructions)");
    check_hip(hipMemcpy(device_input, input.data(), elements * sizeof(float),
                        hipMemcpyHostToDevice),
              "hipMemcpy(input)");
    check_hip(hipMemcpy(device_output, actual.data(), elements * sizeof(float),
                        hipMemcpyHostToDevice),
              "hipMemcpy(output)");

    const Shape<3> instruction_shape{
        {blocks, static_cast<size_t>(num_instructions),
         static_cast<size_t>(kInstructionWidth)}};
    const Shape<1> data_shape{{elements}};
    const HbmLayout<1> data_layout = HbmLayout<1>::contiguous(data_shape);
    const MkGlobals globals{
        Tensor<const int, 3, MemorySpace::HBM>{
            device_instructions, instruction_shape,
            HbmLayout<3>::contiguous(instruction_shape)},
        Tensor<const float, 1, MemorySpace::HBM>{device_input, data_shape,
                                                 data_layout},
        Tensor<float, 1, MemorySpace::HBM>{device_output, data_shape,
                                           data_layout},
        num_instructions,
    };

    if (!dmk::mk::launch_mk(globals, blocks, nullptr))
        fail("launch_mk returned false");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    check_hip(hipMemcpy(actual.data(), device_output, elements * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy(output)");

    size_t mismatches = 0;
    for (size_t i = 0; i < elements; ++i) {
        if (actual[i] != expected[i]) {
            if (mismatches < 5) {
                std::fprintf(stderr, "  %s[%zu]: got %.7g, expected %.7g\n",
                             name, i, actual[i], expected[i]);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "  %s: %zu/%zu values differ\n", name, mismatches,
                     elements);
        std::exit(1);
    }

    std::printf("  %-10s elements=%-7zu tiles=%-3d blocks=%u instructions=%d\n",
                name, elements, tiles, blocks, num_instructions);

    check_hip(hipFree(device_instructions), "hipFree(instructions)");
    check_hip(hipFree(device_input), "hipFree(input)");
    check_hip(hipFree(device_output), "hipFree(output)");
}

void run_gemm_case() {
    constexpr int N = 16;
    std::vector<float> host_a(N * N);
    std::vector<float> host_b(N * N);
    std::vector<float> host_c(N * N, 0.0f);
    for (int i = 0; i < N * N; ++i) {
        host_a[i] = static_cast<float>((i * 7) % 11) - 5.0f;
        host_b[i] = static_cast<float>((i * 3) % 9) - 4.0f;
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    int* device_instructions = nullptr;
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_a),
                        host_a.size() * sizeof(float)),
              "hipMalloc(gemm A)");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_b),
                        host_b.size() * sizeof(float)),
              "hipMalloc(gemm B)");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_c),
                        host_c.size() * sizeof(float)),
              "hipMalloc(gemm C)");
    check_hip(hipMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "hipMemcpy(gemm A)");
    check_hip(hipMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "hipMemcpy(gemm B)");
    check_hip(hipMemcpy(device_c, host_c.data(), host_c.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "hipMemcpy(gemm C)");

    constexpr int kInstructions = 3;
    std::vector<int> words(kInstructions * kInstructionWidth, 0);
    write_noop(words.data());
    write_noop(words.data() + kInstructionWidth);
    words[kInstructionWidth + 0] = GemmOp<MkGlobals>::opcode;
    words[kInstructionWidth + 1] = 0;  // m0
    words[kInstructionWidth + 2] = 0;  // n0
    words[kInstructionWidth + 3] = 1;  // k_tiles (16x16 matrices: one step)
    write_noop(words.data() + 2 * kInstructionWidth);
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_instructions),
                        words.size() * sizeof(int)),
              "hipMalloc(gemm instructions)");
    check_hip(hipMemcpy(device_instructions, words.data(),
                        words.size() * sizeof(int), hipMemcpyHostToDevice),
              "hipMemcpy(gemm instructions)");

    const Shape<3> instruction_shape{
        {1, kInstructions, static_cast<size_t>(kInstructionWidth)}};
    const Shape<2> matrix_shape{{N, N}};
    const Shape<1> empty_shape{{0}};
    const HbmLayout<1> empty_layout = HbmLayout<1>::contiguous(empty_shape);
    const MkGlobals globals{
        Tensor<const int, 3, MemorySpace::HBM>{
            device_instructions, instruction_shape,
            HbmLayout<3>::contiguous(instruction_shape)},
        Tensor<const float, 1, MemorySpace::HBM>{nullptr, empty_shape,
                                                 empty_layout},
        Tensor<float, 1, MemorySpace::HBM>{nullptr, empty_shape, empty_layout},
        kInstructions,
        Tensor<const float, 2, MemorySpace::HBM>{
            device_a, matrix_shape, HbmLayout<2>::contiguous(matrix_shape)},
        Tensor<const float, 2, MemorySpace::HBM>{
            device_b, matrix_shape, HbmLayout<2>::contiguous(matrix_shape)},
        Tensor<float, 2, MemorySpace::HBM>{
            device_c, matrix_shape, HbmLayout<2>::contiguous(matrix_shape)},
    };

    if (!dmk::mk::launch_mk(globals, 1, nullptr))
        fail("gemm launch_mk returned false");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(gemm)");
    std::vector<float> actual(N * N, 0.0f);
    check_hip(hipMemcpy(actual.data(), device_c, actual.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy(gemm C out)");

    size_t bad = 0;
    for (int r = 0; r < N; ++r) {
        for (int col = 0; col < N; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < N; ++k)
                expected += host_a[r * N + k] * host_b[k * N + col];
            if (std::fabs(actual[r * N + col] - expected) > 1e-5f) ++bad;
        }
    }
    if (bad != 0) {
        std::fprintf(stderr, "  gemm: %zu values differ\n", bad);
        std::exit(1);
    }
    std::printf("  %-10s 16x16x16 scalar reference\n", "gemm");
    check_hip(hipFree(device_a), "hipFree(gemm A)");
    check_hip(hipFree(device_b), "hipFree(gemm B)");
    check_hip(hipFree(device_c), "hipFree(gemm C)");
    check_hip(hipFree(device_instructions), "hipFree(gemm instructions)");
}

void run_embedding_case() {
    constexpr int Rows = 4;
    constexpr int Width = 8;
    std::vector<float> host_table(Rows * Width);
    std::vector<float> host_out(Rows * Width, -1.0f);
    for (int i = 0; i < Rows * Width; ++i)
        host_table[i] = static_cast<float>(i);

    float* device_table = nullptr;
    float* device_out = nullptr;
    int* device_instructions = nullptr;
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_table),
                        host_table.size() * sizeof(float)),
              "hipMalloc(embedding table)");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_out),
                        host_out.size() * sizeof(float)),
              "hipMalloc(embedding out)");
    check_hip(hipMemcpy(device_table, host_table.data(),
                        host_table.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "hipMemcpy(embedding table)");
    check_hip(hipMemcpy(device_out, host_out.data(),
                        host_out.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "hipMemcpy(embedding out)");

    constexpr int kInstructions = 3;
    std::vector<int> words(kInstructions * kInstructionWidth, 0);
    write_noop(words.data());
    words[kInstructionWidth + 0] = EmbeddingOp<MkGlobals>::opcode;
    words[kInstructionWidth + 1] = 2;  // src row
    words[kInstructionWidth + 2] = 1;  // dst row
    words[kInstructionWidth + 3] = Width - 1;
    write_noop(words.data() + 2 * kInstructionWidth);
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_instructions),
                        words.size() * sizeof(int)),
              "hipMalloc(embedding instructions)");
    check_hip(hipMemcpy(device_instructions, words.data(),
                        words.size() * sizeof(int), hipMemcpyHostToDevice),
              "hipMemcpy(embedding instructions)");

    const Shape<3> instruction_shape{
        {1, kInstructions, static_cast<size_t>(kInstructionWidth)}};
    const Shape<2> table_shape{{Rows, Width}};
    const Shape<1> empty_shape{{0}};
    const HbmLayout<1> empty_layout = HbmLayout<1>::contiguous(empty_shape);
    const MkGlobals globals{
        Tensor<const int, 3, MemorySpace::HBM>{
            device_instructions, instruction_shape,
            HbmLayout<3>::contiguous(instruction_shape)},
        Tensor<const float, 1, MemorySpace::HBM>{nullptr, empty_shape,
                                                 empty_layout},
        Tensor<float, 1, MemorySpace::HBM>{nullptr, empty_shape, empty_layout},
        kInstructions,
        {},
        {},
        {},
        Tensor<const float, 2, MemorySpace::HBM>{
            device_table, table_shape, HbmLayout<2>::contiguous(table_shape)},
        Tensor<float, 2, MemorySpace::HBM>{
            device_out, table_shape, HbmLayout<2>::contiguous(table_shape)},
    };

    if (!dmk::mk::launch_mk(globals, 1, nullptr))
        fail("embedding launch_mk returned false");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(embedding)");
    std::vector<float> actual(Rows * Width, 0.0f);
    check_hip(hipMemcpy(actual.data(), device_out,
                        actual.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy(embedding out)");

    for (int i = 0; i < Width - 1; ++i) {
        if (actual[Width + i] != host_table[2 * Width + i]) {
            std::fprintf(stderr, "  embedding[%d] differs\n", i);
            std::exit(1);
        }
    }
    if (actual[Width + Width - 1] != -1.0f) {
        std::fprintf(stderr, "  embedding tail was overwritten\n");
        std::exit(1);
    }
    std::printf("  %-10s row 2 -> row 1, width %d (tail scalar)\n",
                "embedding", Width - 1);
    check_hip(hipFree(device_table), "hipFree(embedding table)");
    check_hip(hipFree(device_out), "hipFree(embedding out)");
    check_hip(hipFree(device_instructions), "hipFree(embedding instructions)");
}

}  // namespace

int main() {
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    std::printf("device: %s (%s)\n", properties.name, properties.gcnArchName);
    std::printf(
        "wave=%d LDS/block=%zu static=%zu dynamic=%zu launch=%zu pages=%d x %d B\n",
        properties.warpSize, properties.sharedMemPerBlock, kStaticBytes,
        kDynamicBytes, kLaunchBytes, config::kNumPages, config::kPageSize);

    if (properties.warpSize != config::kWaveSize) {
        std::fprintf(stderr,
                     "SKIP: role partitioning requires 64-lane wavefronts, "
                     "device reports %d\n",
                     properties.warpSize);
        return 2;
    }
    if (kLaunchBytes + kStaticBytes > properties.sharedMemPerBlock) {
        std::fprintf(stderr,
                     "SKIP: plan needs %zu B, device allows %zu B\n",
                     kLaunchBytes + kStaticBytes,
                     properties.sharedMemPerBlock);
        return 2;
    }

    std::printf("running page-based megakernel cases\n");
    run_case("single", kTileElements, 1, false);
    run_case("multi", static_cast<size_t>(8) * kTileElements, 2, false);
    run_case("tail", static_cast<size_t>(3) * kTileElements + 777, 2, false);
    run_case("idle", static_cast<size_t>(2) * kTileElements, 4, false);
    run_case("noop", kTileElements, 1, true);
    run_case("relaunch", static_cast<size_t>(8) * kTileElements, 2, false);
    run_gemm_case();
    run_embedding_case();

    std::printf("page-based megakernel checks passed\n");
    return 0;
}
