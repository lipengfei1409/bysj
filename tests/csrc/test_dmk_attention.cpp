// Attention operator test: the page-based AttentionOp runs inside the
// megakernel VM (controller / loader / storer / consumer waves), but the op
// pack contains nothing else, so this is still a single-operator test.
//
// Build/run through: python tests/csrc/run_dmk_ops_test.py --op attention
#include "attention.h"

#include "megakernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace {

using dmk::tensor::HbmLayout;
using dmk::tensor::MemorySpace;
using dmk::tensor::Shape;
using dmk::tensor::Tensor;

using AttnGlobalsF32 = dmk::attention::AttentionGlobalsT<float>;
using AttnGlobalsF16 = dmk::attention::AttentionGlobalsT<__half>;
using AttnGlobalsBF16 = dmk::attention::AttentionGlobalsT<__hip_bfloat16>;
using AttnOpF32 = dmk::mk::AttentionOp<AttnGlobalsF32>;
using AttnOpF16 = dmk::mk::AttentionOp<AttnGlobalsF16>;
using AttnOpBF16 = dmk::mk::AttentionOp<AttnGlobalsBF16>;

void check_hip(hipError_t status, const char* what) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(status));
    std::exit(1);
}

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
}

struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        check_hip(hipMalloc(&ptr, bytes != 0 ? bytes : 1), "hipMalloc");
    }
    ~DeviceBuffer() {
        if (ptr) hipFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    T* data() const {
        return static_cast<T*>(ptr);
    }
};

struct Event {
    hipEvent_t event{};
    Event() { check_hip(hipEventCreate(&event), "hipEventCreate"); }
    ~Event() {
        if (event) hipEventDestroy(event);
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

// Deterministic values in [-1, 1] without pulling in a generator.
float sample(size_t index) {
    const float x = static_cast<float>((index * 2654435761u) % 1000) / 500.0f;
    return x - 1.0f;
}

template <typename T>
const char* dtype_name() {
    if (std::is_same<T, float>::value) return "f32";
    if (std::is_same<T, __half>::value) return "f16";
    return "bf16";
}

// Fills a [rows, dim] matrix, stored row-major or column-major.
template <typename T>
std::vector<T> make_matrix(size_t rows, size_t dim, bool column_major,
                           size_t salt) {
    std::vector<T> data(rows * dim);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < dim; ++col) {
            const size_t index = column_major ? col * rows + row : row * dim + col;
            data[index] =
                dmk::dtype::from_float<T>(sample(row * 131 + col * 17 + salt));
        }
    }
    return data;
}

HbmLayout<2> matrix_layout(size_t rows, size_t dim, bool column_major) {
    return column_major ? HbmLayout<2>{{1, rows}} : HbmLayout<2>{{dim, 1}};
}

template <typename T>
std::vector<double> reference_attention(const std::vector<T>& q,
                                        const std::vector<T>& k,
                                        const std::vector<T>& v, size_t rows,
                                        size_t keys, size_t dim,
                                        bool column_major) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    std::vector<double> out(rows * dim, 0.0);
    std::vector<double> scores(keys, 0.0);
    for (size_t row = 0; row < rows; ++row) {
        double max_score = -INFINITY;
        for (size_t key = 0; key < keys; ++key) {
            double dot = 0.0;
            for (size_t col = 0; col < dim; ++col) {
                const size_t qi = row * dim + col;
                const size_t ki = column_major ? col * keys + key : key * dim + col;
                dot += static_cast<double>(dmk::dtype::to_float(q[qi])) *
                       static_cast<double>(dmk::dtype::to_float(k[ki]));
            }
            scores[key] = dot * scale;
            max_score = scores[key] > max_score ? scores[key] : max_score;
        }
        double total = 0.0;
        for (size_t key = 0; key < keys; ++key) {
            scores[key] = std::exp(scores[key] - max_score);
            total += scores[key];
        }
        const double inverse = total != 0.0 ? 1.0 / total : 0.0;
        for (size_t key = 0; key < keys; ++key) {
            const double weight = scores[key] * inverse;
            for (size_t col = 0; col < dim; ++col) {
                const size_t vi = column_major ? col * keys + key : key * dim + col;
                out[row * dim + col] +=
                    weight * static_cast<double>(dmk::dtype::to_float(v[vi]));
            }
        }
    }
    return out;
}

// One query tile per block, one instruction per key tile.
std::vector<int> make_instructions(size_t blocks, size_t keys) {
    const size_t tiles = (keys + dmk::attention::kKeyTile - 1) /
                         dmk::attention::kKeyTile;
    std::vector<int> words(blocks * tiles * 32, 0);
    for (size_t block = 0; block < blocks; ++block) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            int* word = &words[(block * tiles + tile) * 32];
            word[0] = AttnOpF32::opcode;
            word[1] = static_cast<int>(block);
            word[2] = static_cast<int>(tile);
            word[3] = static_cast<int>(keys);
            word[4] = 0;  // patched by the caller with the head dim
            word[5] = tile + 1 == tiles ? 1 : 0;
        }
    }
    return words;
}

void set_instruction_dim(std::vector<int>& words, size_t dim) {
    for (size_t index = 0; index + 32 <= words.size(); index += 32)
        words[index + 4] = static_cast<int>(dim);
}

__global__ void attention_kernel_f32(AttnGlobalsF32 g) {
    dmk::mk::mk_internal<AttnGlobalsF32, AttnOpF32>(g);
}

__global__ void attention_kernel_f16(AttnGlobalsF16 g) {
    dmk::mk::mk_internal<AttnGlobalsF16, AttnOpF16>(g);
}

__global__ void attention_kernel_bf16(AttnGlobalsBF16 g) {
    dmk::mk::mk_internal<AttnGlobalsBF16, AttnOpBF16>(g);
}

template <typename T>
void launch_attention(typename dmk::attention::AttentionGlobalsT<T>& g,
                      unsigned blocks) {
    const dim3 grid(blocks);
    const dim3 threads(dmk::config::kThreads);
    const auto shared = static_cast<unsigned>(dmk::mk::kLaunchBytes);
    if constexpr (std::is_same<T, float>::value) {
        check_hip(hipFuncSetAttribute(
                      reinterpret_cast<const void*>(&attention_kernel_f32),
                      hipFuncAttributeMaxDynamicSharedMemorySize,
                      static_cast<int>(shared)),
                  "attribute");
        hipLaunchKernelGGL(attention_kernel_f32, grid, threads, shared, 0, g);
    } else if constexpr (std::is_same<T, __half>::value) {
        check_hip(hipFuncSetAttribute(
                      reinterpret_cast<const void*>(&attention_kernel_f16),
                      hipFuncAttributeMaxDynamicSharedMemorySize,
                      static_cast<int>(shared)),
                  "attribute");
        hipLaunchKernelGGL(attention_kernel_f16, grid, threads, shared, 0, g);
    } else {
        check_hip(hipFuncSetAttribute(
                      reinterpret_cast<const void*>(&attention_kernel_bf16),
                      hipFuncAttributeMaxDynamicSharedMemorySize,
                      static_cast<int>(shared)),
                  "attribute");
        hipLaunchKernelGGL(attention_kernel_bf16, grid, threads, shared, 0, g);
    }
    check_hip(hipGetLastError(), "launch");
}

template <typename T>
void run_case(const char* name, size_t rows, size_t keys, size_t dim,
              bool column_major, double tolerance) {
    if (rows == 0 || dim > dmk::attention::kMaxHeadDim) fail("bad case");

    std::vector<T> host_q = make_matrix<T>(rows, dim, false, 1);
    std::vector<T> host_k = make_matrix<T>(keys, dim, column_major, 2);
    std::vector<T> host_v = make_matrix<T>(keys, dim, column_major, 3);
    std::vector<T> host_out(rows * dim,
                            dmk::dtype::from_float<T>(-12345.0f));
    const std::vector<double> expected =
        reference_attention(host_q, host_k, host_v, rows, keys, dim,
                            column_major);

    const unsigned blocks =
        static_cast<unsigned>((rows + dmk::attention::kQueryTile - 1) /
                              dmk::attention::kQueryTile);
    const size_t tiles = (keys + dmk::attention::kKeyTile - 1) /
                         dmk::attention::kKeyTile;
    std::vector<int> host_words = make_instructions(blocks, keys);
    set_instruction_dim(host_words, dim);

    DeviceBuffer device_q(host_q.size() * sizeof(T));
    DeviceBuffer device_k(host_k.size() * sizeof(T));
    DeviceBuffer device_v(host_v.size() * sizeof(T));
    DeviceBuffer device_out(host_out.size() * sizeof(T));
    DeviceBuffer device_words(host_words.size() * sizeof(int));
    check_hip(hipMemcpy(device_q.data<void>(), host_q.data(),
                        host_q.size() * sizeof(T), hipMemcpyHostToDevice),
              "copy q");
    check_hip(hipMemcpy(device_k.data<void>(), host_k.data(),
                        host_k.size() * sizeof(T), hipMemcpyHostToDevice),
              "copy k");
    check_hip(hipMemcpy(device_v.data<void>(), host_v.data(),
                        host_v.size() * sizeof(T), hipMemcpyHostToDevice),
              "copy v");
    check_hip(hipMemcpy(device_out.data<void>(), host_out.data(),
                        host_out.size() * sizeof(T), hipMemcpyHostToDevice),
              "copy out");
    check_hip(hipMemcpy(device_words.data<void>(), host_words.data(),
                        host_words.size() * sizeof(int),
                        hipMemcpyHostToDevice),
              "copy instructions");

    const Shape<2> matrix_shape{{rows, dim}};
    const Shape<2> key_shape{{keys, dim}};
    const Shape<3> word_shape{{blocks, tiles, 32}};
    dmk::attention::AttentionGlobalsT<T> globals{
        Tensor<const int, 3, MemorySpace::HBM>{
            device_words.data<const int>(), word_shape,
            HbmLayout<3>::contiguous(word_shape)},
        Tensor<const T, 2, MemorySpace::HBM>{device_q.data<const T>(),
                                             matrix_shape,
                                             HbmLayout<2>{{dim, 1}}},
        Tensor<const T, 2, MemorySpace::HBM>{
            device_k.data<const T>(), key_shape,
            matrix_layout(keys, dim, column_major)},
        Tensor<const T, 2, MemorySpace::HBM>{
            device_v.data<const T>(), key_shape,
            matrix_layout(keys, dim, column_major)},
        Tensor<T, 2, MemorySpace::HBM>{device_out.data<T>(), matrix_shape,
                                       HbmLayout<2>{{dim, 1}}},
        static_cast<int>(tiles),
    };
    if (!globals.valid(blocks)) fail("globals rejected");

    launch_attention<T>(globals, blocks);
    check_hip(hipDeviceSynchronize(), "synchronize");
    check_hip(hipMemcpy(host_out.data(), device_out.data<void>(),
                        host_out.size() * sizeof(T), hipMemcpyDeviceToHost),
              "copy result");

    double max_reference = 0.0;
    double max_difference = 0.0;
    for (size_t i = 0; i < host_out.size(); ++i) {
        const double difference =
            std::fabs(static_cast<double>(dmk::dtype::to_float(host_out[i])) -
                      expected[i]);
        max_reference = std::max(max_reference, std::fabs(expected[i]));
        max_difference = std::max(max_difference, difference);
    }
    const double budget = tolerance * std::max(1.0, max_reference);
    const bool passed = max_difference <= budget;
    std::printf("%-26s %-4s rows=%3zu keys=%4zu dim=%3zu %s  max|ref|=%.4f "
                "max|diff|=%.2e\n",
                name, dtype_name<T>(), rows, keys, dim,
                column_major ? "col" : "row", max_reference, max_difference);
    if (!passed) {
        std::fprintf(stderr, "  attention mismatch: budget %.2e\n", budget);
        fail(name);
    }
}

// Same launch path without the reference: reports kernel time and throughput
// (QK^T plus P*V).
void time_case(const char* name, size_t rows, size_t keys, size_t dim,
               int repeats) {
    std::vector<float> host_q = make_matrix<float>(rows, dim, false, 1);
    std::vector<float> host_k = make_matrix<float>(keys, dim, false, 2);
    std::vector<float> host_v = make_matrix<float>(keys, dim, false, 3);
    std::vector<float> host_out(rows * dim, 0.0f);

    const unsigned blocks =
        static_cast<unsigned>((rows + dmk::attention::kQueryTile - 1) /
                              dmk::attention::kQueryTile);
    const size_t tiles = (keys + dmk::attention::kKeyTile - 1) /
                         dmk::attention::kKeyTile;
    std::vector<int> host_words = make_instructions(blocks, keys);
    set_instruction_dim(host_words, dim);

    DeviceBuffer device_q(host_q.size() * sizeof(float));
    DeviceBuffer device_k(host_k.size() * sizeof(float));
    DeviceBuffer device_v(host_v.size() * sizeof(float));
    DeviceBuffer device_out(host_out.size() * sizeof(float));
    DeviceBuffer device_words(host_words.size() * sizeof(int));
    check_hip(hipMemcpy(device_q.data<void>(), host_q.data(),
                        host_q.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy q");
    check_hip(hipMemcpy(device_k.data<void>(), host_k.data(),
                        host_k.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy k");
    check_hip(hipMemcpy(device_v.data<void>(), host_v.data(),
                        host_v.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy v");
    check_hip(hipMemcpy(device_out.data<void>(), host_out.data(),
                        host_out.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy out");
    check_hip(hipMemcpy(device_words.data<void>(), host_words.data(),
                        host_words.size() * sizeof(int),
                        hipMemcpyHostToDevice),
              "copy instructions");

    const Shape<2> matrix_shape{{rows, dim}};
    const Shape<2> key_shape{{keys, dim}};
    const Shape<3> word_shape{{blocks, tiles, 32}};
    AttnGlobalsF32 globals{
        Tensor<const int, 3, MemorySpace::HBM>{
            device_words.data<const int>(), word_shape,
            HbmLayout<3>::contiguous(word_shape)},
        Tensor<const float, 2, MemorySpace::HBM>{
            device_q.data<const float>(), matrix_shape,
            HbmLayout<2>{{dim, 1}}},
        Tensor<const float, 2, MemorySpace::HBM>{
            device_k.data<const float>(), key_shape,
            HbmLayout<2>{{dim, 1}}},
        Tensor<const float, 2, MemorySpace::HBM>{
            device_v.data<const float>(), key_shape,
            HbmLayout<2>{{dim, 1}}},
        Tensor<float, 2, MemorySpace::HBM>{device_out.data<float>(),
                                           matrix_shape,
                                           HbmLayout<2>{{dim, 1}}},
        static_cast<int>(tiles),
    };

    launch_attention<float>(globals, blocks);
    check_hip(hipDeviceSynchronize(), "warm-up");

    Event start;
    Event stop;
    check_hip(hipEventRecord(start.event), "start");
    for (int repeat = 0; repeat < repeats; ++repeat)
        launch_attention<float>(globals, blocks);
    check_hip(hipEventRecord(stop.event), "stop");
    check_hip(hipEventSynchronize(stop.event), "synchronize");

    float total_ms = 0.0f;
    check_hip(hipEventElapsedTime(&total_ms, start.event, stop.event),
              "elapsed");
    const double flops =
        4.0 * static_cast<double>(rows) * keys * dim * repeats;
    const double seconds = static_cast<double>(total_ms) * 1e-3;
    std::printf("%-26s %7.2f ms  %8.1f GFLOP/s\n", name,
                total_ms / static_cast<float>(repeats),
                seconds > 0.0 ? flops / seconds / 1e9 : 0.0);
}

}  // namespace

int main() {
    std::printf("attention operator (page-based, standalone op pack)\n");
    run_case<float>("one query tile 16x64x64", 16, 64, 64, false, 3e-4);
    run_case<float>("partial rows 3x37x32", 3, 37, 32, false, 3e-4);
    run_case<float>("single row and key", 1, 1, 128, false, 3e-4);
    run_case<float>("cross-tile 17x129x96", 17, 129, 96, false, 3e-4);
    run_case<float>("odd shapes 5x129x17", 5, 129, 17, false, 3e-4);
    run_case<float>("head dim 1", 2, 9, 1, false, 3e-4);
    run_case<float>("transposed K/V", 4, 50, 96, true, 3e-4);
    run_case<float>("long keys 16x512x64", 16, 512, 64, false, 3e-4);
    run_case<float>("64 rows 64x1024x64", 64, 1024, 64, false, 3e-4);
    run_case<float>("128 rows 128x256x64", 128, 256, 64, false, 3e-4);
    run_case<float>("head dim 128 32x512x128", 32, 512, 128, false, 3e-4);

    run_case<__half>("dense 16x64x64 f16", 16, 64, 64, false, 8e-3);
    run_case<__half>("transposed K/V f16", 4, 50, 96, true, 8e-3);
    run_case<__half>("64x1024x64 f16", 64, 1024, 64, false, 8e-3);
    run_case<__hip_bfloat16>("dense 16x64x64 bf16", 16, 64, 64, false, 3e-2);
    run_case<__hip_bfloat16>("transposed K/V bf16", 4, 50, 96, true, 3e-2);
    run_case<__hip_bfloat16>("64x1024x64 bf16", 64, 1024, 64, false, 3e-2);

    time_case("128x2048x64 timing", 128, 2048, 64, 5);

    std::printf("attention checks passed\n");
    return 0;
}
