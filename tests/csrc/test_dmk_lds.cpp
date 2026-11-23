// Host-side logic test for the DCU megakernel LDS layout and page protocol.
// It needs no DCU: it checks the static/dynamic LDS budget, page/semaphore
// layout, the mbarrier-compatible phase arithmetic, the page permutation and
// the globals contract. The device test runs the real instruction stream.
//
// Build: hipcc -O2 -std=c++17 -I csrc/dmk tests/csrc/test_dmk_lds.cpp -o /tmp/test_dmk_lds
#include "config.h"
#include "copy.h"
#include "lds.h"
#include "megakernel.h"
#include "tile.h"
#include "state.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <type_traits>

namespace {

using namespace dmk;
namespace config = dmk::config;
using dmk::mk::InstructionState;
using dmk::mk::kDynamicBytes;
using dmk::mk::kLaunchBytes;
using dmk::mk::kStaticBytes;
using dmk::mk::MkGlobals;
using dmk::mk::Page;
using dmk::mk::remap_page_order;
using dmk::tensor::HbmLayout;
using dmk::tensor::MemorySpace;
using dmk::tensor::Shape;
using dmk::tensor::Tensor;

int failures = 0;

void check(bool condition, const char* expression, int line) {
    if (condition) return;
    std::fprintf(stderr, "FAIL line %d: %s\n", line, expression);
    ++failures;
}

#define CHECK(expression) check((expression), #expression, __LINE__)

void test_config_budget() {
    CHECK(config::kStages == 2);
    CHECK(config::kStageBits == 1);
    CHECK(config::kInstructionWidth == 32);
    CHECK(config::kConsumerWaves == 4);
    CHECK(config::kServiceWaves == 3);
    CHECK(config::kWorkerWaves == 6);
    CHECK(config::kTotalWaves == 7);
    CHECK(config::kThreads == 448);
    CHECK(config::kPageSize == 8192);
    CHECK(config::kNumPages == 6);

    CHECK(kDynamicBytes == 6 * 8192);
    CHECK(kLaunchBytes == kDynamicBytes + config::kDynamicAlign);
    CHECK(kStaticBytes ==
          sizeof(InstructionState) * config::kStages +
              sizeof(lds::Semaphore) *
                  (2 * config::kStages + 1 +
                   config::kNumPages * config::kStageBits));
    CHECK(kStaticBytes + kLaunchBytes <= config::kLdsBytesPerCu);

    CHECK(sizeof(lds::Semaphore) == 16);
    CHECK(alignof(lds::Semaphore) == 16);
    CHECK(sizeof(InstructionState) % 128 == 0);
    CHECK(sizeof(Page) == config::kPageSize);
    CHECK(alignof(Page) == 128);
    CHECK(std::is_trivially_destructible<InstructionState>::value);
}

void test_phase_arithmetic() {
    // mbarrier parity: after `init(N); arrive(N);` phase 0 is complete and the
    // current phase parity becomes 1.
    CHECK(dmk::lds::phase_for_ticket(0, 1) == 0);
    CHECK(dmk::lds::phase_for_ticket(1, 1) == 1);
    CHECK(dmk::lds::phase_for_ticket(2, 1) == 0);
    CHECK(dmk::lds::phase_for_ticket(3, 1) == 1);
    CHECK(dmk::lds::phase_for_ticket(0, 4) == 0);
    CHECK(dmk::lds::phase_for_ticket(3, 4) == 0);
    CHECK(dmk::lds::phase_for_ticket(4, 4) == 1);
    CHECK(dmk::lds::phase_for_ticket(7, 4) == 1);
    CHECK(dmk::lds::phase_for_ticket(8, 4) == 0);
    CHECK(dmk::lds::phase_for_ticket(12, 4) == 1);
    CHECK(dmk::lds::phase_for_ticket(42, 0) == 0);
}

void test_page_permutation() {
    constexpr int kPages = config::kNumPages;
    const int previous[kPages] = {2, 4, 0, 5, 1, 3};
    const int identity[kPages] = {0, 1, 2, 3, 4, 5};
    for (int query = 0; query < kPages; ++query) {
        CHECK(remap_page_order(previous, identity[query]) == previous[query]);
    }

    // A rotation of logical page ids rotates the physical assignment.
    const int rotate[kPages] = {1, 2, 3, 4, 5, 0};
    const int expected[kPages] = {4, 0, 5, 1, 3, 2};
    for (int query = 0; query < kPages; ++query) {
        CHECK(remap_page_order(previous, rotate[query]) == expected[query]);
    }

    // Identity page order is what the demo op uses.
    for (int query = 0; query < kPages; ++query) {
        CHECK(remap_page_order(identity, query) == query);
    }

    CHECK(dmk::mk::ring_advance<config::kStages>(0) == 1);
    CHECK(dmk::mk::ring_advance<config::kStages>(1) == 0);
    CHECK(dmk::mk::ring_retreat<config::kStages>(0) == 1);
    CHECK(dmk::mk::ring_retreat<config::kStages>(1) == 0);
}

void test_vector_access() {
    constexpr size_t kCount = 1000;
    constexpr size_t kVector = sizeof(float4) / sizeof(float);
    constexpr size_t kWaveStride = 64 * kVector;  // config::kWaveSize * kVector

    alignas(16) float src[kCount];
    alignas(16) float dst[kCount];
    for (size_t i = 0; i < kCount; ++i) {
        src[i] = static_cast<float>(i) * 0.5f;
        dst[i] = -1.0f;
    }

    CHECK(dmk::copy::aligned<float4>(src));
    CHECK(dmk::copy::aligned<float2>(src + 2));
    CHECK(!dmk::copy::aligned<float4>(src + 1));
    CHECK(!dmk::copy::aligned<float4>(src + 3));

    // Exactly the loop an op's load()/store() runs: lane i owns the 16 B chunk
    // starting at element i * kVector, and the tail below kVector * vectors is
    // left for the scalar path.
    const size_t vector_end = kCount / kVector * kVector;
    for (size_t lane = 0; lane < 64; ++lane) {
        for (size_t element = lane * kVector; element < vector_end;
             element += kWaveStride) {
            dmk::copy::store_vec<float4>(
                dst + element, dmk::copy::load_vec<float4>(src + element));
        }
    }
    for (size_t i = 0; i < vector_end; ++i) CHECK(dst[i] == src[i]);
    for (size_t i = vector_end; i < kCount; ++i) CHECK(dst[i] == -1.0f);

    // The narrow vector sees the same data with a different width.
    alignas(16) float narrow[kCount];
    for (size_t i = 0; i < kCount; ++i) narrow[i] = 0.0f;
    const size_t narrow_end = kCount / 2 * 2;
    for (size_t element = 0; element < narrow_end; element += 2) {
        dmk::copy::store_vec<float2>(narrow + element,
                                     dmk::copy::load_vec<float2>(src + element));
    }
    for (size_t i = 0; i < narrow_end; ++i) CHECK(narrow[i] == src[i]);

    // Zero fill with the same primitive.
    alignas(16) float page[256];
    for (float& value : page) value = -1.0f;
    const float4 zero{0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t element = 3; element + kVector <= 256; element += kVector)
        dmk::copy::store_vec<float4>(page + element, zero);
    for (size_t i = 0; i < 3; ++i) CHECK(page[i] == -1.0f);
    for (size_t i = 3; i < 255; ++i) CHECK(page[i] == 0.0f);
}

// Tile views: a tile is a logical block (tile_m x tile_n, or tile_b x tile_m
// x tile_n for rank 3), not a contiguous run of storage, so the tests check
// the clamped shapes, the copy result and the zero fill of the padded tail.
void test_tile_views() {
    using namespace dmk;
    constexpr size_t kRows = 5;
    constexpr size_t kCols = 7;
    auto value = [](size_t row, size_t col) {
        return static_cast<float>(row * 10 + col);
    };

    // 2D, row-major.
    float row_major[kRows * kCols];
    for (size_t row = 0; row < kRows; ++row)
        for (size_t col = 0; col < kCols; ++col)
            row_major[row * kCols + col] = value(row, col);
    const tensor::Shape<2> shape2{{kRows, kCols}};
    const tensor::Tensor<const float, 2, tensor::MemorySpace::HBM> matrix{
        row_major, shape2, tensor::HbmLayout<2>::contiguous(shape2)};
    const tile::TileShape<2> tile2{{2, 3}};
    CHECK(tile::tile_count(shape2, tile2, 0) == 3);
    CHECK(tile::tile_count(shape2, tile2, 1) == 3);

    alignas(16) float buffer[2 * 3];
    for (size_t tile_row = 0; tile_row < 3; ++tile_row) {
        for (size_t tile_col = 0; tile_col < 3; ++tile_col) {
            const size_t coord[2] = {tile_row, tile_col};
            const auto source = tile::make_tile(matrix, tile2, coord);
            CHECK(source.shape.dims[0] ==
                  std::min<size_t>(2, kRows - tile_row * 2));
            CHECK(source.shape.dims[1] ==
                  std::min<size_t>(3, kCols - tile_col * 3));
            const auto destination = tile::make_lds_view(buffer, source.shape);
            for (size_t worker = 0; worker < 4; ++worker)
                CHECK(tile::copy_tile(destination, source, worker, 4));
            bool same = true;
            for (size_t row = 0; row < source.shape.dims[0]; ++row)
                for (size_t col = 0; col < source.shape.dims[1]; ++col)
                    same = same && buffer[row * source.shape.dims[1] + col] ==
                                       value(tile_row * 2 + row,
                                             tile_col * 3 + col);
            CHECK(same);
        }
    }

    // A tile past the end is empty and copies nothing.
    const size_t outside[2] = {3, 0};
    CHECK(tile::make_tile(matrix, tile2, outside).numel() == 0);

    // load_tile zero-fills the full destination, so the padded rows of an
    // edge tile read as zero.
    alignas(16) float padded[2 * 3];
    for (float& element : padded) element = -1.0f;
    const size_t edge[2] = {2, 2};  // rows 4..5, cols 6..8 -> 1 x 1
    const auto edge_source = tile::make_tile(matrix, tile2, edge);
    const auto edge_destination =
        tile::make_lds_view(padded, tensor::Shape<2>{{2, 3}});
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(tile::load_tile(edge_destination, edge_source, worker, 4));
    CHECK(padded[0] == value(4, 6));
    for (size_t i = 1; i < 6; ++i) CHECK(padded[i] == 0.0f);

    // 2D, column-major: the tile keeps the parent strides, so the copy must
    // see the same logical block.
    float column_major[kRows * kCols];
    for (size_t row = 0; row < kRows; ++row)
        for (size_t col = 0; col < kCols; ++col)
            column_major[col * kRows + row] = value(row, col);
    const tensor::Tensor<const float, 2, tensor::MemorySpace::HBM> transposed{
        column_major, shape2, tensor::HbmLayout<2>{{1, kRows}}};
    const size_t coord[2] = {1, 1};
    const auto source = tile::make_tile(transposed, tile2, coord);
    CHECK(source.shape.dims[0] == 2 && source.shape.dims[1] == 3);
    const auto destination = tile::make_lds_view(buffer, source.shape);
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(tile::copy_tile(destination, source, worker, 4));
    bool same = true;
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col)
            same = same && buffer[row * 3 + col] == value(2 + row, 3 + col);
    CHECK(same);

    // make_tile_at: a block at an arbitrary origin, which is what instruction
    // words carry (e.g. a GEMM tile at (m0, n0) that need not be 16-aligned).
    const size_t origin[2] = {1, 2};
    const auto block = tile::make_tile_at(matrix, tile2, origin);
    CHECK(block.shape.dims[0] == 2 && block.shape.dims[1] == 3);
    const auto block_destination = tile::make_lds_view(buffer, block.shape);
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(tile::copy_tile(block_destination, block, worker, 4));
    same = true;
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col)
            same = same && buffer[row * 3 + col] == value(1 + row, 2 + col);
    CHECK(same);

    // 3D, batched: [2, 5, 7] with tile [1, 2, 3].
    float batched[2 * kRows * kCols];
    for (size_t batch = 0; batch < 2; ++batch)
        for (size_t row = 0; row < kRows; ++row)
            for (size_t col = 0; col < kCols; ++col)
                batched[(batch * kRows + row) * kCols + col] =
                    value(row, col) + static_cast<float>(batch) * 100.0f;
    const tensor::Shape<3> shape3{{2, kRows, kCols}};
    const tensor::Tensor<const float, 3, tensor::MemorySpace::HBM> batch{
        batched, shape3, tensor::HbmLayout<3>::contiguous(shape3)};
    const tile::TileShape<3> tile3{{1, 2, 3}};
    CHECK(tile::tile_count(shape3, tile3, 0) == 2);
    CHECK(tile::tile_count(shape3, tile3, 1) == 3);
    CHECK(tile::tile_count(shape3, tile3, 2) == 3);

    alignas(16) float tile3_buffer[1 * 2 * 3];
    const size_t coord3[3] = {1, 2, 1};
    const auto source3 = tile::make_tile(batch, tile3, coord3);
    CHECK(source3.shape.dims[0] == 1 && source3.shape.dims[1] == 1 &&
          source3.shape.dims[2] == 3);
    const auto destination3 = tile::make_lds_view(tile3_buffer, source3.shape);
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(tile::copy_tile(destination3, source3, worker, 4));
    same = true;
    for (size_t col = 0; col < 3; ++col)
        same = same && tile3_buffer[col] == value(4, 3 + col) + 100.0f;
    CHECK(same);
}

void test_globals_contract() {
    const Shape<3> instruction_shape{{2, 4, 32}};
    const Shape<1> data_shape{{2048}};
    const HbmLayout<1> data_layout = HbmLayout<1>::contiguous(data_shape);
    float input[2048]{};
    float output[2048]{};
    int instructions[2 * 4 * 32]{};

    const MkGlobals good{
        Tensor<const int, 3, MemorySpace::HBM>{
            instructions, instruction_shape,
            HbmLayout<3>::contiguous(instruction_shape)},
        Tensor<const float, 1, MemorySpace::HBM>{input, data_shape,
                                                 data_layout},
        Tensor<float, 1, MemorySpace::HBM>{output, data_shape, data_layout},
        4,
    };
    CHECK(good.valid(2));
    CHECK(!good.valid(3));  // fewer than `blocks` instruction rows

    MkGlobals wrong_width = good;
    wrong_width.instructions.shape.dims[2] = 16;
    CHECK(!wrong_width.valid(2));

    MkGlobals too_few_instructions = good;
    too_few_instructions.num_instructions = 5;
    CHECK(!too_few_instructions.valid(2));

    MkGlobals strided = good;
    strided.output.layout = HbmLayout<1>{{2}};
    CHECK(!strided.valid(2));

    MkGlobals empty = good;
    empty.num_instructions = 0;
    CHECK(!empty.valid(2));

    MkGlobals no_instructions = good;
    no_instructions.instructions.data = nullptr;
    CHECK(!no_instructions.valid(2));
}

}  // namespace

int main() {
    test_config_budget();
    test_phase_arithmetic();
    test_page_permutation();
    test_vector_access();
    test_tile_views();
    test_globals_contract();
    if (failures != 0) {
        std::fprintf(stderr, "%d LDS host check(s) failed\n", failures);
        return 1;
    }
    std::printf("LDS host checks passed\n");
    return 0;
}
