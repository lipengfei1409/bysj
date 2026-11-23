// Tile view tests: tile math and tile copies on the host, plus a device round
// trip that stages 16x16 (rank-2) and batched 1x16x16 (rank-3) tiles through
// LDS. The device part is skipped when no DCU is visible, so the host checks
// run anywhere.
//
// Build/run through: python tests/csrc/run_dmk_ops_test.py --op tile
#include "tile.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using dmk::tile::TileShape;
using dmk::tile::TileView;
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

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
}

float value(size_t row, size_t col) {
    return static_cast<float>(row * 10 + col);
}

// ---------------------------------------------------------------- host ----

void test_tile_counts() {
    const Shape<2> shape2{{5, 7}};
    const TileShape<2> tile2{{2, 3}};
    CHECK(dmk::tile::tile_count(shape2, tile2, 0) == 3);
    CHECK(dmk::tile::tile_count(shape2, tile2, 1) == 3);

    const Shape<3> shape3{{2, 5, 7}};
    const TileShape<3> tile3{{1, 2, 3}};
    CHECK(dmk::tile::tile_count(shape3, tile3, 0) == 2);
    CHECK(dmk::tile::tile_count(shape3, tile3, 1) == 3);
    CHECK(dmk::tile::tile_count(shape3, tile3, 2) == 3);

    // An exact fit needs no extra tile; a zero-sized axis has none.
    const Shape<2> exact{{4, 6}};
    CHECK(dmk::tile::tile_count(exact, tile2, 0) == 2);
    CHECK(dmk::tile::tile_count(exact, tile2, 1) == 2);
    const Shape<2> empty{{0, 7}};
    CHECK(dmk::tile::tile_count(empty, tile2, 0) == 0);
}

void test_make_tile_grid() {
    constexpr size_t kRows = 5;
    constexpr size_t kCols = 7;
    float data[kRows * kCols];
    for (size_t row = 0; row < kRows; ++row)
        for (size_t col = 0; col < kCols; ++col)
            data[row * kCols + col] = value(row, col);
    const Shape<2> shape{{kRows, kCols}};
    const Tensor<const float, 2, MemorySpace::HBM> matrix{
        data, shape, HbmLayout<2>::contiguous(shape)};
    const TileShape<2> tile{{2, 3}};

    // Grid tiles clamp at the edge: row 4 and column 6 are the last ones.
    for (size_t tile_row = 0; tile_row < 3; ++tile_row) {
        for (size_t tile_col = 0; tile_col < 3; ++tile_col) {
            const size_t coord[2] = {tile_row, tile_col};
            const auto view = dmk::tile::make_tile(matrix, tile, coord);
            CHECK(view.shape.dims[0] ==
                  std::min<size_t>(2, kRows - tile_row * 2));
            CHECK(view.shape.dims[1] ==
                  std::min<size_t>(3, kCols - tile_col * 3));
            CHECK(view.data == data + (tile_row * 2) * kCols + tile_col * 3);
        }
    }

    // Past the last tile the view is empty, so a copy has nothing to do.
    const size_t outside[2] = {3, 0};
    CHECK(dmk::tile::make_tile(matrix, tile, outside).numel() == 0);
}

void test_make_tile_at() {
    constexpr size_t kRows = 5;
    constexpr size_t kCols = 7;
    float data[kRows * kCols];
    for (size_t row = 0; row < kRows; ++row)
        for (size_t col = 0; col < kCols; ++col)
            data[row * kCols + col] = value(row, col);
    const Shape<2> shape{{kRows, kCols}};
    const Tensor<const float, 2, MemorySpace::HBM> matrix{
        data, shape, HbmLayout<2>::contiguous(shape)};
    const TileShape<2> tile{{2, 3}};

    // Instruction coordinates need not be tile-aligned.
    const size_t origin[2] = {1, 2};
    const auto block = dmk::tile::make_tile_at(matrix, tile, origin);
    CHECK(block.shape.dims[0] == 2 && block.shape.dims[1] == 3);
    CHECK(block.data == data + 1 * kCols + 2);
    CHECK(block.data[0] == value(1, 2));
    CHECK(block.data[block.offset({0, 2})] == value(1, 4));

    // The edge clamps, and a coordinate past the end is empty.
    const size_t edge[2] = {3, 5};
    const auto clamped = dmk::tile::make_tile_at(matrix, tile, edge);
    CHECK(clamped.shape.dims[0] == 2 && clamped.shape.dims[1] == 2);
    const size_t outside[2] = {5, 0};
    CHECK(dmk::tile::make_tile_at(matrix, tile, outside).numel() == 0);
}

void test_copy_layouts() {
    constexpr size_t kRows = 5;
    constexpr size_t kCols = 7;
    alignas(16) float row_major[kRows * kCols];
    for (size_t row = 0; row < kRows; ++row)
        for (size_t col = 0; col < kCols; ++col)
            row_major[row * kCols + col] = value(row, col);
    const Shape<2> shape{{kRows, kCols}};
    const TileShape<2> tile{{2, 3}};
    const size_t origin[2] = {1, 1};
    alignas(16) float buffer[2 * 3];

    // Row-major source.
    const Tensor<const float, 2, MemorySpace::HBM> contiguous{
        row_major, shape, HbmLayout<2>::contiguous(shape)};
    const auto source = dmk::tile::make_tile_at(contiguous, tile, origin);
    const auto destination = dmk::tile::make_lds_view(buffer, source.shape);
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(dmk::tile::copy_tile(destination, source, worker, 4));
    bool same = true;
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col)
            same = same && buffer[row * 3 + col] == value(1 + row, 1 + col);
    CHECK(same);

    // Column-major source: the tile keeps the parent strides.
    std::vector<float> column_major(kRows * kCols);
    for (size_t row = 0; row < kRows; ++row)
        for (size_t col = 0; col < kCols; ++col)
            column_major[col * kRows + row] = value(row, col);
    const Tensor<const float, 2, MemorySpace::HBM> transposed{
        column_major.data(), shape, HbmLayout<2>{{1, kRows}}};
    const auto column_source = dmk::tile::make_tile_at(transposed, tile, origin);
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(dmk::tile::copy_tile(destination, column_source, worker, 4));
    same = true;
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col)
            same = same && buffer[row * 3 + col] == value(1 + row, 1 + col);
    CHECK(same);

    // Padded rows: stride 9 for a 7-column tensor.
    std::vector<float> padded(kRows * 9, -1.0f);
    for (size_t row = 0; row < kRows; ++row)
        for (size_t col = 0; col < kCols; ++col)
            padded[row * 9 + col] = value(row, col);
    const Tensor<const float, 2, MemorySpace::HBM> strided{
        padded.data(), shape, HbmLayout<2>{{9, 1}}};
    const auto strided_source = dmk::tile::make_tile_at(strided, tile, origin);
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(dmk::tile::copy_tile(destination, strided_source, worker, 4));
    same = true;
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col)
            same = same && buffer[row * 3 + col] == value(1 + row, 1 + col);
    CHECK(same);
}

void test_load_tile() {
    constexpr size_t kRows = 5;
    constexpr size_t kCols = 7;
    alignas(16) float data[kRows * kCols];
    for (size_t row = 0; row < kRows; ++row)
        for (size_t col = 0; col < kCols; ++col)
            data[row * kCols + col] = value(row, col);
    const Shape<2> shape{{kRows, kCols}};
    const Tensor<const float, 2, MemorySpace::HBM> matrix{
        data, shape, HbmLayout<2>::contiguous(shape)};
    const TileShape<2> tile{{2, 3}};

    // An edge tile (1x1) fills the rest of the destination with zeros.
    alignas(16) float padded[2 * 3];
    for (float& element : padded) element = -1.0f;
    const size_t edge[2] = {4, 6};
    const auto source = dmk::tile::make_tile_at(matrix, tile, edge);
    const auto destination =
        dmk::tile::make_lds_view(padded, Shape<2>{{2, 3}});
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(dmk::tile::load_tile(destination, source, worker, 4));
    CHECK(padded[0] == value(4, 6));
    for (size_t index = 1; index < 6; ++index) CHECK(padded[index] == 0.0f);

    // An empty source leaves the destination fully zeroed.
    const size_t outside[2] = {5, 0};
    const auto empty = dmk::tile::make_tile_at(matrix, tile, outside);
    for (float& element : padded) element = -1.0f;
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(dmk::tile::load_tile(destination, empty, worker, 4));
    for (float element : padded) CHECK(element == 0.0f);

    // A full tile is a plain copy and keeps the source values.
    const size_t inside[2] = {0, 0};
    const auto full = dmk::tile::make_tile_at(matrix, tile, inside);
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(dmk::tile::load_tile(destination, full, worker, 4));
    bool same = true;
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col)
            same = same && padded[row * 3 + col] == value(row, col);
    CHECK(same);

    // Every worker covers the tile exactly once: run one worker at a time
    // into separate buffers and compare the union.
    alignas(16) float per_worker[4][2 * 3];
    for (size_t worker = 0; worker < 4; ++worker) {
        for (float& element : per_worker[worker]) element = -2.0f;
        const auto destination_w = dmk::tile::make_lds_view(
            per_worker[worker], Shape<2>{{2, 3}});
        CHECK(dmk::tile::load_tile(destination_w, full, worker, 4));
    }
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col) {
            float found = -2.0f;
            for (size_t worker = 0; worker < 4; ++worker)
                if (per_worker[worker][row * 3 + col] != -2.0f)
                    found = per_worker[worker][row * 3 + col];
            CHECK(found == value(row, col));
        }
}

void test_rank3() {
    constexpr size_t kBatches = 2;
    constexpr size_t kRows = 5;
    constexpr size_t kCols = 7;
    std::vector<float> data(kBatches * kRows * kCols);
    for (size_t batch = 0; batch < kBatches; ++batch)
        for (size_t row = 0; row < kRows; ++row)
            for (size_t col = 0; col < kCols; ++col)
                data[(batch * kRows + row) * kCols + col] =
                    value(row, col) + 100.0f * static_cast<float>(batch);
    const Shape<3> shape{{kBatches, kRows, kCols}};
    const Tensor<const float, 3, MemorySpace::HBM> batched{
        data.data(), shape, HbmLayout<3>::contiguous(shape)};
    const TileShape<3> tile{{1, 2, 3}};

    // A tile at (batch 1, rows 2..3, cols 3..5).
    const size_t origin[3] = {1, 2, 3};
    const auto source = dmk::tile::make_tile_at(batched, tile, origin);
    CHECK(source.shape.dims[0] == 1 && source.shape.dims[1] == 2 &&
          source.shape.dims[2] == 3);
    alignas(16) float buffer[1 * 2 * 3];
    const auto destination =
        dmk::tile::make_lds_view(buffer, Shape<3>{{1, 2, 3}});
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(dmk::tile::copy_tile(destination, source, worker, 4));
    bool same = true;
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col)
            same = same && buffer[row * 3 + col] ==
                               value(2 + row, 3 + col) + 100.0f;
    CHECK(same);
}

void test_misaligned_fallback() {
    // A 16 B-aligned source offset by one float forces the scalar path; the
    // copy must still be correct.
    constexpr size_t kRows = 4;
    constexpr size_t kCols = 8;
    alignas(16) float storage[kRows * kCols + 1];
    for (size_t index = 0; index < kRows * kCols + 1; ++index)
        storage[index] = static_cast<float>(index);
    const Shape<2> shape{{kRows, kCols}};
    const Tensor<const float, 2, MemorySpace::HBM> matrix{
        storage + 1, shape, HbmLayout<2>::contiguous(shape)};
    const TileShape<2> tile{{2, 4}};
    const size_t origin[2] = {1, 2};
    const auto source = dmk::tile::make_tile_at(matrix, tile, origin);
    alignas(16) float buffer[2 * 4];
    const auto destination =
        dmk::tile::make_lds_view(buffer, Shape<2>{{2, 4}});
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(dmk::tile::copy_tile(destination, source, worker, 4));
    bool same = true;
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 4; ++col)
            same = same &&
                   buffer[row * 4 + col] == storage[1 + (1 + row) * kCols +
                                                      (2 + col)];
    CHECK(same);
}

// -------------------------------------------------------------- device ----

__global__ void roundtrip_2d_kernel(Tensor<const float, 2, MemorySpace::HBM> src,
                                    Tensor<float, 2, MemorySpace::HBM> dst,
                                    size_t tiles_m, size_t tiles_n) {
    __shared__ alignas(16) float page[16 * 16];
    const size_t lane = threadIdx.x;
    const size_t workers = blockDim.x;
    const TileShape<2> tile{{16, 16}};
    for (size_t index = blockIdx.x; index < tiles_m * tiles_n;
         index += gridDim.x) {
        const size_t origin[2] = {(index / tiles_n) * 16,
                                  (index % tiles_n) * 16};
        const auto source = dmk::tile::make_tile_at(src, tile, origin);
        const auto page_view =
            dmk::tile::make_lds_view(page, Shape<2>{{16, 16}});
        dmk::tile::load_tile(page_view, source, lane, workers);
        __syncthreads();
        const auto destination = dmk::tile::make_tile_at(dst, tile, origin);
        const auto page_part = dmk::tile::make_lds_view(
            page, destination.shape,
            dmk::tensor::StridedLayout<2>{{16, 1}});
        dmk::tile::copy_tile(destination, page_part, lane, workers);
        __syncthreads();
    }
}

__global__ void roundtrip_3d_kernel(Tensor<const float, 3, MemorySpace::HBM> src,
                                    Tensor<float, 3, MemorySpace::HBM> dst,
                                    size_t tiles_m, size_t tiles_n) {
    __shared__ alignas(16) float page[16 * 16];
    const size_t lane = threadIdx.x;
    const size_t workers = blockDim.x;
    const TileShape<3> tile{{1, 16, 16}};
    for (size_t local = blockIdx.x; local < tiles_m * tiles_n;
         local += gridDim.x) {
        const size_t origin[3] = {blockIdx.y, (local / tiles_n) * 16,
                                  (local % tiles_n) * 16};
        const auto source = dmk::tile::make_tile_at(src, tile, origin);
        const auto page_view =
            dmk::tile::make_lds_view(page, Shape<3>{{1, 16, 16}});
        dmk::tile::load_tile(page_view, source, lane, workers);
        __syncthreads();
        const auto destination = dmk::tile::make_tile_at(dst, tile, origin);
        const auto page_part = dmk::tile::make_lds_view(
            page, destination.shape,
            dmk::tensor::StridedLayout<3>{{256, 16, 1}});
        dmk::tile::copy_tile(destination, page_part, lane, workers);
        __syncthreads();
    }
}

bool has_device() {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

void run_device_checks() {
    if (!has_device()) {
        std::printf("tile: no DCU visible, device checks skipped\n");
        return;
    }

    constexpr size_t kRows = 35;
    constexpr size_t kCols = 29;
    std::vector<float> host_src(kRows * kCols);
    for (size_t row = 0; row < kRows; ++row)
        for (size_t col = 0; col < kCols; ++col)
            host_src[row * kCols + col] = value(row, col);
    std::vector<float> host_dst(kRows * kCols, -1.0f);

    float* device_src = nullptr;
    float* device_dst = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&device_src),
                  host_src.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&device_dst),
                  host_dst.size() * sizeof(float)) != hipSuccess) {
        fail("hipMalloc");
        return;
    }
    hipMemcpy(device_src, host_src.data(), host_src.size() * sizeof(float),
              hipMemcpyHostToDevice);
    hipMemcpy(device_dst, host_dst.data(), host_dst.size() * sizeof(float),
              hipMemcpyHostToDevice);

    const Shape<2> shape{{kRows, kCols}};
    const Shape<3> batch_shape{{2, kRows, kCols}};
    const size_t tiles_m = (kRows + 15) / 16;
    const size_t tiles_n = (kCols + 15) / 16;

    // Rank-2 round trip: HBM tile -> LDS page -> HBM tile.
    hipLaunchKernelGGL(
        roundtrip_2d_kernel, dim3(static_cast<unsigned>(tiles_m * tiles_n)),
        dim3(64), 0, 0,
        Tensor<const float, 2, MemorySpace::HBM>{
            device_src, shape, HbmLayout<2>::contiguous(shape)},
        Tensor<float, 2, MemorySpace::HBM>{
            device_dst, shape, HbmLayout<2>::contiguous(shape)},
        tiles_m, tiles_n);
    hipDeviceSynchronize();
    hipMemcpy(host_dst.data(), device_dst, host_dst.size() * sizeof(float),
              hipMemcpyDeviceToHost);
    CHECK(host_dst == host_src);

    // Rank-3 round trip with one tile per batch.
    std::vector<float> host_batch(2 * kRows * kCols);
    for (size_t index = 0; index < host_batch.size(); ++index)
        host_batch[index] = static_cast<float>(index % 97) + 0.5f;
    std::vector<float> host_batch_out(host_batch.size(), -1.0f);
    float* device_batch_in = nullptr;
    float* device_batch_out = nullptr;
    hipMalloc(reinterpret_cast<void**>(&device_batch_in),
              host_batch.size() * sizeof(float));
    hipMalloc(reinterpret_cast<void**>(&device_batch_out),
              host_batch.size() * sizeof(float));
    hipMemcpy(device_batch_in, host_batch.data(),
              host_batch.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(device_batch_out, host_batch_out.data(),
              host_batch.size() * sizeof(float), hipMemcpyHostToDevice);

    const dim3 grid(static_cast<unsigned>(tiles_m * tiles_n), 2);
    hipLaunchKernelGGL(
        roundtrip_3d_kernel, grid, dim3(64), 0, 0,
        Tensor<const float, 3, MemorySpace::HBM>{
            device_batch_in, batch_shape,
            HbmLayout<3>::contiguous(batch_shape)},
        Tensor<float, 3, MemorySpace::HBM>{
            device_batch_out, batch_shape,
            HbmLayout<3>::contiguous(batch_shape)},
        tiles_m, tiles_n);
    hipDeviceSynchronize();
    hipMemcpy(host_batch_out.data(), device_batch_out,
              host_batch_out.size() * sizeof(float), hipMemcpyDeviceToHost);
    CHECK(host_batch_out == host_batch);

    hipFree(device_src);
    hipFree(device_dst);
    hipFree(device_batch_in);
    hipFree(device_batch_out);
}

}  // namespace

int main() {
    std::printf("tile views\n");
    test_tile_counts();
    test_make_tile_grid();
    test_make_tile_at();
    test_copy_layouts();
    test_load_tile();
    test_rank3();
    test_misaligned_fallback();
    run_device_checks();
    if (failures != 0) {
        std::fprintf(stderr, "%d tile check(s) failed\n", failures);
        return 1;
    }
    std::printf("tile checks passed\n");
    return 0;
}
