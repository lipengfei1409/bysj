#pragma once

#include "ops/tensor.h"
#include "ops/copy.h"
#include "ops/dtype.h"
#include "ops/dumma.h"
#include "ops/op.h"
#include "dmk/state.h"
#include "ops/tile.h"

#include <type_traits>

namespace ops::gemm {

using tensor::MemorySpace;
using tensor::StridedLayout;
using tensor::Tensor;

// DUMMA exposes a single output tile shape, m16n16, and picks the K of that
// tile from the input type and the device (DTK 26.04 DUMMA 使用手册, 表 3.1):
//
//   float           m16n16k4    gfx926, gfx928, gfx936, gfx938
//   float           m16n16k8    gfx928, gfx936, gfx938
//   __half          m16n16k16   gfx928, gfx936, gfx938
//   __hip_bfloat16  m16n16k16   gfx928, gfx936, gfx938
//
// K is therefore a template parameter: a bigger K only means fewer k-steps,
// and every supported dtype flows through the same wave, staging and epilogue.
// Instantiating an unsupported (T, K) pair fails the static_assert below (the
// DTK header would reject the fragment type as well); the test runner only
// enables the pairs its --offload-arch supports.
inline constexpr size_t kTileM = 16;
inline constexpr size_t kTileN = 16;
inline constexpr size_t kWaveSize = 64;
inline constexpr size_t kStageCount = 3;

template <typename T, int K>
inline constexpr bool mma_supported =
    (std::is_same<T, float>::value && (K == 4 || K == 8)) ||
    ((std::is_same<T, __half>::value ||
      std::is_same<T, __hip_bfloat16>::value) &&
     K == 16);

// C[M,N] = A[M,K] * B[K,N]. A and B carry the input dtype, C accumulates in
// float (the accumulator type of every supported DUMMA shape).
template <typename T>
struct GemmParamsT {
    Tensor<const T, 2, MemorySpace::HBM> a;
    Tensor<const T, 2, MemorySpace::HBM> b;
    Tensor<float, 2, MemorySpace::HBM> c;

    DMK_HOST_DEVICE bool valid() const {
        const size_t m = c.shape.dims[0];
        const size_t n = c.shape.dims[1];
        const size_t k = a.shape.dims[1];
        return a.shape.dims[0] == m && b.shape.dims[0] == k &&
               b.shape.dims[1] == n &&
               (m == 0 || k == 0 || a.data != nullptr) &&
               (k == 0 || n == 0 || b.data != nullptr) &&
               (m == 0 || n == 0 || c.data != nullptr);
    }
};

// Exactly one wave64 owns this LDS region. Each array starts on a 16-byte
// boundary; A and B are row-major in LDS even when the HBM views are strided.
// The staging depth is kStageCount K-tiles.
template <typename T, int K>
struct alignas(16) GemmSharedT {
    T a_tile[kStageCount][kTileM * K];
    T b_tile[kStageCount][K * kTileN];
    float output_tile[kTileM * kTileN];
};

template <typename T, int K>
class GemmWaveT final {
public:
    static_assert(mma_supported<T, K>, "unsupported DUMMA (dtype, K) pair");

    static constexpr size_t tile_m = kTileM;
    static constexpr size_t tile_n = kTileN;
    static constexpr int tile_k = K;
    static constexpr unsigned wave_size = static_cast<unsigned>(kWaveSize);
    static constexpr size_t stage_count = kStageCount;

    // 16 B staging chunks: four floats or eight halves/bf16 values.
    static constexpr size_t kVector = 16 / sizeof(T);
    static constexpr size_t kAVectorsPerRow =
        static_cast<size_t>(tile_k) / kVector;
    static constexpr size_t kBVectorsPerRow =
        static_cast<size_t>(tile_n) / kVector;
    static constexpr size_t kCVectorsPerRow = static_cast<size_t>(tile_n) / 4;

    static_assert(static_cast<size_t>(tile_k) % kVector == 0,
                  "the K tile must be a whole number of 16 B chunks");
    static_assert(static_cast<size_t>(tile_n) % kVector == 0,
                  "the N tile must be a whole number of 16 B chunks");

    using AFragment = ::du::dumma::DUFragment<::du::dumma::matrix_a,
                                               tile_m, tile_n, tile_k, T,
                                               ::du::dumma::row_major>;
    using BFragment = ::du::dumma::DUFragment<::du::dumma::matrix_b,
                                               tile_m, tile_n, tile_k, T,
                                               ::du::dumma::row_major>;
    using Accumulator = ::du::dumma::DUFragment<::du::dumma::accumulator,
                                                 tile_m, tile_n, tile_k, float>;

    __device__ GemmWaveT(const GemmParamsT<T>& params,
                         GemmSharedT<T, K>& shared)
        : params_(params), shared_(shared) {}
    __device__ ~GemmWaveT() = default;

    __device__ void shard() {
        row_start_ = static_cast<size_t>(blockIdx.y) * tile_m;
        col_start_ = static_cast<size_t>(blockIdx.x) * tile_n;
    }

    __device__ bool has_output() const {
        return row_start_ < params_.c.shape.dims[0] &&
               col_start_ < params_.c.shape.dims[1];
    }

    // All lanes call this once before the first K tile.
    __device__ void prepare() {
        current_ = &accum_a_;
        next_ = &accum_b_;
        dumma::fill_fragment(*current_, 0.0f);
        active_a_ = &a_fragment_a_;
        active_b_ = &b_fragment_a_;
        spare_a_ = &a_fragment_b_;
        spare_b_ = &b_fragment_b_;
        fragments_ready_ = false;
    }

    __device__ void set_k_tile(size_t k_start) { k_start_ = k_start; }
    __device__ size_t batch_tiles() const { return batch_count_; }

    // Last logical GEMM tile in row-major output-tile order. This does not
    // imply that other blocks have finished storing their output.
    __device__ bool is_last() const {
        return row_start_ < params_.c.shape.dims[0] &&
               col_start_ < params_.c.shape.dims[1] &&
               k_start_ < params_.a.shape.dims[1] &&
               params_.c.shape.dims[0] - row_start_ <= tile_m &&
               params_.c.shape.dims[1] - col_start_ <= tile_n &&
               params_.a.shape.dims[1] - k_start_ <= tile_k;
    }

    // Stage up to three consecutive K tiles. Adjacent lanes visit adjacent HBM
    // addresses along the smallest-stride dimension, including transposed views.
    __device__ void loader() {
        batch_start_ = k_start_;
        const size_t k = params_.a.shape.dims[1];
        assert(batch_start_ < k);
        const size_t remaining = k - batch_start_;
        const size_t tiles_left =
            remaining / tile_k + (remaining % tile_k != 0);
        batch_count_ = tiles_left < stage_count ? tiles_left : stage_count;
        batch_consumed_ = 0;
        fragments_ready_ = false;
        const size_t lane = threadIdx.x;

        // A 16xK tile: four floats or eight halves per 16 B chunk. The fast
        // path needs contiguous columns, 16 B aligned rows and a tile that is
        // fully in bounds; a partial tile falls back to the masked scalar loop.
        const bool a_vectors =
            params_.a.layout.strides[1] == 1 &&
            params_.a.layout.strides[0] % kVector == 0 &&
            copy::aligned<float4>(params_.a.data);
        const bool b_vectors =
            params_.b.layout.strides[1] == 1 &&
            params_.b.layout.strides[0] % kVector == 0 &&
            copy::aligned<float4>(params_.b.data);

        for (size_t stage = 0; stage < batch_count_; ++stage) {
            const size_t k_start = batch_start_ + stage * tile_k;
            const size_t a_cols_left =
                k_start < k ? k - k_start : 0;
            const size_t b_rows_left =
                k_start < params_.b.shape.dims[0]
                    ? params_.b.shape.dims[0] - k_start
                    : 0;
            const bool a_full =
                row_start_ + tile_m <= params_.a.shape.dims[0] &&
                a_cols_left >= static_cast<size_t>(tile_k);
            const bool b_full =
                col_start_ + tile_n <= params_.b.shape.dims[1] &&
                b_rows_left >= static_cast<size_t>(tile_k);

            if (a_vectors && a_full) {
                const T* source =
                    params_.a.data +
                    row_start_ * params_.a.layout.strides[0] + k_start;
                const size_t stride = params_.a.layout.strides[0];
                const size_t total = tile_m * kAVectorsPerRow;
                for (size_t vector = lane; vector < total;
                     vector += wave_size) {
                    const size_t row = vector / kAVectorsPerRow;
                    const size_t col = (vector % kAVectorsPerRow) * kVector;
                    copy::store_vec<float4>(
                        shared_.a_tile[stage] + row * tile_k + col,
                        copy::load_vec<float4>(source + row * stride + col));
                }
            } else {
                for (size_t index = lane; index < tile_m * tile_k;
                     index += wave_size) {
                    size_t row, col;
                    tile_indices(index, tile_m, tile_k, params_.a.layout, row,
                                 col);
                    const size_t a_row = row_start_ + row;
                    const size_t a_col = k_start + col;
                    shared_.a_tile[stage][row * tile_k + col] =
                        a_row < params_.a.shape.dims[0] && a_col < k
                            ? params_.a.data[
                                  a_row * params_.a.layout.strides[0] +
                                  a_col * params_.a.layout.strides[1]]
                            : dtype::from_float<T>(0.0f);
                }
            }

            if (b_vectors && b_full) {
                const T* source =
                    params_.b.data +
                    k_start * params_.b.layout.strides[0] + col_start_;
                const size_t stride = params_.b.layout.strides[0];
                const size_t total = tile_k * kBVectorsPerRow;
                for (size_t vector = lane; vector < total;
                     vector += wave_size) {
                    const size_t row = vector / kBVectorsPerRow;
                    const size_t col = (vector % kBVectorsPerRow) * kVector;
                    copy::store_vec<float4>(
                        shared_.b_tile[stage] + row * tile_n + col,
                        copy::load_vec<float4>(source + row * stride + col));
                }
            } else {
                for (size_t index = lane; index < tile_k * tile_n;
                     index += wave_size) {
                    size_t row, col;
                    tile_indices(index, tile_k, tile_n, params_.b.layout, row,
                                 col);
                    const size_t b_row = k_start + row;
                    const size_t b_col = col_start_ + col;
                    shared_.b_tile[stage][row * tile_n + col] =
                        b_row < params_.b.shape.dims[0] &&
                        b_col < params_.b.shape.dims[1]
                            ? params_.b.data[
                                  b_row * params_.b.layout.strides[0] +
                                  b_col * params_.b.layout.strides[1]]
                            : dtype::from_float<T>(0.0f);
                }
            }
        }
        __syncthreads();
    }

    // Each call performs one MMA. The first call primes the register fragment;
    // each subsequent call uses the fragment loaded by the preceding call.
    __device__ void consumer() {
        assert(batch_consumed_ < batch_count_);
        const size_t stage = batch_consumed_;
        if (!fragments_ready_) {
            dumma::ld_matrix(*active_a_, shared_.a_tile[stage],
                             static_cast<unsigned>(tile_k));
            dumma::ld_matrix(*active_b_, shared_.b_tile[stage],
                             static_cast<unsigned>(tile_n));
            fragments_ready_ = true;
        }
        if (stage + 1 < batch_count_) {
            dumma::ld_matrix(*spare_a_, shared_.a_tile[stage + 1],
                             static_cast<unsigned>(tile_k));
            dumma::ld_matrix(*spare_b_, shared_.b_tile[stage + 1],
                             static_cast<unsigned>(tile_n));
        }
        k_start_ = batch_start_ + stage * tile_k;
        dumma::mma(*next_, *active_a_, *active_b_, *current_);
        Accumulator* previous = current_;
        current_ = next_;
        next_ = previous;
        ++batch_consumed_;
        if (batch_consumed_ < batch_count_) {
            AFragment* previous_a = active_a_;
            active_a_ = spare_a_;
            spare_a_ = previous_a;
            BFragment* previous_b = active_b_;
            active_b_ = spare_b_;
            spare_b_ = previous_b;
        } else {
            fragments_ready_ = false;
        }
    }

    __device__ void storer() {
        dumma::store_matrix(shared_.output_tile, *current_, tile_n,
                            dumma::native::mem_row_major);
        __syncthreads();
        const size_t lane = threadIdx.x;

        // The 256-element epilogue tile is exactly 64 float4 chunks, one per
        // lane, when C is contiguous and the tile is in bounds.
        const bool c_full =
            row_start_ + tile_m <= params_.c.shape.dims[0] &&
            col_start_ + tile_n <= params_.c.shape.dims[1];
        const bool c_vectors =
            params_.c.layout.strides[1] == 1 &&
            params_.c.layout.strides[0] % 4 == 0 &&
            copy::aligned<float4>(params_.c.data);
        if (c_vectors && c_full) {
            float* destination =
                params_.c.data +
                row_start_ * params_.c.layout.strides[0] + col_start_;
            const size_t stride = params_.c.layout.strides[0];
            const size_t total = tile_m * kCVectorsPerRow;
            for (size_t vector = lane; vector < total; vector += wave_size) {
                const size_t row = vector / kCVectorsPerRow;
                const size_t col = (vector % kCVectorsPerRow) * 4;
                copy::store_vec<float4>(
                    destination + row * stride + col,
                    copy::load_vec<float4>(shared_.output_tile +
                                           row * tile_n + col));
            }
            return;
        }

        for (size_t index = lane; index < tile_m * tile_n; index += wave_size) {
            size_t row, col;
            tile_indices(index, tile_m, tile_n, params_.c.layout, row, col);
            const size_t c_row = row_start_ + row;
            const size_t c_col = col_start_ + col;
            if (c_row < params_.c.shape.dims[0] &&
                c_col < params_.c.shape.dims[1]) {
                params_.c.data[c_row * params_.c.layout.strides[0] +
                               c_col * params_.c.layout.strides[1]] =
                    shared_.output_tile[row * tile_n + col];
            }
        }
    }

private:
    // Give neighboring lanes neighboring HBM addresses for row- or
    // column-major views; LDS always retains the DUMMA row-major order.
    __device__ static void tile_indices(size_t index, size_t rows, size_t cols,
                                        const StridedLayout<2>& layout,
                                        size_t& row, size_t& col) {
        if (layout.strides[1] <= layout.strides[0]) {
            row = index / cols;
            col = index % cols;
        } else {
            row = index % rows;
            col = index / rows;
        }
    }

    GemmParamsT<T> params_;
    GemmSharedT<T, K>& shared_;
    size_t row_start_ = 0;
    size_t col_start_ = 0;
    size_t k_start_ = 0;
    size_t batch_start_ = 0;
    size_t batch_count_ = 0;
    size_t batch_consumed_ = 0;
    bool fragments_ready_ = false;
    AFragment a_fragment_a_;
    AFragment a_fragment_b_;
    BFragment b_fragment_a_;
    BFragment b_fragment_b_;
    AFragment* active_a_ = nullptr;
    AFragment* spare_a_ = nullptr;
    BFragment* active_b_ = nullptr;
    BFragment* spare_b_ = nullptr;
    Accumulator accum_a_;
    Accumulator accum_b_;
    Accumulator* current_ = nullptr;
    Accumulator* next_ = nullptr;
};

// Short names for the float m16n16k4 configuration, which the megakernel GEMM
// op and the existing perf test use.
using GemmParams = GemmParamsT<float>;
using GemmShared = GemmSharedT<float, 4>;
using GemmWave = GemmWaveT<float, 4>;

// Launch with one wave64 per block and one distinct __shared__ GemmSharedT per
// block; grid=(ceil(N/16), ceil(M/16)). Every lane follows the same K loop.
template <typename T, int K>
__device__ inline void run_gemm_wave(const GemmParamsT<T>& params,
                                     GemmSharedT<T, K>& shared) {
    static_assert(mma_supported<T, K>, "unsupported DUMMA (dtype, K) pair");
    assert(blockDim.x == kWaveSize && blockDim.y == 1 && blockDim.z == 1);
    if (!params.valid()) return;
    GemmWaveT<T, K> op(params, shared);
    op.shard();
    if (!op.has_output()) return;
    op.prepare();
    for (size_t k_start = 0; k_start < params.a.shape.dims[1];
         k_start += static_cast<size_t>(K) * kStageCount) {
        op.set_k_tile(k_start);
        op.loader();
        for (size_t tile = 0; tile < op.batch_tiles(); ++tile) op.consumer();
    }
    op.storer();
}

}  // namespace ops::gemm

// Page-based GEMM operator. One instruction is one whole output tile over the
// full K of the matrix: load() stages the first A/B k-tile and clears C, run()
// walks every k-tile and accumulates into the C page, store() writes C.
// Pages: lid 0 = A k-tile, lid 1 = B k-tile, lid 2 = C tile; the pages the
// operator does not own are released by load().
namespace ops {

template <typename Globals>
class GemmOp final : public BaseOp<Globals> {
public:
    static constexpr int opcode = 2;
    static constexpr int kTile = 16;
    static constexpr int kTileElements = kTile * kTile;
    static constexpr tile::TileShape<2> kTileShape{{kTile, kTile}};

    struct Instruction {
        int m0;
        int n0;
        int k_tiles;

        __device__ explicit Instruction(const int* words)
            : m0(words[1]), n0(words[2]), k_tiles(words[3]) {}
    };

    __device__ explicit GemmOp(const int* instruction)
        : BaseOp<Globals>(instruction) {}

    __device__ int release_lid(int query) const override { return query; }

    __device__ int init_semaphores(dmk::mk::State& s,
                                   const Globals&) const override {
        s.semaphores()[0].init(1);
        s.semaphores()[1].init(config::kConsumerWaves);
        return 2;
    }

    // Stage the A/B k-tile at (k * 16) into the two operand pages. The runtime
    // calls it once (k = 0), run() calls it for the remaining k.
    __device__ static void stage_operands(dmk::mk::State& s, const Globals& g,
                                          const Instruction& inst, int k,
                                          int lane) {
        const auto shape = tensor::Shape<2>{{kTile, kTile}};
        const auto layout = tensor::StridedLayout<2>{{kTile, 1}};
        const size_t a_origin[2] = {static_cast<size_t>(inst.m0),
                                    static_cast<size_t>(k * kTile)};
        const size_t b_origin[2] = {static_cast<size_t>(k * kTile),
                                    static_cast<size_t>(inst.n0)};
        const auto a_src = tile::make_tile_at(g.matrix_a, kTileShape, a_origin);
        const auto b_src = tile::make_tile_at(g.matrix_b, kTileShape, b_origin);
        const int pid_a = s.pid(0);
        const int pid_b = s.pid(1);
        tile::load_tile(
            tile::make_lds_view(static_cast<float*>(s.pages[pid_a].ptr()), shape,
                                layout),
            a_src, lane, config::kWaveSize);
        tile::load_tile(
            tile::make_lds_view(static_cast<float*>(s.pages[pid_b].ptr()), shape,
                                layout),
            b_src, lane, config::kWaveSize);
    }

    __device__ void load(dmk::mk::State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int pid_a = s.pid(0);
        const int pid_b = s.pid(1);
        const int pid_c = s.pid(2);
        s.wait_page_ready(pid_a);
        s.wait_page_ready(pid_b);
        s.wait_page_ready(pid_c);

        stage_operands(s, g, inst, 0, lane);
        tile::fill_tile(
            tile::make_lds_view(static_cast<float*>(s.pages[pid_c].ptr()),
                                tensor::Shape<2>{{kTile, kTile}}),
            0.0f, lane, config::kWaveSize);

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[0].arrive();

        // Pages the operator does not own.
        if (lane >= 3 && lane < config::kNumPages) {
            const int unused = s.pid(lane);
            s.wait_page_ready(unused);
            s.finish_page(unused, config::kConsumerWaves);
        }
    }

    // The whole GEMM for this output tile: every k-tile, accumulating into C.
    __device__ void run(dmk::mk::State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);

        s.semaphores()[0].wait(0);
        if (wave == 0) {
            const int pid_a = s.pid(0);
            const int pid_b = s.pid(1);
            const int pid_c = s.pid(2);
            const float* a_page =
                static_cast<const float*>(s.pages[pid_a].ptr());
            const float* b_page =
                static_cast<const float*>(s.pages[pid_b].ptr());
            float* c_page = static_cast<float*>(s.pages[pid_c].ptr());

            for (int k = 0; k < inst.k_tiles; ++k) {
                if (k > 0) {
                    stage_operands(s, g, inst, k, lane);
                    lds::wave_barrier();
                    __threadfence_block();
                }
                for (int i = lane; i < kTileElements; i += config::kWaveSize) {
                    const int row = i / kTile;
                    const int col = i % kTile;
                    float sum = 0.0f;
#pragma unroll
                    for (int kk = 0; kk < kTile; ++kk)
                        sum += a_page[row * kTile + kk] *
                               b_page[kk * kTile + col];
                    c_page[i] += sum;
                }
                lds::wave_barrier();
                __threadfence_block();
            }
        }
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[1].arrive();
    }

    __device__ void store(dmk::mk::State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int pid_a = s.pid(0);
        const int pid_b = s.pid(1);
        const int pid_c = s.pid(2);

        s.semaphores()[1].wait(0);

        const size_t c_origin[2] = {static_cast<size_t>(inst.m0),
                                    static_cast<size_t>(inst.n0)};
        const auto c_dst = tile::make_tile_at(g.matrix_c, kTileShape, c_origin);
        const auto c_src = tile::make_lds_view(
            static_cast<const float*>(s.pages[pid_c].ptr()), c_dst.shape,
            tensor::StridedLayout<2>{{kTile, 1}});
        tile::copy_tile(c_dst, c_src, lane, config::kWaveSize);

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) {
            s.finish_page(pid_a, config::kConsumerWaves);
            s.finish_page(pid_b, config::kConsumerWaves);
            s.finish_page(pid_c, config::kConsumerWaves);
        }
    }
};

}  // namespace ops
