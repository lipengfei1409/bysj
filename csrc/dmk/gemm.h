#pragma once

#include "tensor.h"
#include "dumma.h"
#include "op.h"

namespace dmk::gemm {

using tensor::MemorySpace;
using tensor::StridedLayout;
using tensor::Tensor;

// C[M,N] = A[M,K] * B[K,N]. The three HBM views may be strided.
struct GemmParams {
    Tensor<const float, 2, MemorySpace::HBM> a;
    Tensor<const float, 2, MemorySpace::HBM> b;
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
// boundary; A and B are row-major even when their HBM views are strided.
struct alignas(16) GemmShared {
    float a_tile[3][16 * 4];
    float b_tile[3][4 * 16];
    float output_tile[16 * 16];
};

class GemmOp final : public BaseOp {
public:
    static constexpr size_t tile_m = 16;
    static constexpr size_t tile_n = 16;
    static constexpr size_t tile_k = 4;
    static constexpr unsigned wave_size = 64;
    static constexpr size_t stage_count = 3;

    using AFragment = ::du::dumma::DUFragment<::du::dumma::matrix_a,
                                               tile_m, tile_n, tile_k,
                                               float, ::du::dumma::row_major>;
    using BFragment = ::du::dumma::DUFragment<::du::dumma::matrix_b,
                                               tile_m, tile_n, tile_k,
                                               float, ::du::dumma::row_major>;
    using Accumulator = ::du::dumma::DUFragment<::du::dumma::accumulator,
                                                 tile_m, tile_n, tile_k, float>;

    __device__ GemmOp(const GemmParams& params, GemmShared& shared)
        : params_(params), shared_(shared) {}
    __device__ ~GemmOp() override = default;

    __device__ void shard() override {
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
    __device__ bool is_last() const override {
        return row_start_ < params_.c.shape.dims[0] &&
               col_start_ < params_.c.shape.dims[1] &&
               k_start_ < params_.a.shape.dims[1] &&
               params_.c.shape.dims[0] - row_start_ <= tile_m &&
               params_.c.shape.dims[1] - col_start_ <= tile_n &&
               params_.a.shape.dims[1] - k_start_ <= tile_k;
    }

    // Stage up to three consecutive K tiles. Adjacent lanes visit adjacent HBM
    // addresses along the smallest-stride dimension, including transposed views.
    __device__ void loader() override {
        batch_start_ = k_start_;
        const size_t k = params_.a.shape.dims[1];
        assert(batch_start_ < k);
        const size_t remaining = k - batch_start_;
        const size_t tiles_left = remaining / tile_k + (remaining % tile_k != 0);
        batch_count_ = tiles_left < stage_count ? tiles_left : stage_count;
        batch_consumed_ = 0;
        fragments_ready_ = false;
        const size_t lane = threadIdx.x;
        for (size_t stage = 0; stage < batch_count_; ++stage) {
            const size_t k_start = batch_start_ + stage * tile_k;
            size_t row, col;
            tile_indices(lane, tile_m, tile_k, params_.a.layout, row, col);
            const size_t a_row = row_start_ + row;
            const size_t a_col = k_start + col;
            shared_.a_tile[stage][row * tile_k + col] =
                a_row < params_.a.shape.dims[0] && a_col < k
                    ? params_.a.data[a_row * params_.a.layout.strides[0] +
                                     a_col * params_.a.layout.strides[1]]
                    : 0.0f;

            tile_indices(lane, tile_k, tile_n, params_.b.layout, row, col);
            const size_t b_row = k_start + row;
            const size_t b_col = col_start_ + col;
            shared_.b_tile[stage][row * tile_n + col] =
                b_row < params_.b.shape.dims[0] &&
                b_col < params_.b.shape.dims[1]
                    ? params_.b.data[b_row * params_.b.layout.strides[0] +
                                     b_col * params_.b.layout.strides[1]]
                    : 0.0f;
        }
        __syncwarp();
    }

    // Each call performs one MMA. The first call primes the register fragment;
    // each subsequent call uses the fragment loaded by the preceding call.
    __device__ void consumer() override {
        assert(batch_consumed_ < batch_count_);
        const size_t stage = batch_consumed_;
        if (!fragments_ready_) {
            dumma::ld_matrix(*active_a_, shared_.a_tile[stage], tile_k);
            dumma::ld_matrix(*active_b_, shared_.b_tile[stage], tile_n);
            fragments_ready_ = true;
        }
        if (stage + 1 < batch_count_) {
            dumma::ld_matrix(*spare_a_, shared_.a_tile[stage + 1], tile_k);
            dumma::ld_matrix(*spare_b_, shared_.b_tile[stage + 1], tile_n);
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

    __device__ void storer() override {
        dumma::store_matrix(shared_.output_tile, *current_, tile_n,
                            dumma::native::mem_row_major);
        __syncwarp();
        const size_t lane = threadIdx.x;
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

    GemmParams params_;
    GemmShared& shared_;
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

// Launch with one wave64 per block and one distinct __shared__ GemmShared per
// block; grid=(ceil(N/16), ceil(M/16)). Every lane follows the same K loop.
__device__ inline void run_gemm_wave(const GemmParams& params,
                                     GemmShared& shared) {
    assert(blockDim.x == GemmOp::wave_size &&
           blockDim.y == 1 && blockDim.z == 1);
    if (!params.valid()) return;
    GemmOp op(params, shared);
    op.shard();
    if (!op.has_output()) return;
    op.prepare();
    for (size_t k_start = 0; k_start < params.a.shape.dims[1];
         k_start += GemmOp::tile_k * GemmOp::stage_count) {
        op.set_k_tile(k_start);
        op.loader();
        for (size_t tile = 0; tile < op.batch_tiles(); ++tile)
            op.consumer();
    }
    op.storer();
}

}  // namespace dmk::gemm
