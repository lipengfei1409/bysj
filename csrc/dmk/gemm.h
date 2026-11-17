#pragma once

#include <hip/hip_runtime.h>
#include <du_mma.h>

#include "copy.h"
#include "op.h"

namespace dmk {

// FP32 GEMM: C[M,N] = A[M,K] * B[K,N]. Leading dimensions count elements.
// A/B/C layouts describe HBM only. Validate sizes, strides, buffers, and the
// DUMMA-capable gfx target before launching.
struct GemmParams {
    const float* a;
    const float* b;
    float* c;
    int m, n, k;
    int lda, ldb, ldc;
    HbmLayout layout_a = HbmLayout::row_major;
    HbmLayout layout_b = HbmLayout::row_major;
    HbmLayout layout_c = HbmLayout::row_major;
};

// One wave owns this LDS region. DUMMA reads row-major A/B and writes a
// row-major output tile; copy.h converts external HBM row/column layouts.
struct alignas(16) GemmShared {
    static constexpr ShmemLayout layout = ShmemLayout::row_major;
    float a_tile[16 * 4];
    float b_tile[4 * 16];
    float output_tile[16 * 16];
};

class GemmOp final : public BaseOp {
public:
    static constexpr int tile_m = 16;
    static constexpr int tile_n = 16;
    static constexpr int tile_k = 4;

    using AFragment = ::du::dumma::DUFragment<::du::dumma::matrix_a,
                                               tile_m, tile_n, tile_k,
                                               float, ::du::dumma::row_major>;
    using BFragment = ::du::dumma::DUFragment<::du::dumma::matrix_b,
                                               tile_m, tile_n, tile_k,
                                               float, ::du::dumma::row_major>;
    using Accumulator = ::du::dumma::DUFragment<::du::dumma::accumulator,
                                                 tile_m, tile_n, tile_k, float>;

    __device__ GemmOp(const GemmParams& params, GemmShared& shared,
                      int row_start, int col_start)
        : params_(params), shared_(shared), row_start_(row_start),
          col_start_(col_start), k_start_(0), current_(&accum_a_),
          next_(&accum_b_) {}

    __device__ ~GemmOp() override = default;

    // Called by all 64 lanes of the same wave before the first K tile.
    __device__ void prepare() {
        ::du::dumma::du_fill_fragment(*current_, 0.0f);
    }

    __device__ void set_k_tile(int k_start) { k_start_ = k_start; }

    __device__ void loader() override {
        copy::stage_to_shmem<tile_m, tile_k>(
            shared_.a_tile, tile_k, GemmShared::layout,
            params_.a, params_.m, params_.k, params_.lda,
            params_.layout_a, row_start_, k_start_);
        copy::stage_to_shmem<tile_k, tile_n>(
            shared_.b_tile, tile_n, GemmShared::layout,
            params_.b, params_.k, params_.n, params_.ldb,
            params_.layout_b, k_start_, col_start_);
        __syncwarp();
        copy::ld_matrix(a_fragment_, shared_.a_tile, tile_k);
        copy::ld_matrix(b_fragment_, shared_.b_tile, tile_n);
    }

    __device__ void consumer() override {
        ::du::dumma::du_mma_sync(*next_, a_fragment_, b_fragment_, *current_);
        auto* previous = current_;
        current_ = next_;
        next_ = previous;
    }

    __device__ void storer() override {
        copy::store_matrix(shared_.output_tile, *current_, tile_n,
                           ::du::dumma::mem_row_major);
        __syncwarp();
        copy::store_from_shmem<tile_m, tile_n>(
            params_.c, params_.m, params_.n, params_.ldc, params_.layout_c,
            shared_.output_tile, tile_n, GemmShared::layout,
            row_start_, col_start_);
    }

private:
    GemmParams params_;
    GemmShared& shared_;
    int row_start_;
    int col_start_;
    int k_start_;
    AFragment a_fragment_;
    BFragment b_fragment_;
    Accumulator accum_a_;
    Accumulator accum_b_;
    Accumulator* current_;
    Accumulator* next_;
};

// Invoke on one complete wave64. The same wave calls all three virtual stages,
// because DUMMA fragments live in its lane registers. The caller supplies a
// distinct __shared__ GemmShared per active wave; grid.x=ceil(N/16),
// grid.y=ceil(M/16). No other wave may reuse that LDS region concurrently.
__device__ inline void run_gemm_wave(const GemmParams& params,
                                     GemmShared& shared) {
    GemmOp op(params, shared, blockIdx.y * GemmOp::tile_m,
              blockIdx.x * GemmOp::tile_n);
    BaseOp* base = &op;
    op.prepare();
    for (int k_start = 0; k_start < params.k; k_start += GemmOp::tile_k) {
        op.set_k_tile(k_start);
        base->loader();
        base->consumer();
    }
    base->storer();
}

}  // namespace dmk
