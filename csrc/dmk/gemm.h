#pragma once

#include "tensor.h"
#include "copy.h"
#include "dumma.h"
#include "op.h"
#include "state.h"

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

class GemmWave final {
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

    __device__ GemmWave(const GemmParams& params, GemmShared& shared)
        : params_(params), shared_(shared) {}
    __device__ ~GemmWave() = default;

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
        __syncthreads();
    }

    // Each call performs one MMA. The first call primes the register fragment;
    // each subsequent call uses the fragment loaded by the preceding call.
    __device__ void consumer() {
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

    __device__ void storer() {
        dumma::store_matrix(shared_.output_tile, *current_, tile_n,
                            dumma::native::mem_row_major);
        __syncthreads();
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
    assert(blockDim.x == GemmWave::wave_size &&
           blockDim.y == 1 && blockDim.z == 1);
    if (!params.valid()) return;
    GemmWave op(params, shared);
    op.shard();
    if (!op.has_output()) return;
    op.prepare();
    for (size_t k_start = 0; k_start < params.a.shape.dims[1];
         k_start += GemmWave::tile_k * GemmWave::stage_count) {
        op.set_k_tile(k_start);
        op.loader();
        for (size_t tile = 0; tile < op.batch_tiles(); ++tile)
            op.consumer();
    }
    op.storer();
}

}  // namespace dmk::gemm

// Page-based concrete GEMM operation for the megakernel VM. The low-level
// GemmWave/run_gemm_wave above remain the standalone DUMMA microbenchmark;
// this class owns the page/semaphore lifecycle and is what the controller sees.
namespace dmk::mk {

template <typename Config, typename Globals>
class GemmOp final : public BaseOp<Config, Globals> {
public:
    static constexpr int opcode = 2;
    static constexpr int kTile = 16;
    static constexpr int kTileElements = kTile * kTile;

    struct Instruction {
        int m0;
        int n0;
        int k0;

        __device__ explicit Instruction(const int* words)
            : m0(words[1]), n0(words[2]), k0(words[3]) {}
    };

    __device__ explicit GemmOp(const int* instruction)
        : BaseOp<Config, Globals>(instruction) {}

    __device__ int release_lid(int query) const override { return query; }

    __device__ int init_semaphores(State<Config>& s,
                                   const Globals&) const override {
        s.semaphores()[0].init(1);
        s.semaphores()[1].init(1);
        return 2;
    }

    __device__ void loader(State<Config>& s,
                           const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);

        const int pid_a = s.pid(0);
        const int pid_b = s.pid(1);
        const int pid_c = s.pid(2);
        s.wait_page_ready(pid_a);
        s.wait_page_ready(pid_b);
        s.wait_page_ready(pid_c);

        float* a_page = static_cast<float*>(s.pages[pid_a].ptr());
        float* b_page = static_cast<float*>(s.pages[pid_b].ptr());
        const auto a = g.matrix_a;
        const auto b = g.matrix_b;

        const int a_rows = static_cast<int>(a.shape.dims[0]);
        const int a_cols = static_cast<int>(a.shape.dims[1]);
        const int b_rows = static_cast<int>(b.shape.dims[0]);
        const int b_cols = static_cast<int>(b.shape.dims[1]);
        const bool a_full = inst.m0 >= 0 && inst.k0 >= 0 &&
                            inst.m0 + kTile <= a_rows &&
                            inst.k0 + kTile <= a_cols &&
                            a.layout.strides[1] == 1;
        const bool b_full = inst.k0 >= 0 && inst.n0 >= 0 &&
                            inst.k0 + kTile <= b_rows &&
                            inst.n0 + kTile <= b_cols &&
                            b.layout.strides[1] == 1;

        if (a_full) {
            copy::global_rows_to_lds_vec<float4>(
                a_page, kTile,
                a.data + static_cast<size_t>(inst.m0) * a.layout.strides[0] +
                    static_cast<size_t>(inst.k0) * a.layout.strides[1],
                a.layout.strides[0], kTile, kTile,
                static_cast<size_t>(lane), Config::kWaveSize);
        } else {
            for (int i = lane; i < kTileElements; i += Config::kWaveSize) {
                const int row = i / kTile;
                const int col = i % kTile;
                const int ar = inst.m0 + row;
                const int ak = inst.k0 + col;
                a_page[i] =
                    ar >= 0 && ar < a_rows && ak >= 0 && ak < a_cols
                        ? a.data[static_cast<size_t>(ar) *
                                     a.layout.strides[0] +
                                 static_cast<size_t>(ak) *
                                     a.layout.strides[1]]
                        : 0.0f;
            }
        }

        if (b_full) {
            copy::global_rows_to_lds_vec<float4>(
                b_page, kTile,
                b.data + static_cast<size_t>(inst.k0) * b.layout.strides[0] +
                    static_cast<size_t>(inst.n0) * b.layout.strides[1],
                b.layout.strides[0], kTile, kTile,
                static_cast<size_t>(lane), Config::kWaveSize);
        } else {
            for (int i = lane; i < kTileElements; i += Config::kWaveSize) {
                const int row = i / kTile;
                const int col = i % kTile;
                const int bk = inst.k0 + row;
                const int bn = inst.n0 + col;
                b_page[i] =
                    bk >= 0 && bk < b_rows && bn >= 0 && bn < b_cols
                        ? b.data[static_cast<size_t>(bk) *
                                     b.layout.strides[0] +
                                 static_cast<size_t>(bn) *
                                     b.layout.strides[1]]
                        : 0.0f;
            }
        }
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[0].arrive();

        // Pages this op does not use.
        if (lane >= 3 && lane < Config::kNumPages) {
            const int unused = s.pid(lane);
            s.wait_page_ready(unused);
            s.finish_page(unused, Config::kConsumerWaves);
        }
    }

    __device__ void consumer(State<Config>& s,
                             const Globals&) const override {
        const int wave = static_cast<int>(threadIdx.x / Config::kWaveSize);
        if (wave != 0) return;  // only one consumer wave runs this tile

        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        const int pid_a = s.pid(0);
        const int pid_b = s.pid(1);
        const int pid_c = s.pid(2);

        s.semaphores()[0].wait(0);

        const float* a_page = static_cast<const float*>(s.pages[pid_a].ptr());
        const float* b_page = static_cast<const float*>(s.pages[pid_b].ptr());
        float* c_page = static_cast<float*>(s.pages[pid_c].ptr());

        for (int i = lane; i < kTileElements; i += Config::kWaveSize) {
            const int row = i / kTile;
            const int col = i % kTile;
            float sum = 0.0f;
#pragma unroll
            for (int kk = 0; kk < kTile; ++kk) {
                sum += a_page[row * kTile + kk] *
                       b_page[kk * kTile + col];
            }
            c_page[i] = sum;
        }
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[1].arrive();
    }

    __device__ void storer(State<Config>& s,
                           const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        const int pid_a = s.pid(0);
        const int pid_b = s.pid(1);
        const int pid_c = s.pid(2);

        s.semaphores()[1].wait(0);

        const float* c_page =
            static_cast<const float*>(s.pages[pid_c].ptr());
        auto c = g.matrix_c;
        const int c_rows = static_cast<int>(c.shape.dims[0]);
        const int c_cols = static_cast<int>(c.shape.dims[1]);
        const bool c_full = inst.m0 >= 0 && inst.n0 >= 0 &&
                            inst.m0 + kTile <= c_rows &&
                            inst.n0 + kTile <= c_cols &&
                            c.layout.strides[1] == 1;
        if (c_full) {
            copy::lds_rows_to_global_vec<float4>(
                c.data + static_cast<size_t>(inst.m0) * c.layout.strides[0] +
                    static_cast<size_t>(inst.n0) * c.layout.strides[1],
                c.layout.strides[0], c_page, kTile, kTile, kTile,
                static_cast<size_t>(lane), Config::kWaveSize);
        } else {
            for (int i = lane; i < kTileElements; i += Config::kWaveSize) {
                const int row = i / kTile;
                const int col = i % kTile;
                const int cr = inst.m0 + row;
                const int cc = inst.n0 + col;
                if (cr >= 0 && cr < c_rows && cc >= 0 && cc < c_cols) {
                    c.data[static_cast<size_t>(cr) * c.layout.strides[0] +
                           static_cast<size_t>(cc) * c.layout.strides[1]] =
                        c_page[i];
                }
            }
        }
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) {
            s.finish_page(pid_a, Config::kConsumerWaves);
            s.finish_page(pid_b, Config::kConsumerWaves);
            s.finish_page(pid_c, Config::kConsumerWaves);
        }
    }
};

}  // namespace dmk::mk
