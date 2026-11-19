#pragma once

// FlashAttention-style single-head attention operator.
//
//   out[m, :] = softmax_j( dot(q[m, :], k[j, :]) / sqrt(D) ) * v[j, :]
//
// The kernel is organized like FlashAttention-2 instead of materializing the
// score matrix:
//   - one block owns kQueryTile (16) query rows and keeps them in LDS for the
//     whole kernel,
//   - keys are visited in tiles of kKeyTile (16); each K/V tile is staged in
//     LDS once and reused by all 16 query rows,
//   - the tile scores are folded into a running max/sum (online softmax), the
//     output tile is rescaled once per key tile by exp(m_old - m_new), and
//     P * V is accumulated into the LDS output tile,
//   - only out[M, D] is written back; the M x N score matrix never exists.
//
// Both tiles are 16 wide, so each S tile is exactly one DUMMA m16n16 tile when
// the scalar QK^T loop is replaced by fragments later.
//
// The element type is a template parameter (float, __half, __hip_bfloat16 for
// Q/K/V and the output); all arithmetic is float, and the LDS tiles keep the
// input dtype to stay inside the 64 KiB budget.
//
// Limitations of this version:
//   - QK^T and P*V are still scalar MAC loops; wiring DUMMA in needs the
//     accumulator-fragment rescaling question answered on hardware,
//   - one wave (64 threads) per block: the D=128 float tiles already use
//     ~34 KiB of LDS, so a second wave would not fit,
//   - no causal mask and no multi-head/batch dimension yet.

#include "config.h"
#include "dtype.h"
#include "tensor.h"

#include <cmath>

namespace dmk::attention {

using tensor::MemorySpace;
using tensor::Tensor;

inline constexpr int kMaxHeadDim = 128;
inline constexpr int kWaveSize = config::kWaveSize;
inline constexpr int kQueryTile = 16;  // rows per block
inline constexpr int kKeyTile = 16;    // keys per tile
inline constexpr int kThreadsPerBlock = kWaveSize;
inline constexpr int kLanesPerRow = kWaveSize / kQueryTile;      // 4
inline constexpr int kColumnsPerLane = kKeyTile / kLanesPerRow;  // 4

static_assert(kWaveSize % kQueryTile == 0,
              "query tile must split the wave into whole lane groups");
static_assert(kKeyTile % kLanesPerRow == 0,
              "key tile must split evenly across a row's lane group");

// O[M, D] = softmax(Q[M, D] * K[N, D]^T / sqrt(D)) * V[N, D].
template <typename T>
struct AttentionParamsT {
    Tensor<const T, 2, MemorySpace::HBM> q;
    Tensor<const T, 2, MemorySpace::HBM> k;
    Tensor<const T, 2, MemorySpace::HBM> v;
    Tensor<T, 2, MemorySpace::HBM> out;

    DMK_HOST_DEVICE bool valid() const {
        const size_t rows = out.shape.dims[0];
        const size_t dim = out.shape.dims[1];
        const size_t keys = k.shape.dims[0];
        if (dim == 0 || dim > kMaxHeadDim) return false;
        if (q.shape.dims[0] != rows || q.shape.dims[1] != dim) return false;
        if (k.shape.dims[1] != dim || v.shape.dims[1] != dim) return false;
        if (v.shape.dims[0] != keys) return false;
        if (rows != 0 && (q.data == nullptr || out.data == nullptr)) return false;
        if (keys != 0 && (k.data == nullptr || v.data == nullptr)) return false;
        return true;
    }
};

using AttentionParams = AttentionParamsT<float>;

// LDS budget at D = kMaxHeadDim: ~34 KiB for float, ~22 KiB for half/bf16.
template <typename T>
struct alignas(16) AttentionSharedT {
    T q_tile[kQueryTile][kMaxHeadDim];
    T k_tile[kKeyTile][kMaxHeadDim];
    T v_tile[kKeyTile][kMaxHeadDim];
    float s_tile[kQueryTile][kKeyTile];
    T p_tile[kQueryTile][kKeyTile];
    float o_tile[kQueryTile][kMaxHeadDim];
    float m_row[kQueryTile];      // running max per query row
    float l_row[kQueryTile];      // running softmax denominator
    float alpha_row[kQueryTile];  // exp(m_old - m_new) of the current tile
};

// Grid size for a launch with kThreadsPerBlock threads.
inline unsigned attention_blocks(size_t rows) {
    return static_cast<unsigned>((rows + kQueryTile - 1) / kQueryTile);
}

// Converge the wave and publish its LDS writes.
__device__ inline void block_sync() {
    lds::wave_barrier();
    __threadfence_block();
}

// One query tile. Every lane of the block calls this together; `shared` is the
// block's __shared__ AttentionSharedT<T>.
template <typename T>
__device__ inline void run_attention(const AttentionParamsT<T>& params,
                                     AttentionSharedT<T>& shared) {
    if (!params.valid()) return;

    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    const int rows = static_cast<int>(params.out.shape.dims[0]);
    const int keys = static_cast<int>(params.k.shape.dims[0]);
    const int dim = static_cast<int>(params.out.shape.dims[1]);
    const int row_start = static_cast<int>(blockIdx.x) * kQueryTile;
    if (row_start >= rows) return;
    const float scale = rsqrtf(static_cast<float>(dim));

    const auto q_stride0 = params.q.layout.strides[0];
    const auto q_stride1 = params.q.layout.strides[1];
    const auto k_stride0 = params.k.layout.strides[0];
    const auto k_stride1 = params.k.layout.strides[1];
    const auto v_stride0 = params.v.layout.strides[0];
    const auto v_stride1 = params.v.layout.strides[1];
    const auto o_stride0 = params.out.layout.strides[0];
    const auto o_stride1 = params.out.layout.strides[1];

    // Q tile plus the per-row softmax state.
    for (int index = lane; index < kQueryTile * dim; index += kWaveSize) {
        const int row = index / dim;
        const int col = index % dim;
        const int query = row_start + row;
        shared.q_tile[row][col] =
            query < rows
                ? params.q.data[static_cast<size_t>(query) * q_stride0 +
                                static_cast<size_t>(col) * q_stride1]
                : dtype::from_float<T>(0.0f);
        shared.o_tile[row][col] = 0.0f;
    }
    if (lane < kQueryTile) {
        shared.m_row[lane] = -INFINITY;
        shared.l_row[lane] = 0.0f;
        shared.alpha_row[lane] = 1.0f;
    }
    block_sync();

    for (int key_start = 0; key_start < keys; key_start += kKeyTile) {
        // Stage the next K/V tile; out-of-range rows are zeroed and masked in
        // the score step.
        for (int index = lane; index < kKeyTile * dim; index += kWaveSize) {
            const int row = index / dim;
            const int col = index % dim;
            const int key = key_start + row;
            const bool valid = key < keys;
            shared.k_tile[row][col] =
                valid ? params.k.data[static_cast<size_t>(key) * k_stride0 +
                                      static_cast<size_t>(col) * k_stride1]
                      : dtype::from_float<T>(0.0f);
            shared.v_tile[row][col] =
                valid ? params.v.data[static_cast<size_t>(key) * v_stride0 +
                                      static_cast<size_t>(col) * v_stride1]
                      : dtype::from_float<T>(0.0f);
        }
        block_sync();

        // S = Q * K^T / sqrt(D); keys past the end are masked to -inf so they
        // contribute nothing to either the max or the denominator.
        for (int index = lane; index < kQueryTile * kKeyTile;
             index += kWaveSize) {
            const int row = index / kKeyTile;
            const int column = index % kKeyTile;
            float score = -INFINITY;
            if (key_start + column < keys) {
                score = 0.0f;
                for (int col = 0; col < dim; ++col) {
                    score += dtype::to_float(shared.q_tile[row][col]) *
                             dtype::to_float(shared.k_tile[column][col]);
                }
                score *= scale;
            }
            shared.s_tile[row][column] = score;
        }
        block_sync();

        // Row statistics: kLanesPerRow lanes share a query row and reduce with
        // two shuffles inside their four-lane group.
        const int stats_row = lane / kLanesPerRow;
        const int stats_part = lane % kLanesPerRow;
        float tile_max = -INFINITY;
        for (int c = 0; c < kColumnsPerLane; ++c) {
            const float score =
                shared.s_tile[stats_row][stats_part * kColumnsPerLane + c];
            tile_max = score > tile_max ? score : tile_max;
        }
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 2));

        const float previous_max = shared.m_row[stats_row];
        const float next_max = previous_max > tile_max ? previous_max : tile_max;
        const float alpha = expf(previous_max - next_max);

        float tile_sum = 0.0f;
        for (int c = 0; c < kColumnsPerLane; ++c) {
            const int column = stats_part * kColumnsPerLane + c;
            const float weight =
                expf(shared.s_tile[stats_row][column] - next_max);
            shared.p_tile[stats_row][column] = dtype::from_float<T>(weight);
            tile_sum += weight;
        }
        tile_sum += __shfl_xor(tile_sum, 1);
        tile_sum += __shfl_xor(tile_sum, 2);

        if (stats_part == 0) {
            shared.m_row[stats_row] = next_max;
            shared.l_row[stats_row] =
                shared.l_row[stats_row] * alpha + tile_sum;
            shared.alpha_row[stats_row] = alpha;
        }
        block_sync();

        // O = O * alpha + P * V for the whole output tile.
        for (int index = lane; index < kQueryTile * dim; index += kWaveSize) {
            const int out_row = index / dim;
            const int out_col = index % dim;
            float sum = 0.0f;
            for (int key = 0; key < kKeyTile; ++key) {
                sum += dtype::to_float(shared.p_tile[out_row][key]) *
                       dtype::to_float(shared.v_tile[key][out_col]);
            }
            shared.o_tile[out_row][out_col] =
                shared.o_tile[out_row][out_col] *
                    shared.alpha_row[out_row] +
                sum;
        }
        block_sync();
    }

    // Out = O / l, written straight from LDS.
    for (int index = lane; index < kQueryTile * dim; index += kWaveSize) {
        const int out_row = index / dim;
        const int out_col = index % dim;
        const int query = row_start + out_row;
        if (query >= rows) continue;
        const float inverse = shared.l_row[out_row] != 0.0f
                                  ? 1.0f / shared.l_row[out_row]
                                  : 0.0f;
        params.out.data[static_cast<size_t>(query) * o_stride0 +
                        static_cast<size_t>(out_col) * o_stride1] =
            dtype::from_float<T>(shared.o_tile[out_row][out_col] * inverse);
    }
}

}  // namespace dmk::attention
