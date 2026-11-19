#pragma once

// Standalone single-head attention operator.
//
//   out[m, :] = softmax_j( dot(q[m, :], k[j, :]) / sqrt(D) ) * v[j, :]
//
// Q is [M, D], K and V are [N, D] and out is [M, D]; every view may be strided.
// D must be 1..kMaxHeadDim, so one query row fits in two registers per lane.
//
// The element type is a template parameter: Q/K/V may be float, __half or
// __hip_bfloat16 and the output follows the input. All arithmetic happens in
// float and the loads/stores convert, which keeps fp16/bf16 exact enough for
// the softmax while avoiding a second kernel for each dtype.
//
// One wave64 computes one query row and a block runs kRowsPerBlock rows:
//   - lane l owns query elements l, l + 64, ...,
//   - for every key the wave reduces the partial dot product (six shuffles) and
//     every lane then holds the same score,
//   - the softmax is online (flash-style), so K and V are read once:
//         alpha  = exp(old_max - new_max)      (0 on the first key)
//         weight = exp(score - new_max)
//         sum    = sum * alpha + weight
//         acc    = acc * alpha + weight * v[j]
//     and out = acc / sum once the last key is folded in.
//
// The query row, the running max/sum and the accumulator stay in registers, so
// the operator needs no LDS and can later become a megakernel op without
// competing for pages. Tiling K/V through LDS is the next optimization once
// this version is validated.

#include "config.h"
#include "dtype.h"
#include "tensor.h"

#include <cmath>

namespace dmk::attention {

using tensor::MemorySpace;
using tensor::Tensor;

inline constexpr int kMaxHeadDim = 128;
inline constexpr int kWaveSize = config::kWaveSize;
inline constexpr int kMaxVectors = kMaxHeadDim / kWaveSize;  // 2 per lane
inline constexpr int kRowsPerBlock = 4;
inline constexpr int kThreadsPerBlock = kRowsPerBlock * kWaveSize;

static_assert(kMaxHeadDim % kWaveSize == 0,
              "the head dim must map to whole vectors per lane");

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

// Grid size for a launch with kThreadsPerBlock threads.
inline unsigned attention_blocks(size_t rows) {
    return static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
}

// Sum across the 64 lanes; every lane ends up with the total.
__device__ inline float wave_sum(float value) {
#pragma unroll
    for (int offset = kWaveSize / 2; offset > 0; offset >>= 1)
        value += __shfl_xor(value, offset);
    return value;
}

// One query row, computed by one wave. All lanes call this together.
template <typename T>
__device__ inline void attention_row(const AttentionParamsT<T>& params,
                                     size_t row, int lane) {
    const size_t dim = params.out.shape.dims[1];
    const size_t keys = params.k.shape.dims[0];
    const float scale = rsqrtf(static_cast<float>(dim));

    // The query row never changes, so keep it in registers.
    float query[kMaxVectors] = {};
#pragma unroll
    for (int vector = 0; vector < kMaxVectors; ++vector) {
        const size_t index = static_cast<size_t>(lane) + vector * kWaveSize;
        if (index < dim) {
            query[vector] = dtype::to_float(
                params.q.data[row * params.q.layout.strides[0] +
                              index * params.q.layout.strides[1]]);
        }
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[kMaxVectors] = {};

    for (size_t key = 0; key < keys; ++key) {
        const size_t k_row = key * params.k.layout.strides[0];
        const size_t v_row = key * params.v.layout.strides[0];

        float partial = 0.0f;
#pragma unroll
        for (int vector = 0; vector < kMaxVectors; ++vector) {
            const size_t index = static_cast<size_t>(lane) + vector * kWaveSize;
            if (index < dim) {
                partial += query[vector] *
                           dtype::to_float(
                               params.k.data[k_row +
                                             index * params.k.layout.strides[1]]);
            }
        }
        const float score = wave_sum(partial) * scale;

        const float next_max = score > running_max ? score : running_max;
        // -inf - next_max is -inf, so the first key rescales to zero.
        const float alpha = expf(running_max - next_max);
        const float weight = expf(score - next_max);
        running_sum = running_sum * alpha + weight;

#pragma unroll
        for (int vector = 0; vector < kMaxVectors; ++vector) {
            const size_t index = static_cast<size_t>(lane) + vector * kWaveSize;
            float value = 0.0f;
            if (index < dim) {
                value = dtype::to_float(
                    params.v.data[v_row +
                                  index * params.v.layout.strides[1]]);
            }
            acc[vector] = acc[vector] * alpha + weight * value;
        }
        running_max = next_max;
    }

    const float inverse = running_sum != 0.0f ? 1.0f / running_sum : 0.0f;
#pragma unroll
    for (int vector = 0; vector < kMaxVectors; ++vector) {
        const size_t index = static_cast<size_t>(lane) + vector * kWaveSize;
        if (index < dim) {
            params.out.data[row * params.out.layout.strides[0] +
                            index * params.out.layout.strides[1]] =
                dtype::from_float<T>(acc[vector] * inverse);
        }
    }
}

// Entry point for a kernel whose block is a multiple of 64 threads. Every wave
// of the block takes one query row, so the block's row count is blockDim.x / 64.
template <typename T>
__device__ inline void run_attention(const AttentionParamsT<T>& params) {
    if (!params.valid()) return;
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    const int wave = static_cast<int>(threadIdx.x / kWaveSize);
    const size_t rows_per_block =
        static_cast<size_t>(blockDim.x) / kWaveSize;
    const size_t row =
        static_cast<size_t>(blockIdx.x) * rows_per_block + wave;
    if (row >= params.out.shape.dims[0]) return;
    attention_row<T>(params, row, lane);
}

}  // namespace dmk::attention
