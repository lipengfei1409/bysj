#pragma once

// Page-based flash-attention operator for the megakernel VM.
//
// One instruction is one (query tile, key tile) step: it builds a 16x16 score
// tile and folds it into the 16xD output tile of one query tile, so the
// loader wave of the next instruction can stage the next K/V tile while the
// consumer wave is still computing. The Q, O and softmax-state pages persist
// across the instructions of a query tile; only K/V are refilled every step.
//
// Pages (one 8 KiB page each, so D <= kMaxHeadDim = 128):
//   lid 0  Q tile     16 x D, element type T
//   lid 1  K tile     16 x D, element type T
//   lid 2  V tile     16 x D, element type T
//   lid 3  O tile     16 x D, float accumulator
//   lid 4  state      m[16], l[16], alpha[16] (float)
//   lid 5  scratch    S 16x16 (float) followed by P 16x16 (T)
//
// Instruction words:
//   [0] opcode (5)
//   [1] query tile index   (rows 16*q .. 16*q+15)
//   [2] key tile index
//   [3] number of keys N   (the last tile is masked past the end)
//   [4] head dim D
//   [5] 1 on the last key tile of the query tile, which writes O / l
//
// Dynamic semaphores:
//   [0] data_ready   load lane 0 arrives once; run waits
//   [1] data_done    each consumer wave arrives once; store waits
//
// Memory behaviour: the tile copies keep the 16 B chunk layout used by the
// standalone operators (lane i takes the i-th chunk of a row) and the scalar
// fallback orders lanes along the smallest HBM stride, so column-major K/V
// views stay coalesced. The pages themselves are unpadded (a padded 132-float
// row would not fit a 16 x 128 float tile in one page), so the S-tile column
// reads still pay the bank conflicts the standalone kernel had.

#include "config.h"
#include "dtype.h"
#include "op.h"
#include "state.h"
#include "tensor.h"

#include <cmath>

namespace dmk::attention {

using tensor::MemorySpace;
using tensor::Tensor;

inline constexpr int kQueryTile = 16;
inline constexpr int kKeyTile = 16;
inline constexpr int kMaxHeadDim = 128;
inline constexpr int kRowStride = kMaxHeadDim;  // elements per page row
inline constexpr int kLanesPerRow = config::kWaveSize / kQueryTile;
inline constexpr int kColumnsPerLane = kKeyTile / kLanesPerRow;

inline constexpr int kQueryPage = 0;
inline constexpr int kKeyPage = 1;
inline constexpr int kValuePage = 2;
inline constexpr int kOutputPage = 3;
inline constexpr int kStatePage = 4;
inline constexpr int kScratchPage = 5;

static_assert(kLanesPerRow * kQueryTile == config::kWaveSize,
              "a query tile must map onto whole waves");
static_assert(kKeyTile % kLanesPerRow == 0,
              "key tile must split evenly across a row's lane group");

// O[M, D] = softmax(Q[M, D] * K[N, D]^T / sqrt(D)) * V[N, D].
template <typename T>
struct AttentionGlobalsT {
    using element = T;

    // [blocks][instructions][kInstructionWidth]
    Tensor<const int, 3, MemorySpace::HBM> instructions;
    Tensor<const T, 2, MemorySpace::HBM> q;
    Tensor<const T, 2, MemorySpace::HBM> k;
    Tensor<const T, 2, MemorySpace::HBM> v;
    Tensor<T, 2, MemorySpace::HBM> out;
    int num_instructions = 0;

    DMK_HOST_DEVICE bool valid(uint32_t blocks) const {
        if (num_instructions <= 0) return false;
        if (instructions.data == nullptr) return false;
        if (instructions.shape.dims[0] < blocks ||
            instructions.shape.dims[1] < static_cast<size_t>(num_instructions) ||
            instructions.shape.dims[2] != config::kInstructionWidth)
            return false;
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

}  // namespace dmk::attention

namespace dmk::mk {

template <typename Globals>
class AttentionOp final : public BaseOp<Globals> {
public:
    using T = typename Globals::element;

    static constexpr int opcode = 5;
    static constexpr int kInstructionWidth = config::kInstructionWidth;

    struct Instruction {
        int query_tile;
        int key_tile;
        int num_keys;
        int dim;
        int last;

        __device__ explicit Instruction(const int* words)
            : query_tile(words[1]),
              key_tile(words[2]),
              num_keys(words[3]),
              dim(words[4]),
              last(words[5]) {}
    };

    __device__ explicit AttentionOp(const int* instruction)
        : BaseOp<Globals>(instruction) {}

    // Q/O/state keep their physical page across the instructions of a query
    // tile, so the page order stays the identity.
    __device__ int release_lid(int query) const override { return query; }

    __device__ int init_semaphores(State& s,
                                   const Globals&) const override {
        s.semaphores()[0].init(1);
        s.semaphores()[1].init(config::kConsumerWaves);
        return 2;
    }

    __device__ void load(State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int dim = inst.dim;
        const int rows = static_cast<int>(g.out.shape.dims[0]);
        const int first_row = inst.query_tile * attention::kQueryTile;
        const int first_key = inst.key_tile * attention::kKeyTile;

        const int q_pid = s.pid(attention::kQueryPage);
        const int k_pid = s.pid(attention::kKeyPage);
        const int v_pid = s.pid(attention::kValuePage);
        const int o_pid = s.pid(attention::kOutputPage);
        const int state_pid = s.pid(attention::kStatePage);
        s.wait_page_ready(q_pid);
        s.wait_page_ready(k_pid);
        s.wait_page_ready(v_pid);
        s.wait_page_ready(o_pid);
        s.wait_page_ready(state_pid);
        s.wait_page_ready(s.pid(attention::kScratchPage));

        // The first key tile of a query tile (re)loads Q and resets the
        // output tile and the softmax state; later tiles reuse the pages.
        if (inst.key_tile == 0) {
            T* q_page = static_cast<T*>(s.pages[q_pid].ptr());
            float* o_page = static_cast<float*>(s.pages[o_pid].ptr());
            float* state = static_cast<float*>(s.pages[state_pid].ptr());
            const auto stride0 = g.q.layout.strides[0];
            const auto stride1 = g.q.layout.strides[1];
            for (int index = lane; index < attention::kQueryTile * dim;
                 index += config::kWaveSize) {
                const int row = index / dim;
                const int col = index % dim;
                const int query = first_row + row;
                q_page[row * attention::kRowStride + col] =
                    query < rows
                        ? g.q.data[static_cast<size_t>(query) * stride0 +
                                   static_cast<size_t>(col) * stride1]
                        : dtype::from_float<T>(0.0f);
                o_page[row * attention::kRowStride + col] = 0.0f;
            }
            if (lane < attention::kQueryTile) {
                state[lane] = -INFINITY;
                state[attention::kQueryTile + lane] = 0.0f;
                state[2 * attention::kQueryTile + lane] = 1.0f;
            }
        }

        // K/V tile; rows past the end are zeroed and masked in the scores.
        T* k_page = static_cast<T*>(s.pages[k_pid].ptr());
        T* v_page = static_cast<T*>(s.pages[v_pid].ptr());
        const auto k_stride0 = g.k.layout.strides[0];
        const auto k_stride1 = g.k.layout.strides[1];
        const auto v_stride0 = g.v.layout.strides[0];
        const auto v_stride1 = g.v.layout.strides[1];
        for (int index = lane; index < attention::kKeyTile * dim;
             index += config::kWaveSize) {
            const int row = index / dim;
            const int col = index % dim;
            const int key = first_key + row;
            const bool valid = key < inst.num_keys;
            k_page[row * attention::kRowStride + col] =
                valid ? g.k.data[static_cast<size_t>(key) * k_stride0 +
                                  static_cast<size_t>(col) * k_stride1]
                      : dtype::from_float<T>(0.0f);
            v_page[row * attention::kRowStride + col] =
                valid ? g.v.data[static_cast<size_t>(key) * v_stride0 +
                                  static_cast<size_t>(col) * v_stride1]
                      : dtype::from_float<T>(0.0f);
        }

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[0].arrive();
    }

    __device__ void run(State& s, const Globals& g) const override {
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);
        s.semaphores()[0].wait(0);
        if (wave == 0) run_tile(s, g, lane);
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[1].arrive();
    }

    __device__ void store(State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int dim = inst.dim;

        s.semaphores()[1].wait(0);

        if (inst.last != 0) {
            const float* o_page = static_cast<const float*>(
                s.pages[s.pid(attention::kOutputPage)].ptr());
            const float* state = static_cast<const float*>(
                s.pages[s.pid(attention::kStatePage)].ptr());
            const float* l_row = state + attention::kQueryTile;
            const int rows = static_cast<int>(g.out.shape.dims[0]);
            const int first_row = inst.query_tile * attention::kQueryTile;
            const auto stride0 = g.out.layout.strides[0];
            const auto stride1 = g.out.layout.strides[1];
            for (int index = lane; index < attention::kQueryTile * dim;
                 index += config::kWaveSize) {
                const int row = index / dim;
                const int col = index % dim;
                const int query = first_row + row;
                if (query >= rows) continue;
                const float inverse =
                    l_row[row] != 0.0f ? 1.0f / l_row[row] : 0.0f;
                g.out.data[static_cast<size_t>(query) * stride0 +
                           static_cast<size_t>(col) * stride1] =
                    dtype::from_float<T>(
                        o_page[row * attention::kRowStride + col] * inverse);
            }
        }

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) {
            for (int lid = 0; lid < config::kNumPages; ++lid)
                s.finish_page(s.pid(lid), config::kConsumerWaves);
        }
    }

private:
    // The whole 16x16 score tile and its contribution to the 16xD output,
    // executed by wave 0; the other consumer waves only carry the semaphore.
    __device__ void run_tile(State& s, const Globals&, int lane) const {
        const Instruction inst(this->instruction());
        const int dim = inst.dim;
        const int first_key = inst.key_tile * attention::kKeyTile;

        const T* q_page = static_cast<const T*>(
            s.pages[s.pid(attention::kQueryPage)].ptr());
        const T* k_page = static_cast<const T*>(
            s.pages[s.pid(attention::kKeyPage)].ptr());
        const T* v_page = static_cast<const T*>(
            s.pages[s.pid(attention::kValuePage)].ptr());
        float* o_page = static_cast<float*>(
            s.pages[s.pid(attention::kOutputPage)].ptr());
        float* state = static_cast<float*>(
            s.pages[s.pid(attention::kStatePage)].ptr());
        float* s_tile = static_cast<float*>(
            s.pages[s.pid(attention::kScratchPage)].ptr());
        T* p_tile = reinterpret_cast<T*>(s_tile +
                                         attention::kQueryTile *
                                             attention::kKeyTile);
        float* m_row = state;
        float* l_row = state + attention::kQueryTile;
        float* alpha_row = state + 2 * attention::kQueryTile;

        const float scale = rsqrtf(static_cast<float>(dim));

        // S = Q * K^T / sqrt(D), with keys past the end masked to -inf.
        for (int index = lane; index < attention::kQueryTile * attention::kKeyTile;
             index += config::kWaveSize) {
            const int row = index / attention::kKeyTile;
            const int column = index % attention::kKeyTile;
            float score = -INFINITY;
            if (first_key + column < inst.num_keys) {
                score = 0.0f;
                for (int col = 0; col < dim; ++col) {
                    score += dtype::to_float(
                                 q_page[row * attention::kRowStride + col]) *
                             dtype::to_float(
                                 k_page[column * attention::kRowStride + col]);
                }
                score *= scale;
            }
            s_tile[row * attention::kKeyTile + column] = score;
        }
        lds::wave_barrier();
        __threadfence_block();

        // Row statistics: kLanesPerRow lanes share a row and reduce inside
        // their four-lane group.
        const int stats_row = lane / attention::kLanesPerRow;
        const int stats_part = lane % attention::kLanesPerRow;
        float tile_max = -INFINITY;
        for (int c = 0; c < attention::kColumnsPerLane; ++c) {
            const float score =
                s_tile[stats_row * attention::kKeyTile +
                       stats_part * attention::kColumnsPerLane + c];
            tile_max = score > tile_max ? score : tile_max;
        }
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 2));

        const float previous_max = m_row[stats_row];
        const float next_max = previous_max > tile_max ? previous_max : tile_max;
        const float alpha = expf(previous_max - next_max);

        float tile_sum = 0.0f;
        for (int c = 0; c < attention::kColumnsPerLane; ++c) {
            const int column =
                stats_part * attention::kColumnsPerLane + c;
            const float weight = expf(
                s_tile[stats_row * attention::kKeyTile + column] - next_max);
            p_tile[stats_row * attention::kKeyTile + column] =
                dtype::from_float<T>(weight);
            tile_sum += weight;
        }
        tile_sum += __shfl_xor(tile_sum, 1);
        tile_sum += __shfl_xor(tile_sum, 2);

        if (stats_part == 0) {
            m_row[stats_row] = next_max;
            l_row[stats_row] = l_row[stats_row] * alpha + tile_sum;
            alpha_row[stats_row] = alpha;
        }
        lds::wave_barrier();
        __threadfence_block();

        // O = O * alpha + P * V.
        for (int index = lane; index < attention::kQueryTile * dim;
             index += config::kWaveSize) {
            const int row = index / dim;
            const int col = index % dim;
            float sum = 0.0f;
            for (int key = 0; key < attention::kKeyTile; ++key) {
                sum += dtype::to_float(
                           p_tile[row * attention::kKeyTile + key]) *
                       dtype::to_float(
                           v_page[key * attention::kRowStride + col]);
            }
            o_page[row * attention::kRowStride + col] =
                o_page[row * attention::kRowStride + col] *
                    alpha_row[row] +
                sum;
        }
        lds::wave_barrier();
        __threadfence_block();
    }
};

}  // namespace dmk::mk
