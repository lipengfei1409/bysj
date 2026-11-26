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

#include "ops/config.h"
#include "ops/dtype.h"
#include "ops/op.h"
#include "dmk/state.h"
#include "ops/tile.h"
#include "ops/tensor.h"

#include <cmath>

namespace ops::attention {

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

}  // namespace ops::attention

namespace ops::mk {

// Page-based flash-attention operator. One instruction is one whole query
// tile over all keys: load() stages Q and the first K/V tile, run() walks the
// remaining key tiles itself (online softmax, one 16x16 score tile per step),
// and store() writes O / l. Pages: lid 0 Q, 1 K, 2 V, 3 O, 4 state,
// 5 scratch (S and P).
template <typename Globals>
class AttentionOp final : public BaseOp<Globals> {
public:
    using T = typename Globals::element;

    static constexpr int opcode = 5;

    struct Instruction {
        int query_tile;
        int num_keys;
        int dim;

        __device__ explicit Instruction(const int* words)
            : query_tile(words[1]), num_keys(words[2]), dim(words[3]) {}
    };

    __device__ explicit AttentionOp(const int* instruction)
        : BaseOp<Globals>(instruction) {}

    __device__ int release_lid(int query) const override { return query; }

    __device__ int init_semaphores(dmk::mk::State& s,
                                   const Globals&) const override {
        s.semaphores()[0].init(1);
        s.semaphores()[1].init(config::kConsumerWaves);
        return 2;
    }

    // Q tile of this query tile, clamped at the matrix edge.
    __device__ static void stage_query(dmk::mk::State& s, const Globals& g,
                                       const Instruction& inst, int lane) {
        const auto shape = tensor::Shape<2>{
            {attention::kQueryTile, static_cast<size_t>(inst.dim)}};
        const auto layout =
            tensor::StridedLayout<2>{{attention::kRowStride, 1}};
        const size_t q_origin[2] = {
            static_cast<size_t>(inst.query_tile) * attention::kQueryTile, 0};
        const auto q_src = tile::make_tile_at(
            g.q, tile::TileShape<2>{{attention::kQueryTile,
                                     static_cast<size_t>(inst.dim)}},
            q_origin);
        tile::load_tile(
            tile::make_lds_view(
                static_cast<T*>(s.pages[s.pid(attention::kQueryPage)].ptr()),
                shape, layout),
            q_src, lane, config::kWaveSize);
    }

    // K/V tile `key_tile`, clamped at the number of keys.
    __device__ static void stage_keys(dmk::mk::State& s, const Globals& g,
                                      const Instruction& inst, int key_tile,
                                      int lane) {
        const auto shape = tensor::Shape<2>{
            {attention::kKeyTile, static_cast<size_t>(inst.dim)}};
        const auto layout =
            tensor::StridedLayout<2>{{attention::kRowStride, 1}};
        const size_t origin[2] = {
            static_cast<size_t>(key_tile) * attention::kKeyTile, 0};
        const auto tile_shape = tile::TileShape<2>{
            {attention::kKeyTile, static_cast<size_t>(inst.dim)}};
        const auto k_src = tile::make_tile_at(g.k, tile_shape, origin);
        const auto v_src = tile::make_tile_at(g.v, tile_shape, origin);
        tile::load_tile(
            tile::make_lds_view(
                static_cast<T*>(s.pages[s.pid(attention::kKeyPage)].ptr()),
                shape, layout),
            k_src, lane, config::kWaveSize);
        tile::load_tile(
            tile::make_lds_view(
                static_cast<T*>(s.pages[s.pid(attention::kValuePage)].ptr()),
                shape, layout),
            v_src, lane, config::kWaveSize);
    }

    __device__ void load(dmk::mk::State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);

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

        stage_query(s, g, inst, lane);
        stage_keys(s, g, inst, 0, lane);

        const auto shape = tensor::Shape<2>{
            {attention::kQueryTile, static_cast<size_t>(inst.dim)}};
        tile::fill_tile(
            tile::make_lds_view(
                static_cast<float*>(s.pages[o_pid].ptr()), shape,
                tensor::StridedLayout<2>{{attention::kRowStride, 1}}),
            0.0f, lane, config::kWaveSize);

        float* state = static_cast<float*>(s.pages[state_pid].ptr());
        if (lane < attention::kQueryTile) {
            state[lane] = -INFINITY;
            state[attention::kQueryTile + lane] = 0.0f;
            state[2 * attention::kQueryTile + lane] = 1.0f;
        }

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[0].arrive();
    }

    // The whole attention for this query tile: every key tile in turn.
    __device__ void run(dmk::mk::State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);

        s.semaphores()[0].wait(0);
        if (wave == 0) {
            const int key_tiles =
                (inst.num_keys + attention::kKeyTile - 1) /
                attention::kKeyTile;
            for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
                if (key_tile > 0) {
                    stage_keys(s, g, inst, key_tile, lane);
                    lds::wave_barrier();
                    __threadfence_block();
                }
                score_and_accumulate(s, inst, key_tile, lane);
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
        const int dim = inst.dim;

        s.semaphores()[1].wait(0);

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
            const float inverse = l_row[row] != 0.0f ? 1.0f / l_row[row] : 0.0f;
            g.out.data[static_cast<size_t>(query) * stride0 +
                       static_cast<size_t>(col) * stride1] =
                dtype::from_float<T>(
                    o_page[row * attention::kRowStride + col] * inverse);
        }

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) {
            for (int lid = 0; lid < config::kNumPages; ++lid)
                s.finish_page(s.pid(lid), config::kConsumerWaves);
        }
    }

private:
    // One 16x16 score tile: S = Q K^T, online-softmax update, O += P V.
    __device__ void score_and_accumulate(dmk::mk::State& s, const Instruction& inst,
                                         int key_tile, int lane) const {
        const int dim = inst.dim;
        const int first_key = key_tile * attention::kKeyTile;

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
        T* p_tile = reinterpret_cast<T*>(
            s_tile + attention::kQueryTile * attention::kKeyTile);
        float* m_row = state;
        float* l_row = state + attention::kQueryTile;
        float* alpha_row = state + 2 * attention::kQueryTile;

        const float scale = rsqrtf(static_cast<float>(dim));

        for (int index = lane;
             index < attention::kQueryTile * attention::kKeyTile;
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
            const int column = stats_part * attention::kColumnsPerLane + c;
            const float weight =
                expf(s_tile[stats_row * attention::kKeyTile + column] -
                     next_max);
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
    }
};

}  // namespace ops::mk
