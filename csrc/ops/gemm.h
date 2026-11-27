#pragma once

#include "ops/tensor.h"
#include "ops/copy.h"
#include "ops/dtype.h"
#include "ops/op.h"
#include "ops/state.h"
#include "ops/tile.h"

#include <type_traits>

// Page-based GEMM operator -- the only GEMM in this tree.
//
// One instruction is one whole output tile over the full K of the matrix:
// load() stages the first A/B k-tile and clears C, run() walks every k-tile and
// accumulates into the C page, store() writes C.  The tile arithmetic is a
// straightforward scalar loop: a DUMMA-fragment version lived here too, but two
// implementations of one operator is one too many, and the persistent op never
// called it.
// Pages: lid 0 = A k-tile, lid 1 = B k-tile, lid 2 = C tile; the pages the
// operator does not own are released by load().
namespace ops {

template <typename Globals>
class GemmOp final : public BaseOp<Globals> {
public:
    static constexpr int opcode = config::kOpGemm;
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

    __device__ int init_semaphores(State& s,
                                   const Globals&) const override {
        s.semaphores()[0].init(1);
        s.semaphores()[1].init(config::kConsumerWaves);
        return 2;
    }

    // Stage the A/B k-tile at (k * 16) into the two operand pages. The runtime
    // calls it once (k = 0), run() calls it for the remaining k.
    __device__ static void stage_operands(State& s, const Globals& g,
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

    __device__ void load(State& s, const Globals& g) const override {
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
    __device__ void run(State& s, const Globals& g) const override {
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

    __device__ void store(State& s, const Globals& g) const override {
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
