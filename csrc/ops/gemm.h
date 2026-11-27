#pragma once

#include "ops/dumma.h"
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
// load() stages the first A/B k-tile, run() walks every k-tile accumulating in a
// DUMMA register fragment, store() writes the tile out.  The arithmetic is the
// hardware matrix unit (`ops/dumma.h`), not scalar FFMA: its ceiling is the
// tensor-core rate rather than the ~18 TFLOP/s an FFMA loop can reach.
//
// Tile arithmetic, for the float32 path:
//
//   accumulator : DUFragment<accumulator, 16, 16, 4, float>   (registers)
//   per k-step  : four m16n16k4 mma, each reading a 16x4 slice of A and a 4x16
//                 slice of B straight out of the staged pages (leading
//                 dimension = kTile, so no packing is needed)
//   epilogue    : store_matrix -> C page -> HBM (store())
//
// Two accumulator fragments alternate because `du_mma_sync` writes a different
// destination than it reads; the pair is the same trick the deleted standalone
// kernel used.
//
// One consumer wave computes, and that is not an oversight: DUMMA is a
// whole-wave instruction, so four waves cannot split one 16x16 tile the way
// scalar lanes could.
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

    //: K width of one DUMMA step in the float32 path (m16n16k4).
    static constexpr int kStep = 4;

    using AFragment = ::du::dumma::DUFragment<::du::dumma::matrix_a, kTile, kTile,
                                              kStep, float,
                                              ::du::dumma::row_major>;
    using BFragment = ::du::dumma::DUFragment<::du::dumma::matrix_b, kTile, kTile,
                                              kStep, float,
                                              ::du::dumma::row_major>;
    using Accumulator = ::du::dumma::DUFragment<::du::dumma::accumulator, kTile,
                                                kTile, kStep, float>;

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
        s.wait_page_ready(pid_a);
        s.wait_page_ready(pid_b);
        // The C page needs no wait or clear here: run()'s `store_matrix` writes
        // all 256 elements, and every instruction takes at least one k-step.
        s.wait_page_ready(s.pid(2));

        stage_operands(s, g, inst, 0, lane);

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

    // The whole GEMM for this output tile: every k-tile, accumulated in a DUMMA
    // register fragment.
    //
    // Only the first consumer wave takes part -- DUMMA is a whole-wave
    // instruction, so the tile cannot be split across waves the way a scalar
    // loop could.
    __device__ void run(State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);
        const int pid_a = s.pid(0);
        const int pid_b = s.pid(1);
        const int pid_c = s.pid(2);

        s.semaphores()[0].wait(0);
        if (wave == 0) {
            const float* a_page = static_cast<const float*>(s.pages[pid_a].ptr());
            const float* b_page = static_cast<const float*>(s.pages[pid_b].ptr());
            float* c_page = static_cast<float*>(s.pages[pid_c].ptr());

            Accumulator current;
            Accumulator next;
            dumma::fill_fragment(current, 0.0f);
            dumma::fill_fragment(next, 0.0f);
            AFragment a_fragment;
            BFragment b_fragment;

            for (int k = 0; k < inst.k_tiles; ++k) {
                if (k > 0) {
                    stage_operands(s, g, inst, k, lane);
                    lds::wave_barrier();
                    __threadfence_block();
                }
                // One k-tile is 16 wide; the float path consumes it in four
                // m16n16k4 steps, reading the slices straight out of the pages
                // (leading dimension kTile, so no repacking).
#pragma unroll
                for (int step = 0; step < kTile; step += kStep) {
                    dumma::ld_matrix(a_fragment, a_page + step,
                                     static_cast<unsigned>(kTile));
                    dumma::ld_matrix(b_fragment, b_page + step * kTile,
                                     static_cast<unsigned>(kTile));
                    dumma::mma(next, a_fragment, b_fragment, current);
                    Accumulator previous = current;
                    current = next;
                    next = previous;
                }
                lds::wave_barrier();
                __threadfence_block();
            }
            // Epilogue: the accumulator goes to the C *page*, never straight to
            // HBM.  Two reasons: a fragment exists only in this wave's
            // registers, and the storer is a different wave that can only read
            // shared memory; and the page is what `store()` already copies out.
            dumma::store_matrix(c_page, current, static_cast<unsigned>(kTile),
                                dumma::native::mem_row_major);
            // Publish the tile before the arrival below: every lane of this wave
            // finishes its write, then the fence orders LDS, then the count
            // reaches the storer's wait.
            lds::wave_barrier();
            __threadfence_block();
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
