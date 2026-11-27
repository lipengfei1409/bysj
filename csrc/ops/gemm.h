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
// **All consumer waves compute, one output tile each.**  DUMMA is a whole-wave
// instruction, so a wave cannot split a 16x16 tile -- what it can split is a
// *run* of tiles.  One instruction therefore covers up to `kConsumerWaves`
// consecutive tiles along N:
//
//     wave w  ->  tile (m0, n0 + w * 16)    (skipped when it is past the N edge)
//
// Each wave stages its own copy of the A slice and of its own B slice, so the
// waves share no pages and need no cross-wave semaphore in the k loop: the only
// hand-off is the C tile at the end (each wave's own page -> the storer).
// Staging costs 4x the A traffic of a single-wave op; that is the price of four
// waves working without a per-k-tile handshake.
//
// Pages: lid 0 = A copies (one 16x16 per wave), lid 1 = B copies (one 16x16 per
// wave), lid 2 + w = wave w's C tile.  The pages the operator does not own are
// released by load().
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

    //: Output tiles one instruction covers: one per consumer wave.
    static constexpr int kTilesPerInstruction = config::kConsumerWaves;
    //: Elements one wave's staged A or B copy occupies in its page.
    static constexpr int kSliceElements = kTile * kTile;

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
        // [0] loader -> consumers (the pages are yours), [1] every consumer wave
        // -> storer (all four C tiles are written).
        s.semaphores()[0].init(1);
        s.semaphores()[1].init(config::kConsumerWaves);
        return 2;
    }

    //: Column of the tile this wave owns, or a value past the N edge if none.
    __device__ static int tile_column(const Instruction& inst, int wave,
                                      size_t n) {
        const int column = inst.n0 + wave * kTile;
        return static_cast<size_t>(column) < n ? column : -1;
    }

    // Stage this wave's A slice and B slice for k-tile ``k``.
    //
    // A copy: rows m0..m0+15, columns k*16..k*16+15          (identical for every
    //        wave, which is the replication cost of not sharing a page)
    // B copy: rows k*16..k*16+15, columns this wave's tile   (16 wide)
    //
    // Both land at ``wave * kSliceElements`` in their page, so the four waves
    // never touch each other's bytes and no barrier between them is needed.
    __device__ static void stage_slices(State& s, const Globals& g,
                                        const Instruction& inst, int k, int wave,
                                        int lane) {
        const size_t n = g.matrix_c.shape.dims[1];
        const int column = tile_column(inst, wave, n);
        const auto shape = tensor::Shape<2>{{kTile, kTile}};
        const auto layout = tensor::StridedLayout<2>{{kTile, 1}};
        const size_t offset = static_cast<size_t>(wave) * kSliceElements;

        const size_t a_origin[2] = {static_cast<size_t>(inst.m0),
                                    static_cast<size_t>(k * kTile)};
        const auto a_src = tile::make_tile_at(g.matrix_a, kTileShape, a_origin);
        tile::load_tile(
            tile::make_lds_view(static_cast<float*>(s.pages[s.pid(0)].ptr()) +
                                    offset,
                                shape, layout),
            a_src, lane, config::kWaveSize);

        if (column < 0) return;  // this wave has no tile: leave its B slice alone
        const size_t b_origin[2] = {static_cast<size_t>(k * kTile),
                                    static_cast<size_t>(column)};
        const auto b_src = tile::make_tile_at(g.matrix_b, kTileShape, b_origin);
        tile::load_tile(
            tile::make_lds_view(static_cast<float*>(s.pages[s.pid(1)].ptr()) +
                                    offset,
                                shape, layout),
            b_src, lane, config::kWaveSize);
    }

    __device__ void load(State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);

        // The consumer waves own pages 0 and 1 (A and B copies) plus one C page
        // each.  The loader only has to hand the pages over; staging is the
        // consumers' job, because each wave stages for itself.
        s.wait_page_ready(s.pid(0));
        s.wait_page_ready(s.pid(1));
        for (int w = 0; w < kTilesPerInstruction; ++w) s.wait_page_ready(s.pid(2 + w));
        (void)inst;

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[0].arrive();

        // Pages the operator does not own.
        if (lane >= 2 + kTilesPerInstruction && lane < config::kNumPages) {
            const int unused = s.pid(lane);
            s.wait_page_ready(unused);
            s.finish_page(unused, config::kConsumerWaves);
        }
        (void)wave;
    }

    // Every consumer wave runs this, on its own output tile, with its own
    // accumulator and its own staged slices.
    __device__ void run(State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);
        const size_t n = g.matrix_c.shape.dims[1];
        const int column = tile_column(inst, wave, n);

        s.semaphores()[0].wait(0);

        float* a_page = static_cast<float*>(s.pages[s.pid(0)].ptr()) +
                        static_cast<size_t>(wave) * kSliceElements;
        float* b_page = static_cast<float*>(s.pages[s.pid(1)].ptr()) +
                        static_cast<size_t>(wave) * kSliceElements;
        float* c_page = static_cast<float*>(s.pages[s.pid(2 + wave)].ptr());

        Accumulator current;
        Accumulator next;
        AFragment a_fragment;
        BFragment b_fragment;
        dumma::fill_fragment(current, 0.0f);
        dumma::fill_fragment(next, 0.0f);

        if (column >= 0) {
            // The k loop is the hot path: it is fully unrolled over the four
            // m16n16k4 steps of a 16-wide k-tile, and carries no barrier --
            // each wave reads only the slices it staged itself.
            for (int k = 0; k < inst.k_tiles; ++k) {
                stage_slices(s, g, inst, k, wave, lane);
                lds::wave_barrier();
                __threadfence_block();
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
                // Free this wave's slices for the next k-tile.
                lds::wave_barrier();
                __threadfence_block();
            }
            // Epilogue: this wave's tile lands in its own C page, never in HBM --
            // the storer is another wave and can only read shared memory.
            dumma::store_matrix(c_page, current, static_cast<unsigned>(kTile),
                                dumma::native::mem_row_major);
            lds::wave_barrier();
            __threadfence_block();
        }
        if (lane == 0) s.semaphores()[1].arrive();
    }

    // The storer copies every tile this instruction produced: one per consumer
    // wave, skipping the waves whose column is past the N edge.
    __device__ void store(State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const size_t n = g.matrix_c.shape.dims[1];

        s.semaphores()[1].wait(0);

        // Every lane of the storer wave takes part: this is not the consumer
        // role, so there is no wave index to gate on.
        {
            for (int w = 0; w < kTilesPerInstruction; ++w) {
                const int column = tile_column(inst, w, n);
                if (column < 0) continue;
                const size_t c_origin[2] = {static_cast<size_t>(inst.m0),
                                            static_cast<size_t>(column)};
                const auto c_dst = tile::make_tile_at(g.matrix_c, kTileShape,
                                                      c_origin);
                const auto c_src = tile::make_lds_view(
                    static_cast<const float*>(s.pages[s.pid(2 + w)].ptr()),
                    c_dst.shape, tensor::StridedLayout<2>{{kTile, 1}});
                tile::copy_tile(c_dst, c_src, lane, config::kWaveSize);
            }
            lds::wave_barrier();
            __threadfence_block();
            if (lane == 0) {
                for (int w = 0; w < kTilesPerInstruction; ++w)
                    s.finish_page(s.pid(2 + w), config::kConsumerWaves);
                s.finish_page(s.pid(0), config::kConsumerWaves);
                s.finish_page(s.pid(1), config::kConsumerWaves);
            }
        }
    }

};

}  // namespace ops
