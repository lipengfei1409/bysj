// The persistent GEMM without roles: every wave does the whole pipeline.
//
// `megakernel.cpp` runs the scheduler -- a controller wave publishes instructions, a
// loader stages, a consumer computes, a storer writes back, and per-instruction
// semaphores order the four roles.  That is the right shape when fusing
// *heterogeneous* operators, where one op's store is the next one's load.  For a
// single GEMM the phases never change, and every cross-wave handshake is really
// "wait for the staging I just did", paid in semaphores and page bookkeeping.
//
// This file is the other shape for the same operator:
//
//   * one role.  All `kFlatWaves` waves are workers: no loader, no storer, no
//     controller, no semaphores, no page permutation.
//   * staging is cooperative -- A is copied once by the whole block (it is shared by
//     every wave), B by the wave that needs it -- and `__syncthreads()` replaces the
//     handshakes.
//   * the k loop is double buffered: the loads of k+1 are issued into registers
//     before the mma of k, so global latency hides under arithmetic.  This is the
//     overlap the loader wave used to provide, done by hand and without a producer.
//   * the instruction stream is kept, because it is where the sharding lives: a block
//     still gets a row of tiles, so this kernel and the scheduler one are comparable
//     on the same schedule.
//
// Given up: the cross-instruction pipeline (instruction i+1 is not prefetched while i
// computes).  For a GEMM whose k loop is long -- the regime measured here -- that is
// a fair trade.
//
// LDS: five of six pages.  A0, A1, B0, B1 are the double buffers, C holds one 16x16
// tile per wave and is copied out once per instruction.

#include "dmk/launch.h"
#include "dmk/megakernel.h"  // MkGlobals
#include "ops/dumma.h"

#include "checks.h"
#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace dmk::mk {

//: Worker waves: one output tile each per instruction.  Declared outside the
//: anonymous namespace because the host entry needs the block size.
constexpr int kFlatWaves = 4;
constexpr int kFlatThreads = kFlatWaves * config::kWaveSize;

namespace {

constexpr int kTile = 16;
constexpr int kTileElements = kTile * kTile;

static_assert(kFlatThreads == 256, "the staging scheme assumes one element per thread");
static_assert(kFlatThreads <= 1024, "the block must fit one CU");

//: K width of one DUMMA step in the float32 path (m16n16k4).
constexpr int kStep = 4;

using AFragment = ::du::dumma::DUFragment<::du::dumma::matrix_a, kTile, kTile,
                                          kStep, float, ::du::dumma::row_major>;
using BFragment = ::du::dumma::DUFragment<::du::dumma::matrix_b, kTile, kTile,
                                          kStep, float, ::du::dumma::row_major>;
using Accumulator =
    ::du::dumma::DUFragment<::du::dumma::accumulator, kTile, kTile, kStep, float>;

//: Page ids in the dynamic array.
constexpr int kPageA0 = 0;
constexpr int kPageA1 = 1;
constexpr int kPageB0 = 2;
constexpr int kPageB1 = 3;
constexpr int kPageC = 4;
static_assert(kPageC < config::kNumPages, "the flat GEMM needs five pages");

//: The three words of a GEMM instruction: four tiles from (m0, n0), k_tiles k-steps.
struct Words {
    int m0;
    int n0;
    int k_tiles;
};

__device__ __forceinline__ float* page_base(float* pages, int id) {
    return pages + static_cast<size_t>(id) * (config::kPageSize / sizeof(float));
}

// A k-tile is 256 floats and the block is 256 threads: one element each, so the copy
// is a plain coalesced read across the whole block.
__device__ __forceinline__ void load_a(float& reg, const MkGlobals& g,
                                       const Words& w, int k) {
    const int index = static_cast<int>(threadIdx.x);
    const int row = index / kTile;
    const int col = index % kTile;
    reg = g.matrix_a.data[static_cast<size_t>(w.m0 + row) *
                              g.matrix_a.layout.strides[0] +
                          static_cast<size_t>(k * kTile + col)];
}

__device__ __forceinline__ void store_a(float* dst, float reg) {
    const int index = static_cast<int>(threadIdx.x);
    dst[index] = reg;
}

// This wave's B k-tile: 16 rows of k by 16 columns of n.  A wave is 64 lanes and the
// slice is 256 floats, so each lane moves four consecutive elements -- neighbouring
// lanes touch neighbouring addresses.
__device__ __forceinline__ void load_b(float (&reg)[4], const MkGlobals& g,
                                       const Words& w, int k, int column) {
    const int base = static_cast<int>(threadIdx.x % config::kWaveSize) * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int index = base + j;
        const int row = index / kTile;
        const int col = index % kTile;
        reg[j] = g.matrix_b.data[static_cast<size_t>(k * kTile + row) *
                                     g.matrix_b.layout.strides[0] +
                                 static_cast<size_t>(column + col)];
    }
}

__device__ __forceinline__ void store_b(float* dst, const float (&reg)[4]) {
    const int base = static_cast<int>(threadIdx.x % config::kWaveSize) * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) dst[base + j] = reg[j];
}

//: Every worker runs this for its own tile: stage, accumulate over k with the loads of
//: k+1 overlapping the mma of k, then land the tile in the C page.
__device__ __forceinline__ void run_tile(const MkGlobals& g, const Words& words,
                                         float* pages, int wave, size_t n) {
    const int column = words.n0 + wave * kTile;
    const bool active = static_cast<size_t>(column) < n;

    float* const page_a[2] = {page_base(pages, kPageA0), page_base(pages, kPageA1)};
    float* const page_b[2] = {page_base(pages, kPageB0), page_base(pages, kPageB1)};
    float* const page_c = page_base(pages, kPageC);

    // Prologue: stage k = 0 into buffer 0.
    float a_reg = 0.0f;
    float b_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    load_a(a_reg, g, words, 0);
    if (active) load_b(b_reg, g, words, 0, column);
    store_a(page_a[0], a_reg);
    if (active) store_b(page_b[0] + wave * kTileElements, b_reg);
    __syncthreads();

    Accumulator current;
    Accumulator next;
    dumma::fill_fragment(current, 0.0f);
    dumma::fill_fragment(next, 0.0f);
    AFragment a_fragment;
    BFragment b_fragment;

    for (int k = 0; k < words.k_tiles; ++k) {
        const int buffer = k & 1;
        // Issue the next tile's loads now: the addresses are independent of the mma
        // below, so the memory system works while the matrix unit does.
        if (k + 1 < words.k_tiles) {
            load_a(a_reg, g, words, k + 1);
            if (active) load_b(b_reg, g, words, k + 1, column);
        }
        if (active) {
#pragma unroll
            for (int step = 0; step < kTile; step += kStep) {
                dumma::ld_matrix(a_fragment, page_a[buffer] + step,
                                 static_cast<unsigned>(kTile));
                dumma::ld_matrix(b_fragment,
                                 page_b[buffer] + wave * kTileElements +
                                     step * kTile,
                                 static_cast<unsigned>(kTile));
                dumma::mma(next, a_fragment, b_fragment, current);
                Accumulator previous = current;
                current = next;
                next = previous;
            }
        }
        // The buffer that was just read is free to receive the next tile.
        __syncthreads();
        if (k + 1 < words.k_tiles) {
            store_a(page_a[buffer ^ 1], a_reg);
            if (active) store_b(page_b[buffer ^ 1] + wave * kTileElements, b_reg);
        }
        __syncthreads();
    }

    if (active) {
        dumma::store_matrix(page_c + wave * kTileElements, current,
                            static_cast<unsigned>(kTile),
                            dumma::native::mem_row_major);
    }
    __syncthreads();
}

//: Cooperative write-back: each thread moves one element of every tile that exists.
__device__ __forceinline__ void write_tiles(const MkGlobals& g, const Words& words,
                                            float* pages, size_t n) {
    float* const page_c = page_base(pages, kPageC);
    const int element = static_cast<int>(threadIdx.x);
    const int row = element / kTile;
    const int col = element % kTile;
    const size_t stride = g.matrix_c.layout.strides[0];
#pragma unroll
    for (int wave = 0; wave < kFlatWaves; ++wave) {
        const int column = words.n0 + wave * kTile;
        if (static_cast<size_t>(column) >= n) continue;
        g.matrix_c.data[static_cast<size_t>(words.m0 + row) * stride +
                        static_cast<size_t>(column + col)] =
            page_c[wave * kTileElements + element];
    }
}

}  // namespace

__global__ void __launch_bounds__(kFlatThreads)
gemm_flat(const MkGlobals g) {
    extern __shared__ int dynamic_lds[];
    const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(dynamic_lds);
    const std::uintptr_t aligned =
        (raw + (config::kDynamicAlign - 1)) &
        ~static_cast<std::uintptr_t>(config::kDynamicAlign - 1);
    float* const pages = reinterpret_cast<float*>(aligned);

    const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);
    const size_t n = g.matrix_c.shape.dims[1];
    const size_t instruction_stride = g.instructions.layout.strides[1];
    const size_t word_stride = g.instructions.layout.strides[2];
    const std::int32_t* const row_data =
        g.instructions.data +
        static_cast<size_t>(blockIdx.x) * g.instructions.layout.strides[0];

    for (int index = 0; index < g.num_instructions; ++index) {
        // Three words, read straight from HBM: they are a cache line apart at most and
        // the whole block asks for the same ones.
        const std::int32_t* const words =
            row_data + static_cast<size_t>(index) * instruction_stride;
        const Words instruction{words[word_stride],
                                words[2 * word_stride],
                                words[3 * word_stride]};
        // An instruction with no k-steps would leave C untouched; the host never emits
        // one (K >= 16 everywhere), and skipping keeps the page state clean.
        if (instruction.k_tiles <= 0) continue;

        run_tile(g, instruction, pages, wave, n);
        write_tiles(g, instruction, pages, n);
        __syncthreads();  // C is reused by the next instruction
    }
}

}  // namespace dmk::mk

#ifndef __HIP_DEVICE_COMPILE__
namespace {

using namespace dmk;

bool tileable(int64_t m, int64_t n, int64_t k) {
    return m % 16 == 0 && n % 16 == 0 && k % 16 == 0;
}

}  // namespace

Tensor dmk_gemm_flat_hip(Tensor instructions, Tensor matrix_a, Tensor matrix_b,
                         Tensor matrix_c) {
    TORCH_CHECK(instructions.scalar_type() == torch::kInt &&
                    instructions.is_contiguous() && instructions.dim() == 3 &&
                    instructions.size(2) == ops::config::kInstructionWidth,
                "instructions must be a contiguous int32 (blocks, length, 32) tensor");
    f32(matrix_a);
    f32(matrix_b);
    f32(matrix_c);
    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2 && matrix_c.dim() == 2,
                "matrices must be 2-D");
    const int64_t m = matrix_c.size(0);
    const int64_t n = matrix_c.size(1);
    const int64_t k = matrix_a.size(1);
    TORCH_CHECK(matrix_a.size(0) == m && matrix_b.size(0) == k &&
                    matrix_b.size(1) == n,
                "expected A (M,K), B (K,N), C (M,N)");
    TORCH_CHECK(tileable(m, n, k),
                "M, N and K must be multiples of 16: the flat GEMM works in 16x16 "
                "tiles and has no edge path (got ", m, "x", n, "x", k, ")");
    same_device(matrix_a, {matrix_b, matrix_c});

    const at::cuda::OptionalCUDAGuard guard(matrix_a.device());

    mk::MkGlobals g{};
    at::Tensor instructions_handle = instructions;
    at::Tensor a_handle = matrix_a;
    at::Tensor b_handle = matrix_b;
    at::Tensor c_handle = matrix_c;
    ops::tensor::from_torch(instructions_handle, g.instructions);
    ops::tensor::from_torch(a_handle, g.matrix_a);
    ops::tensor::from_torch(b_handle, g.matrix_b);
    ops::tensor::from_torch(c_handle, g.matrix_c);
    g.num_instructions = static_cast<int>(instructions.size(1));

    const uint32_t blocks = static_cast<uint32_t>(instructions.size(0));
    TORCH_CHECK(g.valid_instructions(blocks),
                "the instruction stream does not match the globals");
    TORCH_CHECK(mk::launch_kernel(mk::gemm_flat, g, blocks,
                                  at::cuda::getCurrentCUDAStream(),
                                  mk::kFlatThreads),
                "the flat persistent GEMM launch failed");
    launch_check();
    return matrix_c;
}
#endif  // __HIP_DEVICE_COMPILE__
