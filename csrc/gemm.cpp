// An ordinary dense GEMM: no instruction stream, no pages, no roles.
//
// The persistent kernels in `csrc/dmk/` exist to fuse many short operators.  For one
// dense product they carry bookkeeping that buys nothing, and their 16x16 output tile
// gives an arithmetic intensity of 2 MAC/byte -- a quarter of what a 64x64 block tile
// gets, which is the whole reason a library GEMM is faster.  This file is the plain
// kernel to measure against:
//
//   block tile   64 x 64         (8 MAC/byte, four times the persistent tiling)
//   wave         16 rows of it, all four column groups -> four accumulators per lane
//   k step       8               (DUMMA m16n16k8, the widest float32 fragment)
//   staging      static LDS: A 64x8, B 8x64, C 64x64 = 24 KiB
//   threads      256 (four waves), __launch_bounds__ so nothing is clamped
//
// 24 KiB of static LDS also leaves room for two blocks per CU, which is the ordinary
// way to hide the staging latency that the persistent kernel hides with a software
// pipeline.
//
// Edges are handled by predicate rather than by requiring multiples of the tile: the
// stage zero-fills what is out of bounds and the epilogue skips it, so M, N and K may
// be anything (performance is only claimed for shapes that fill the tile).

#include "ops/config.h"
#include "ops/dumma.h"
#include "ops/tensor.h"

#include "checks.h"
#include "ops.h"

#include <cstdio>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

// `csrc/gemm.cpp` sits outside the operator headers' namespace (that is the point:
// it is not an op), so the two names this file borrows are aliased here.
namespace config = ops::config;
namespace dumma = ops::dumma;

constexpr int kMT = 64;   // block tile rows
constexpr int kNT = 64;   // block tile cols
constexpr int kKT = 8;    // k step: m16n16k8 for float32
constexpr int kWaves = 4; // 16 rows each
constexpr int kThreads = kWaves * config::kWaveSize;
constexpr int kGroups = kNT / 16;      // 16x16 column groups per wave

// Row padding, and the reason it is padding rather than swizzling:
// `du_load_matrix_sync(fragment&, const T* p, unsigned ldm)` carries no layout except
// a leading dimension -- it reads `p + row * ldm`, so the tile it consumes must be
// plain row-major.  `ops/tensor.h` has `ShmemSwizzleLayout` (the XOR-by-row one), but
// wiring it in means either a DTK loader that takes a swizzle mode or hand-rolled
// `ds_read` fragment packing; padding keeps the layout the loader understands and only
// changes the stride, so it cannot be silently wrong.
//
// The numbers: a row of the A stage is kKT = 8 floats = 32 B, so rows start 8 banks
// apart and only four distinct bank offsets exist -> rows alias.  A stride of 12
// (48 B, still 16 B aligned) gives eight distinct offsets before repeating.  The B
// stage row is kNT = 64 floats = 256 B, exactly a multiple of the 128 B bank cycle, so
// *every* row starts at the same bank; a stride of 68 shifts by four banks per row and
// spreads eight rows across the cycle.
// Build-time knobs, so the machine can answer "which of these four costs what"
// without a profiler: each one changes exactly one decision, and the others stay put.
//
//   DPA4_DENSE_PAD      row padding in floats (0 disables: the conflict hypothesis)
//   DPA4_DENSE_KSTAGE   k8 steps staged per buffer (1 = a barrier every 8 of k)
//   DPA4_DENSE_CCHUNK   1 = per-wave 4 KiB epilogue buffer instead of one 16 KiB
#ifndef DPA4_DENSE_PAD
#define DPA4_DENSE_PAD 4
#endif
#ifndef DPA4_DENSE_KSTAGE
#define DPA4_DENSE_KSTAGE 1
#endif
#ifndef DPA4_DENSE_CCHUNK
#define DPA4_DENSE_CCHUNK 0
#endif
constexpr int kPad = DPA4_DENSE_PAD;   // floats per row, keeps every stride 16 B aligned
constexpr int kSteps = DPA4_DENSE_KSTAGE;   // k8 steps staged per buffer
constexpr int kStageK = kKT * kSteps;       // k values staged per buffer
constexpr int kStrideA = kStageK + kPad;    // 12 with the defaults
constexpr int kStrideB = kNT + kPad;        // 68
constexpr int kStageA = kMT * kStrideA;
constexpr int kStageB = kStageK * kStrideB;
// The epilogue buffer is the largest single LDS allocation; a quarter of it, written
// one wave at a time, buys occupancy at the cost of four extra barriers.
constexpr int kChunk = DPA4_DENSE_CCHUNK ? 16 : kMT;
constexpr int kStageC = kChunk * kNT;

static_assert(kWaves * 16 == kMT, "each wave owns exactly 16 rows");
static_assert(kGroups == 4, "four column groups per wave");
static_assert(kThreads == 256, "the staging scheme assumes one element per thread");

using AFragment = ::du::dumma::DUFragment<::du::dumma::matrix_a, 16, 16, kKT, float,
                                          ::du::dumma::row_major>;
using BFragment = ::du::dumma::DUFragment<::du::dumma::matrix_b, 16, 16, kKT, float,
                                          ::du::dumma::row_major>;
using Accumulator =
    ::du::dumma::DUFragment<::du::dumma::accumulator, 16, 16, kKT, float>;

//: What a launch reads and writes.  Plain HBM views, no instruction stream.
struct DenseGemm {
    ops::tensor::Tensor<const float, 2, ops::tensor::MemorySpace::HBM> a;  // (M, K)
    ops::tensor::Tensor<const float, 2, ops::tensor::MemorySpace::HBM> b;  // (K, N)
    ops::tensor::Tensor<float, 2, ops::tensor::MemorySpace::HBM> c;        // (M, N)

    DMK_HOST_DEVICE bool valid() const {
        if (a.data == nullptr || b.data == nullptr || c.data == nullptr) return false;
        const size_t m = c.shape.dims[0];
        const size_t n = c.shape.dims[1];
        const size_t k = a.shape.dims[1];
        return m != 0 && n != 0 && k != 0 && a.shape.dims[0] == m &&
               b.shape.dims[0] == k && b.shape.dims[1] == n;
    }
};

//: Elements of the A and B stages, and how many each thread carries in registers.
constexpr int kSlotsA = kMT * kStageK;
constexpr int kSlotsB = kStageK * kNT;
constexpr int kRegsA = (kSlotsA + kThreads - 1) / kThreads;
constexpr int kRegsB = (kSlotsB + kThreads - 1) / kThreads;

//: Read the next k-block of A and B into registers.
//:
//: Two buffers and this split are what turn two barriers per k-block into one: the
//: global reads are issued *before* the mma of the current block, the LDS writes land
//: in the other buffer, and a single `__syncthreads()` then covers both "the buffer I
//: am about to read is complete" and "the buffer I just read may be reused".
__device__ __forceinline__ void load_block(float (&regs_a)[kRegsA],
                                           float (&regs_b)[kRegsB],
                                           const DenseGemm& g, size_t m0, size_t n0,
                                           size_t k0) {
    const size_t stride_a = g.a.layout.strides[0];
    const size_t stride_b = g.b.layout.strides[0];
    constexpr int kColumns = kStageK * kKT;
#pragma unroll
    for (int chunk = 0; chunk < kRegsA; ++chunk) {
        const int slot = static_cast<int>(threadIdx.x) + chunk * kThreads;
        if (slot >= kSlotsA) break;
        const int row = slot / kColumns;
        const int col = slot % kColumns;
        const size_t absolute_row = m0 + static_cast<size_t>(row);
        const size_t absolute_col = k0 + static_cast<size_t>(col);
        regs_a[chunk] =
            (absolute_row < g.a.shape.dims[0] &&
             absolute_col < g.a.shape.dims[1])
                ? g.a.data[absolute_row * stride_a + absolute_col]
                : 0.0f;
    }
#pragma unroll
    for (int chunk = 0; chunk < kRegsB; ++chunk) {
        const int slot = static_cast<int>(threadIdx.x) + chunk * kThreads;
        if (slot >= kSlotsB) break;
        const int row = slot / kNT;
        const int col = slot % kNT;
        const size_t absolute_row = k0 + static_cast<size_t>(row);
        const size_t absolute_col = n0 + static_cast<size_t>(col);
        regs_b[chunk] =
            (absolute_row < g.b.shape.dims[0] &&
             absolute_col < g.b.shape.dims[1])
                ? g.b.data[absolute_row * stride_b + absolute_col]
                : 0.0f;
    }
}

//: Write those registers into a stage buffer (padded rows).
__device__ __forceinline__ void store_block(float* dst_a, float* dst_b,
                                            const float (&regs_a)[kRegsA],
                                            const float (&regs_b)[kRegsB]) {
    constexpr int kColumns = kStageK * kKT;
#pragma unroll
    for (int chunk = 0; chunk < kRegsA; ++chunk) {
        const int slot = static_cast<int>(threadIdx.x) + chunk * kThreads;
        if (slot >= kSlotsA) break;
        dst_a[static_cast<size_t>(slot / kColumns) * kStrideA +
              slot % kColumns] = regs_a[chunk];
    }
#pragma unroll
    for (int chunk = 0; chunk < kRegsB; ++chunk) {
        const int slot = static_cast<int>(threadIdx.x) + chunk * kThreads;
        if (slot >= kSlotsB) break;
        dst_b[static_cast<size_t>(slot / kNT) * kStrideB + slot % kNT] =
            regs_b[chunk];
    }
}

//: Copy `kChunk` rows of the C stage out to HBM, one thread per element.
__device__ __forceinline__ void write_chunk(const DenseGemm& g, const float* stage,
                                            size_t m0, size_t n0) {
    const size_t stride = g.c.layout.strides[0];
    for (int slot = static_cast<int>(threadIdx.x); slot < kStageC;
         slot += kThreads) {
        const int row = slot / kNT;
        const int col = slot % kNT;
        const size_t absolute_row = m0 + static_cast<size_t>(row);
        const size_t absolute_col = n0 + static_cast<size_t>(col);
        if (absolute_row < g.c.shape.dims[0] && absolute_col < g.c.shape.dims[1])
            g.c.data[absolute_row * stride + absolute_col] = stage[slot];
    }
}

__global__ void __launch_bounds__(kThreads)
dense_gemm_kernel(const DenseGemm g) {
    // Two of each: the k-block being computed and the one being filled.  With one
    // buffer an iteration needs two barriers (wrote-then-read, read-then-overwrite);
    // with two, a single barrier orders both, and the global reads for k+1 sit in
    // registers while the mma of k runs.
    __shared__ float stage_a_buffer[2][kStageA];
    __shared__ float stage_b_buffer[2][kStageB];
    __shared__ float stage_c_buffer[kStageC];

    const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);
    const size_t m0 = static_cast<size_t>(blockIdx.y) * kMT;
    const size_t n0 = static_cast<size_t>(blockIdx.x) * kNT;
    const size_t k_total = g.a.shape.dims[1];

    // Two accumulator sets, alternating: `du_mma_sync` writes a destination other than
    // the one it reads, and a pair per column group is 32 registers -- affordable.
    Accumulator current[kGroups];
    Accumulator spare[kGroups];
    AFragment a_fragment;
    BFragment b_fragment;
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
        dumma::fill_fragment(current[group], 0.0f);
        dumma::fill_fragment(spare[group], 0.0f);
    }

    // Prologue: fill buffer 0, then the loop's single barrier is all that is needed.
    float regs_a[kRegsA];
    float regs_b[kRegsB];
    load_block(regs_a, regs_b, g, m0, n0, 0);
    store_block(stage_a_buffer[0], stage_b_buffer[0], regs_a, regs_b);
    __syncthreads();

    for (size_t k0 = 0, buffer = 0; k0 < k_total;
         k0 += kStageK, buffer ^= 1) {
        const size_t next = k0 + kStageK;
        // Issue the next block's global reads now: their latency overlaps the mma
        // below instead of being paid at the top of the next iteration.
        if (next < k_total) load_block(regs_a, regs_b, g, m0, n0, next);

        // This wave owns rows [wave*16, wave*16+16); its A fragment is shared by the
        // four column groups, which is where the reuse comes from.  With more than one
        // k8 step staged, the barrier count per k drops by that factor.
        const float* a_stage = stage_a_buffer[buffer];
        const float* b_stage = stage_b_buffer[buffer];
#pragma unroll
        for (int step = 0; step < kSteps; ++step) {
            dumma::ld_matrix(a_fragment,
                             const_cast<float*>(a_stage) +
                                 wave * 16 * kStrideA + step * kKT,
                             static_cast<unsigned>(kStrideA));
#pragma unroll
            for (int group = 0; group < kGroups; ++group) {
                dumma::ld_matrix(b_fragment,
                                 const_cast<float*>(b_stage) +
                                     step * kKT * kStrideB + group * 16,
                                 static_cast<unsigned>(kStrideB));
                dumma::mma(spare[group], a_fragment, b_fragment, current[group]);
                Accumulator previous = current[group];
                current[group] = spare[group];
                spare[group] = previous;
            }
        }

        // One barrier: every thread has finished reading this buffer *and* finished
        // writing the other one, which is what the next iteration reads.
        if (next < k_total) {
            store_block(stage_a_buffer[buffer ^ 1], stage_b_buffer[buffer ^ 1],
                        regs_a, regs_b);
        }
        __syncthreads();
    }

    // Epilogue through LDS: the accumulator lives in registers and the copy out is a
    // plain vectorised store, which also lets one wave write rows another block owns.
#if DPA4_DENSE_CCHUNK
    // One wave's 16 rows at a time through a quarter-sized buffer: four barriers per
    // instruction in exchange for 12 KiB of LDS, which is occupancy.
    for (int owner = 0; owner < kWaves; ++owner) {
        if (wave == owner) {
#pragma unroll
            for (int group = 0; group < kGroups; ++group) {
                dumma::store_matrix(stage_c_buffer + group * 16, current[group],
                                    static_cast<unsigned>(kNT),
                                    dumma::native::mem_row_major);
            }
        }
        __syncthreads();
        write_chunk(g, stage_c_buffer, m0 + static_cast<size_t>(owner) * 16, n0);
        __syncthreads();
    }
#else
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
        dumma::store_matrix(stage_c_buffer + wave * 16 * kNT + group * 16,
                            current[group], static_cast<unsigned>(kNT),
                            dumma::native::mem_row_major);
    }
    __syncthreads();

#if !DPA4_DENSE_CCHUNK
    write_chunk(g, stage_c_buffer, m0, n0);
#endif
}

}  // namespace

#ifndef __HIP_DEVICE_COMPILE__
Tensor gemm_dense_hip(Tensor matrix_a, Tensor matrix_b) {
    f32(matrix_a);
    f32(matrix_b);
    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2, "matrices must be 2-D");
    TORCH_CHECK(matrix_a.size(1) == matrix_b.size(0),
                "expected A (M,K) and B (K,N); got K = ", matrix_a.size(1), " and ",
                matrix_b.size(0));
    same_device(matrix_a, {matrix_b});

    const int64_t m = matrix_a.size(0);
    const int64_t n = matrix_b.size(1);
    const at::cuda::OptionalCUDAGuard guard(matrix_a.device());
    Tensor matrix_c = torch::empty({m, n}, matrix_a.options());

    DenseGemm g{};
    at::Tensor a_handle = matrix_a;
    at::Tensor b_handle = matrix_b;
    at::Tensor c_handle = matrix_c;
    ops::tensor::from_torch(a_handle, g.a);
    ops::tensor::from_torch(b_handle, g.b);
    ops::tensor::from_torch(c_handle, g.c);
    TORCH_CHECK(g.valid(), "the dense GEMM operands are not consistent");

    const dim3 grid(static_cast<unsigned>((n + kNT - 1) / kNT),
                    static_cast<unsigned>((m + kMT - 1) / kMT));

    // `DPA4_DEBUG_GEMM=1` prints what decides occupancy for this kernel -- the facts a
    // profiler would show first, available without one.  LDS per block is the tile
    // staging (A, B, C); registers decide how many blocks fit next to it.
    if (std::getenv("DPA4_DEBUG_GEMM") != nullptr) {
        hipFuncAttributes attributes{};
        if (hipFuncGetAttributes(&attributes,
                                 reinterpret_cast<const void*>(dense_gemm_kernel)) ==
            hipSuccess) {
            std::printf(
                "dense gemm: regs=%d static_lds=%zu B threads=%d grid=(%u,%u) "
                "kMT=%d kNT=%d kKT=%d\n",
                attributes.numRegs,
                static_cast<size_t>(attributes.sharedSizeBytes), kThreads,
                grid.x, grid.y, kMT, kNT, kKT);
            std::fflush(stdout);
        }
    }
    dense_gemm_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(g);
    launch_check();
    return matrix_c;
}
#endif  // __HIP_DEVICE_COMPILE__
