// An ordinary dense GEMM: no instruction stream, no pages, no roles.
//
// The persistent kernels in `csrc/dmk/` exist to fuse many short operators.  For one
// dense product they carry bookkeeping that buys nothing, and their 16x16 output tile
// gives an arithmetic intensity of 2 MAC/byte -- a quarter of what a 64x64 block tile
// gets, which is the whole reason a library GEMM is faster.  This file is the plain
// kernel to measure against:
//
//   block tile   128x64 (10.7 MAC/byte) or 64x64 (8 MAC/byte), chosen per shape
//   wave         32 rows of the tile: two 16-row halves by four 16-wide column groups,
//                i.e. eight accumulators, which is what gives the scheduler something to
//                hide the fragment loads behind
//   k step       8     (DUMMA m16n16k8; float32 has no k16 fragment)
//   staging      two buffers of A and B, filled one k-block ahead through registers, so
//                the global reads overlap the mma instead of preceding it
//   epilogue     one wave's rows through a small C stage, so the whole-tile buffer does
//                not cost the second block per CU
//
// Two tiles are instantiated because one does not fit all shapes: a 128x64 tile halves
// the grid, and a 512^3 product only has 32 such blocks against 120 CUs -- it runs at
// 27% occupancy of the part while the same kernel at 4096^3 reaches 0.52x rocBLAS.
// Picking the tile from the shape costs one branch at launch.
//
// Edges are handled by predicate rather than by requiring multiples of the tile: the
// stage zero-fills what is out of bounds and the epilogue skips it, so M, N and K may be
// anything (performance is only claimed for shapes that fill the tile).

#include "ops/config.h"
#include "ops/dumma.h"
#include "ops/tensor.h"

#include "checks.h"
#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// The kernel lives in a *named* namespace on purpose.  Inside an anonymous one it has
// internal linkage, and the HIP runtime resolves a launch by symbol name -- it then fails
// with "Cannot find Symbol with name: _ZN12_GLOBAL__N_1...dense_kernel...", which aborts
// the process rather than reporting an error.
namespace dense_gemm {

// `csrc/gemm.cpp` sits outside the operator headers' namespace (that is the point:
// it is not an op), so the two names this file borrows are aliased here.
namespace config = ops::config;
namespace dumma = ops::dumma;

// Build-time knobs, so the machine can answer "which of these costs what" without a
// profiler: each one changes exactly one decision, and the others stay put.
//
//   DPA4_DENSE_PAD      row padding in floats (0 disables: the conflict hypothesis)
//   DPA4_DENSE_KSTAGE   k8 steps staged per buffer (1 = a barrier every 8 of k)
//   DPA4_DENSE_CCHUNK   0 = whole-tile epilogue buffer instead of one wave's rows
#ifndef DPA4_DENSE_PAD
#define DPA4_DENSE_PAD 4
#endif
#ifndef DPA4_DENSE_KSTAGE
#define DPA4_DENSE_KSTAGE 1
#endif
#ifndef DPA4_DENSE_CCHUNK
#define DPA4_DENSE_CCHUNK 1
#endif
//: `du_mma_sync` may or may not accept the accumulator it reads as its destination.
//: If it does, the spare accumulator set -- one whole register set per fragment -- becomes
//: unnecessary, which is what would make a 128x128 tile fit in registers.  Build with
//: DPA4_DENSE_INPLACE=1 to try it: the accuracy tests answer the question, since a
//: rejected in-place update produces wrong numbers rather than an error.
#ifndef DPA4_DENSE_INPLACE
#define DPA4_DENSE_INPLACE 0
#endif

constexpr int kPad = DPA4_DENSE_PAD;
constexpr int kSteps = DPA4_DENSE_KSTAGE;
constexpr int kKT = 8;   // m16n16k8: the widest float32 fragment
constexpr int kWaves = 4;
constexpr int kThreads = kWaves * config::kWaveSize;
constexpr int kStageK = kKT * kSteps;   // k values per stage buffer

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

//: Everything about one tile size, derived from (MT, NT) in one place.
//:
//: Row padding, and why it is padding rather than swizzling:
//: `du_load_matrix_sync(fragment&, const T* p, unsigned ldm)` carries no layout except a
//: leading dimension -- it reads `p + row * ldm`, so the tile it consumes must be plain
//: row-major.  `ops/tensor.h` has `ShmemSwizzleLayout` (the XOR-by-row one), but wiring it
//: in means either a DTK loader that takes a swizzle mode or hand-rolled `ds_read`
//: fragment packing; padding keeps the layout the loader understands and only changes the
//: stride, so it cannot be silently wrong.
template <int MT, int NT>
struct Tile {
    static constexpr int mt = MT;
    static constexpr int nt = NT;
    static constexpr int rows_per_wave = MT / kWaves;
    static constexpr int row_halves = rows_per_wave / 16;
    static constexpr int groups = NT / 16;
    static constexpr int fragments = row_halves * groups;

    //: A row is kStageK k values plus padding (48 B / 80 B, both 16 B aligned); a B row is
    //: NT plus padding, which shifts each row by four banks instead of the whole cycle.
    static constexpr int stride_a = kStageK + kPad;
    static constexpr int stride_b = NT + kPad;

    static constexpr int stage_a = MT * stride_a;
    static constexpr int stage_b = kStageK * stride_b;
    //: The epilogue buffer is the largest single allocation at a tall tile (MT*NT floats),
    //: so by default it holds one wave's rows and is written `kWaves` times.
    static constexpr int chunk_rows = DPA4_DENSE_CCHUNK ? rows_per_wave : MT;
    static constexpr int stage_c = chunk_rows * NT;

    static constexpr int slots_a = MT * kStageK;
    static constexpr int slots_b = kStageK * NT;
    static constexpr int regs_a = (slots_a + kThreads - 1) / kThreads;
    static constexpr int regs_b = (slots_b + kThreads - 1) / kThreads;

    static constexpr int lds_bytes = (2 * stage_a + 2 * stage_b + stage_c) * 4;

    static_assert(kWaves * rows_per_wave == MT, "the waves must cover the tile rows");
    static_assert(rows_per_wave % 16 == 0, "a wave owns whole 16-row fragments");
    static_assert(NT % 16 == 0, "the tile width must be whole 16-wide groups");
    static_assert(stride_a * 4 % 16 == 0 && stride_b * 4 % 16 == 0,
                  "a padded row must stay 16 B aligned for the fragment loader");
    static_assert(stage_b == kStageK * stride_b, "B rows are k values");
    static_assert(slots_a == MT * kStageK, "A stage is MT rows of kStageK columns");
    static_assert(slots_b == kStageK * NT, "B stage is kStageK rows of NT columns");
    static_assert(regs_a * kThreads >= slots_a, "the register split must cover the stage");
    static_assert(regs_b * kThreads >= slots_b, "the register split must cover the stage");
};

//: Read the next k-block of A and B into registers.
//:
//: Two stage buffers plus this split are what turn two barriers per k-block into one: the
//: global reads are issued *before* the mma of the current block, the LDS writes land in
//: the other buffer, and a single `__syncthreads()` then covers both "the buffer I am
//: about to read is complete" and "the buffer I just read may be reused".
template <class G>
__device__ __forceinline__ void load_block(float (&regs_a)[G::regs_a],
                                           float (&regs_b)[G::regs_b],
                                           const DenseGemm& g, size_t m0, size_t n0,
                                           size_t k0) {
    const size_t stride_a = g.a.layout.strides[0];
    const size_t stride_b = g.b.layout.strides[0];
#pragma unroll
    for (int chunk = 0; chunk < G::regs_a; ++chunk) {
        const int slot = static_cast<int>(threadIdx.x) + chunk * kThreads;
        if (slot >= G::slots_a) break;
        const int row = slot / kStageK;
        const int col = slot % kStageK;
        const size_t absolute_row = m0 + static_cast<size_t>(row);
        const size_t absolute_col = k0 + static_cast<size_t>(col);
        regs_a[chunk] =
            (absolute_row < g.a.shape.dims[0] &&
             absolute_col < g.a.shape.dims[1])
                ? g.a.data[absolute_row * stride_a + absolute_col]
                : 0.0f;
    }
#pragma unroll
    for (int chunk = 0; chunk < G::regs_b; ++chunk) {
        const int slot = static_cast<int>(threadIdx.x) + chunk * kThreads;
        if (slot >= G::slots_b) break;
        const int row = slot / G::nt;
        const int col = slot % G::nt;
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
template <class G>
__device__ __forceinline__ void store_block(float* dst_a, float* dst_b,
                                            const float (&regs_a)[G::regs_a],
                                            const float (&regs_b)[G::regs_b]) {
#pragma unroll
    for (int chunk = 0; chunk < G::regs_a; ++chunk) {
        const int slot = static_cast<int>(threadIdx.x) + chunk * kThreads;
        if (slot >= G::slots_a) break;
        dst_a[static_cast<size_t>(slot / kStageK) * G::stride_a +
              slot % kStageK] = regs_a[chunk];
    }
#pragma unroll
    for (int chunk = 0; chunk < G::regs_b; ++chunk) {
        const int slot = static_cast<int>(threadIdx.x) + chunk * kThreads;
        if (slot >= G::slots_b) break;
        dst_b[static_cast<size_t>(slot / G::nt) * G::stride_b + slot % G::nt] =
            regs_b[chunk];
    }
}

//: Copy the C stage out to HBM, one thread per element.
template <class G>
__device__ __forceinline__ void write_chunk(const DenseGemm& g, const float* stage,
                                            size_t m0, size_t n0) {
    const size_t stride = g.c.layout.strides[0];
    for (int slot = static_cast<int>(threadIdx.x); slot < G::stage_c;
         slot += kThreads) {
        const int row = slot / G::nt;
        const int col = slot % G::nt;
        const size_t absolute_row = m0 + static_cast<size_t>(row);
        const size_t absolute_col = n0 + static_cast<size_t>(col);
        if (absolute_row < g.c.shape.dims[0] && absolute_col < g.c.shape.dims[1])
            g.c.data[absolute_row * stride + absolute_col] = stage[slot];
    }
}

//: The whole kernel body, parameterised by the tile.
//:
//: `__device__`, not `__global__`: a templated *kernel* is registered under a mangled name
//: this runtime failed to resolve at launch ("Cannot find Symbol with name:
//: ...dense_kernel...TileILi128ELi64E..."), while a plain kernel is registered the way the
//: non-template version always was.  The two instantiations are the wrappers below.
template <class G>
__device__ __forceinline__ void dense_body(const DenseGemm& g) {
    // Two of each: the k-block being computed and the one being filled.  With one buffer
    // an iteration needs two barriers (wrote-then-read, read-then-overwrite); with two, a
    // single barrier orders both, and the global reads for k+1 sit in registers while the
    // mma of k runs.
    __shared__ float stage_a_buffer[2][G::stage_a];
    __shared__ float stage_b_buffer[2][G::stage_b];
    __shared__ float stage_c_buffer[G::stage_c];

    const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);
    const size_t m0 = static_cast<size_t>(blockIdx.y) * G::mt;
    const size_t n0 = static_cast<size_t>(blockIdx.x) * G::nt;
    const size_t k_total = g.a.shape.dims[1];

    // Two accumulator sets, alternating: `du_mma_sync` writes a destination other than
    // the one it reads, and a pair per fragment is what that costs.
    Accumulator current[G::fragments];
#if !DPA4_DENSE_INPLACE
    Accumulator spare[G::fragments];
#endif
    // Every fragment a k-step needs, materialised up front: the B loads and the A loads
    // are independent, so the scheduler issues them together and the first mma starts
    // while the rest are still in flight.  Loading one group at a time inside the mma
    // loop makes each load wait for the one before it.
    AFragment a_fragment[G::row_halves];
    BFragment b_fragment[G::groups];
#pragma unroll
    for (int index = 0; index < G::fragments; ++index) {
        dumma::fill_fragment(current[index], 0.0f);
#if !DPA4_DENSE_INPLACE
        dumma::fill_fragment(spare[index], 0.0f);
#endif
    }

    // Prologue: fill buffer 0, then the loop's single barrier is all that is needed.
    float regs_a[G::regs_a];
    float regs_b[G::regs_b];
    load_block<G>(regs_a, regs_b, g, m0, n0, 0);
    store_block<G>(stage_a_buffer[0], stage_b_buffer[0], regs_a, regs_b);
    __syncthreads();

    for (size_t k0 = 0, buffer = 0; k0 < k_total; k0 += kStageK, buffer ^= 1) {
        const size_t next = k0 + kStageK;
        // Issue the next block's global reads now: their latency overlaps the mma below
        // instead of being paid at the top of the next iteration.
        if (next < k_total) load_block<G>(regs_a, regs_b, g, m0, n0, next);

        const float* a_stage = stage_a_buffer[buffer];
        const float* b_stage = stage_b_buffer[buffer];
#pragma unroll
        for (int step = 0; step < kSteps; ++step) {
#pragma unroll
            for (int half = 0; half < G::row_halves; ++half) {
                dumma::ld_matrix(a_fragment[half],
                                 const_cast<float*>(a_stage) +
                                     (wave * G::rows_per_wave + half * 16) *
                                         G::stride_a +
                                     step * kKT,
                                 static_cast<unsigned>(G::stride_a));
            }
#pragma unroll
            for (int group = 0; group < G::groups; ++group) {
                dumma::ld_matrix(b_fragment[group],
                                 const_cast<float*>(b_stage) +
                                     step * kKT * G::stride_b + group * 16,
                                 static_cast<unsigned>(G::stride_b));
            }
#pragma unroll
            for (int half = 0; half < G::row_halves; ++half) {
#pragma unroll
                for (int group = 0; group < G::groups; ++group) {
                    const int index = half * G::groups + group;
#if DPA4_DENSE_INPLACE
                    dumma::mma(current[index], a_fragment[half], b_fragment[group],
                               current[index]);
#else
                    dumma::mma(spare[index], a_fragment[half], b_fragment[group],
                               current[index]);
                    Accumulator previous = current[index];
                    current[index] = spare[index];
                    spare[index] = previous;
#endif
                }
            }
        }

        // One barrier: every thread has finished reading this buffer *and* finished
        // writing the other one, which is what the next iteration reads.
        if (next < k_total) {
            store_block<G>(stage_a_buffer[buffer ^ 1], stage_b_buffer[buffer ^ 1],
                           regs_a, regs_b);
        }
        __syncthreads();
    }

    // Epilogue through LDS: the accumulator lives in registers and the copy out is a plain
    // store.  With the chunked buffer one wave writes its rows at a time, which is what
    // keeps a tall tile at two blocks per CU.
#if DPA4_DENSE_CCHUNK
    for (int owner = 0; owner < kWaves; ++owner) {
        if (wave == owner) {
#pragma unroll
            for (int half = 0; half < G::row_halves; ++half) {
#pragma unroll
                for (int group = 0; group < G::groups; ++group) {
                    const int index = half * G::groups + group;
                    dumma::store_matrix(stage_c_buffer + (half * 16) * G::nt +
                                            group * 16,
                                        current[index],
                                        static_cast<unsigned>(G::nt),
                                        dumma::native::mem_row_major);
                }
            }
        }
        __syncthreads();
        write_chunk<G>(g, stage_c_buffer,
                       m0 + static_cast<size_t>(owner) * G::rows_per_wave, n0);
        __syncthreads();
    }
#else
#pragma unroll
    for (int half = 0; half < G::row_halves; ++half) {
#pragma unroll
        for (int group = 0; group < G::groups; ++group) {
            const int index = half * G::groups + group;
            dumma::store_matrix(stage_c_buffer +
                                    (wave * G::rows_per_wave + half * 16) * G::nt +
                                    group * 16,
                                current[index], static_cast<unsigned>(G::nt),
                                dumma::native::mem_row_major);
        }
    }
    __syncthreads();
    write_chunk<G>(g, stage_c_buffer, m0, n0);
#endif
}

//: The two tiles this build carries, each behind a plain kernel.
using TileWide = Tile<128, 64>;
using TileSquare = Tile<64, 64>;

__global__ void __launch_bounds__(kThreads)
dense_kernel_wide(const DenseGemm g) {
    dense_body<TileWide>(g);
}

__global__ void __launch_bounds__(kThreads)
dense_kernel_square(const DenseGemm g) {
    dense_body<TileSquare>(g);
}

}  // namespace dense_gemm

#ifndef __HIP_DEVICE_COMPILE__
namespace {

//: The two launch symbols, kept as `const void*` for the attribute queries and as typed
//: pointers for the launches.
const void* const kWideSymbol =
    reinterpret_cast<const void*>(&dense_gemm::dense_kernel_wide);
const void* const kSquareSymbol =
    reinterpret_cast<const void*>(&dense_gemm::dense_kernel_square);

//: Which tile to use for this shape.
//:
//: The wide tile has the better arithmetic intensity and the square one has the bigger
//: grid; at 512^3 the wide tile leaves 32 blocks against 120 CUs, so the square one wins
//: despite doing more traffic per FLOP.
bool prefer_wide(int64_t m, int64_t n) {
    const int64_t wide = ((m + dense_gemm::TileWide::mt - 1) / dense_gemm::TileWide::mt) *
                         ((n + dense_gemm::TileWide::nt - 1) / dense_gemm::TileWide::nt);
    const int64_t square = ((m + dense_gemm::TileSquare::mt - 1) / dense_gemm::TileSquare::mt) *
                           ((n + dense_gemm::TileSquare::nt - 1) / dense_gemm::TileSquare::nt);
    int cus = 0;
    int device = 0;
    if (hipGetDevice(&device) == hipSuccess) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, device) == hipSuccess)
            cus = properties.multiProcessorCount;
    }
    if (cus > 0 && wide < cus && square > wide) return false;
    return wide >= square;
}

bool report(const char* message) {
    std::fprintf(stderr, "dense gemm: %s\n", message);
    return false;
}

}  // namespace

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

    dense_gemm::DenseGemm g{};
    at::Tensor a_handle = matrix_a;
    at::Tensor b_handle = matrix_b;
    at::Tensor c_handle = matrix_c;
    ops::tensor::from_torch(a_handle, g.a);
    ops::tensor::from_torch(b_handle, g.b);
    ops::tensor::from_torch(c_handle, g.c);
    TORCH_CHECK(g.valid(), "the dense GEMM operands are not consistent");

    const bool wide = prefer_wide(m, n);
    const int tile_m = wide ? dense_gemm::TileWide::mt : dense_gemm::TileSquare::mt;
    const int tile_n = wide ? dense_gemm::TileWide::nt : dense_gemm::TileSquare::nt;
    const dim3 grid(static_cast<unsigned>((n + tile_n - 1) / tile_n),
                    static_cast<unsigned>((m + tile_m - 1) / tile_m));

    // `DPA4_DEBUG_GEMM=1` prints what decides occupancy for this kernel -- the facts a
    // profiler would show first, available without one.
    if (std::getenv("DPA4_DEBUG_GEMM") != nullptr) {
        const void* symbol = wide ? kWideSymbol : kSquareSymbol;
        hipFuncAttributes attributes{};
        if (hipFuncGetAttributes(&attributes, symbol) == hipSuccess) {
            std::printf(
                "dense gemm: tile=%dx%d regs=%d static_lds=%zu B threads=%d "
                "grid=(%u,%u) kKT=%d kStageK=%d\n",
                tile_m, tile_n, attributes.numRegs,
                static_cast<size_t>(attributes.sharedSizeBytes), dense_gemm::kThreads, grid.x,
                grid.y, dense_gemm::kKT, dense_gemm::kStageK);
            std::fflush(stdout);
        }
    }

    const hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const unsigned threads = dense_gemm::kThreads;
    if (wide) {
        dense_gemm::dense_kernel_wide<<<grid, threads, 0, stream>>>(g);
    } else {
        dense_gemm::dense_kernel_square<<<grid, threads, 0, stream>>>(g);
    }
    launch_check();
    return matrix_c;
}
#endif  // __HIP_DEVICE_COMPILE__
