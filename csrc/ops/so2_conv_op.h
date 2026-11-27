#pragma once

// SO(2) convolution as a page-based megakernel operator (ops::BaseOp).
//
// One instruction is one operator invocation: an edge range of one SO(2)
// convolution layer, forward.  run() owns the tile loop over that range, so the
// runtime only drives one prefetch (load) and one epilogue (store), exactly as
// op.h prescribes.
//
// Why this op exists at all: the hipified upstream kernel is built around a
// 32-lane logical unit (a CUDA warp) while a DCU wave is 64 lanes, so half of
// every warp-idle analysis is wasted and the layout is padded to match.  Here
// the lane axis *is* the channel axis -- lane t owns channel t -- which makes
// every HBM access a fully coalesced 64-float row (256 B) and removes the
// packed-lane staging entirely (c12's 32-lane design forced a per-warp stage of
// 26 x 33 floats; see docs/so2-conv-hip.md).  C <= 64 keeps it one lane per
// channel; a wider focus needs a channel loop, which is the documented limit of
// this version.
//
// One edge, one channel, all stages in registers:
//
//   x[d]            (DIM,)      node features of the edge's source atom
//   xl[r] = sum_d   R[e][r][d] * x[d]                     rotate to the local frame
//   for layer 0..L-1:
//     z[r]    = sum_r' K[e][layer][r][r'] * xl[r']         block-diagonal content
//     gate[r] = sum_r' G[e][layer][r][r'] * xl[r']         block-diagonal gate
//     xl[r]  += silu(z[r]) * sigmoid(gate[r])              gated mixing + residual
//   y[d]   = sum_r   R[e]^T[d][r] * xl[r]                 rotate back
//   msg[e][d] = alpha[e] * y[d]                           envelope / attention
//
// Everything per edge is staged in that wave's own page; the intermediate
// xl/z/gate never leave registers, which is the whole point -- the dense graph
// materialises five (E, RED, C) tensors for this span.
//
// Instruction words (config::kInstructionWidth = 32):
//   [0] opcode
//   [1] edge_begin
//   [2] edge_end (exclusive)
//   [3] dim  = (lmax + 1)^2
//   [4] red  = 3 * lmax + 1
//   [5] layers
//   [6] channels
//
// Page layout (one page per consumer wave, so four edges are in flight):
//   lid w    wave w's per-edge tables: R (red*dim), R^T (dim*red),
//            K and G for every layer (layers*2*red*red), alpha (1)
//            -> (2*red*dim + 2*layers*red*red + 1) floats
//   lid 4,5  unused; load() releases them immediately
//
// Dynamic semaphores:
//   [0] tables_ready  load lane 0 arrives once; run waits
//   [1] msg_done      each consumer wave arrives once; store waits
//
// Not in this op, on purpose:
//   * the destination reduction (edge message -> atom sum, atomics-free CSR);
//     that is SegmentSumOp's job and a separate instruction, which is the
//     megakernel split rather than the fused kernel's single-span structure;
//   * the backward.  The hipified port has one; this v2 does forward only until
//     the forward is measured (docs/so2-conv-hip.md section 6 records why the
//     backward is where the ridge bites hardest).
//   * the Wigner run packing: the host produces R from the packed runs with the
//     existing dpa4_wigner_runs op, so this op consumes a dense per-edge matrix
//     instead of re-deriving the block-diagonal algebra.

#include "ops/mode.h"
#include "ops/so2_conv_math.h"

#include <cstddef>

#if OPS_MODE == OPS_MODE_PERSISTENT
#include "ops/copy.h"
#include "ops/op.h"
#include "ops/state.h"
#else
#include <hip/hip_runtime.h>
#endif

namespace ops::so2_conv {

#if OPS_MODE == OPS_MODE_PERSISTENT
// The interface lives one level up, in `ops`; this operator keeps its own
// namespace so its helpers (table budget, instruction encoder) do not widen the
// op library's surface.
using ops::BaseOp;
#endif

//: Host-side pointers this op reads. The kernel's Globals must carry one.
//
// The host has to add a member of this type to its Globals struct; the persistent
// form is then instantiated as So2Conv::Op<Globals> and reaches it as g.so2.
struct So2ConvTensorPack {
    const float* x;      // (n_node, dim, channels) node features
    const long* src;     // (n_edge,) source node of each edge
    const float* rot;    // (n_edge, red, dim) local-frame rotation R
    const float* rot_t;  // (n_edge, dim, red) its transpose R^T
    const float* kc;     // (n_edge, layers, 2, red, red) content then gate kernels
    const float* alpha;  // (n_edge,) envelope / attention weight
    float* msg;          // (n_edge, dim, channels) per-edge message, head-gated
};

//: Opcode of this operator. 4 is the free slot in the dmk table (0 noop,
//: 1 elementwise, 2 gemm, 3 embedding, 5 attention), and a free constant so the
//: host encoder does not have to instantiate So2Conv::Op<Globals> for its value.
inline constexpr int kSo2ConvOpcode = 4;

#if OPS_MODE == OPS_MODE_ORDINARY
// Declared here, defined below: So2Conv::launch() names both.
__global__ void so2_conv_span_kernel(const So2ConvTensorPack pack, long edges,
                                     int dim, int red, int layers, int channels);
__host__ inline unsigned so2_conv_span_blocks(long edges);
#endif

//: The operator: one class, two forms.
//:
//:   So2Conv::Op<Globals>  persistent form, a BaseOp subclass the megakernel VM
//:                         drives (compiled in persistent mode only -- it needs
//:                         the runtime's BaseOp and State)
//:   So2Conv::launch()     ordinary form, one kernel launch (ordinary mode only)
//:
//: Both call the same ops::so2::* math and the same table helpers, so the forms
//: cannot disagree about the arithmetic, the layouts or the budgets.  The kernel
//: itself stays a free function: whether a static member can be `__global__` is a
//: compiler detail not worth betting the build on, and `launch()` gives the class
//: its entry point either way.
struct So2Conv {
    static constexpr int kEdgeWaves = kSo2ConvEdgeWaves;
    static constexpr int kBlockThreads = kSo2ConvBlockThreads;

    //: Grid for an edge count, clamped so a huge edge array still launches.
    __host__ inline static unsigned blocks(long edges) {
        return so2_conv_span_blocks(edges);
    }

#if OPS_MODE == OPS_MODE_ORDINARY
    //: Ordinary entry point: one launch of so2_conv_span_kernel.
    __host__ inline static void launch(const So2ConvTensorPack& pack, long edges,
                                       int dim, int red, int layers,
                                       int channels, hipStream_t stream) {
        so2_conv_span_kernel<<<blocks(edges), kSo2ConvBlockThreads, 0, stream>>>(
            pack, edges, dim, red, layers, channels);
    }
#endif

#if OPS_MODE == OPS_MODE_PERSISTENT

    template <typename Globals>
    class Op final : public BaseOp<Globals> {
public:
    static constexpr int opcode = kSo2ConvOpcode;
    static constexpr int kLaneChannels = config::kWaveSize;  // 64

    //: Floats one wave stages for its edge.
    __host__ __device__ static constexpr int table_floats(int dim, int red,
                                                         int layers) {
        return 2 * red * dim + 2 * layers * red * red + 1;
    }

    struct Instruction {
        int begin;
        int end;
        int dim;
        int red;
        int layers;
        int channels;

        __device__ explicit Instruction(const int* words)
            : begin(words[1]),
              end(words[2]),
              dim(words[3]),
              red(words[4]),
              layers(words[5]),
              channels(words[6]) {}
    };

    __device__ explicit Op(const int* instruction)
        : BaseOp<Globals>(instruction) {}

    __device__ int release_lid(int query) const override { return query; }

    __device__ int init_semaphores(State& s, const Globals&) const override {
        s.semaphores()[0].init(1);
        s.semaphores()[1].init(config::kConsumerWaves);
        return 2;
    }

    // --- stage one edge's tables into `page` -------------------------------
    //
    // The loader wave does this for the first wave-many edges so a consumer's
    // slow start overlaps the previous instruction; run() calls it again for its
    // own later edges.  Lane i copies the i-th 16 B chunk, so it is the same
    // coalesced form as the reference op's copy loop.
    __device__ static void stage_tables(const So2ConvTensorPack& pack, int edge,
                                        float* page, const Instruction& inst,
                                        int lane) {
        const size_t rot_n = static_cast<size_t>(inst.red) * inst.dim;
        const size_t kc_n = static_cast<size_t>(inst.layers) * 2 * inst.red * inst.red;
        float* rot = page;
        float* rot_t = rot + rot_n;
        float* kc = rot_t + rot_n;
        float* alpha = kc + kc_n;

        const float* src_rot = pack.rot + static_cast<size_t>(edge) * rot_n;
        const float* src_rot_t = pack.rot_t + static_cast<size_t>(edge) * rot_n;
        const float* src_kc = pack.kc + static_cast<size_t>(edge) * kc_n;

        for (size_t i = static_cast<size_t>(lane) * 4; i < rot_n;
             i += config::kWaveSize * 4)
            copy::store_vec<float4>(rot + i, copy::load_vec<float4>(src_rot + i));
        for (size_t i = static_cast<size_t>(lane) * 4; i < rot_n;
             i += config::kWaveSize * 4)
            copy::store_vec<float4>(rot_t + i,
                                    copy::load_vec<float4>(src_rot_t + i));
        for (size_t i = static_cast<size_t>(lane) * 4; i + 3 < kc_n;
             i += config::kWaveSize * 4)
            copy::store_vec<float4>(kc + i, copy::load_vec<float4>(src_kc + i));
        // Tail of the kernel table when it is not a whole number of vectors.
        for (size_t i = (kc_n / 4) * 4 + lane; i < kc_n; i += config::kWaveSize)
            kc[i] = src_kc[i];
        if (lane == 0) alpha[0] = pack.alpha[edge];
    }

    __device__ void load(State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const auto& pack = g.so2;

        for (int w = 0; w < config::kConsumerWaves; ++w) {
            const int edge = inst.begin + w;
            const int pid = s.pid(w);
            s.wait_page_ready(pid);
            if (edge < inst.end) {
                float* page = static_cast<float*>(s.pages[pid].ptr());
                stage_tables(pack, edge, page, inst, lane);
            }
            lds::wave_barrier();
            __threadfence_block();
        }

        if (lane == 0) s.semaphores()[0].arrive();

        // Pages this op does not use: release so their phases advance.
        for (int lid = config::kConsumerWaves; lid < config::kNumPages; ++lid) {
            if (lane == lid - config::kConsumerWaves) {
                const int unused = s.pid(lid);
                s.wait_page_ready(unused);
                s.finish_page(unused, config::kConsumerWaves);
            }
        }
    }

    __device__ void run(State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int pid = s.pid(wave);
        const auto& pack = g.so2;
        const int channel = lane;
        const bool active = channel < inst.channels;

        const size_t rot_n = static_cast<size_t>(inst.red) * inst.dim;
        const size_t kc_n = static_cast<size_t>(inst.layers) * 2 * inst.red * inst.red;
        float* page = static_cast<float*>(s.pages[pid].ptr());

        s.semaphores()[0].wait(0);

        // run() owns the tile loop: this wave walks its own edges, restaging the
        // tables into its own page when it moves past the prefetched one.
        for (int edge = inst.begin + wave; edge < inst.end;
             edge += config::kConsumerWaves) {
            if (edge != inst.begin + wave) {
                lds::wave_barrier();
                stage_tables(pack, edge, page, inst, lane);
                lds::wave_barrier();
                __threadfence_block();
            }

            const float* rot = page;
            const float* rot_t = rot + rot_n;
            const float* kc = rot_t + rot_n;
            const float alpha = page[2 * rot_n + kc_n];

            if (active) {
                // Node features of the edge's source atom: one 256 B row per
                // depth, fully coalesced across the wave. The arithmetic itself
                // is ops/so2_conv_math.h, shared with the ordinary kernel so the
                // two entry points cannot drift.
                const long src = pack.src[edge];
                const float* xrow = pack.x + static_cast<size_t>(src) *
                                                inst.dim * inst.channels;
                float x[so2::kMaxDim];
                for (int d = 0; d < inst.dim; ++d)
                    x[d] = xrow[static_cast<size_t>(d) * inst.channels + channel];

                float xl[so2::kMaxRed];
                so2::rotate_to_local(rot, x, xl, inst.dim, inst.red);
                const long block = static_cast<long>(inst.red) * inst.red;
                for (int layer = 0; layer < inst.layers; ++layer) {
                    const float* content = kc + 2 * block * layer;
                    so2::gated_layer(content, content + block, xl, inst.red);
                }

                // Rotate back and apply the envelope/attention weight here, so
                // the weight's placement stays visible at this call site.
                float* out = pack.msg + static_cast<size_t>(edge) *
                                            inst.dim * inst.channels;
                float y[so2::kMaxDim];
                so2::rotate_back(rot_t, xl, y, inst.dim, inst.red);
                for (int d = 0; d < inst.dim; ++d)
                    out[static_cast<size_t>(d) * inst.channels + channel] = alpha * y[d];
            }

            lds::wave_barrier();
            __threadfence_block();
        }

        if (lane == 0) s.semaphores()[1].arrive();
    }

    __device__ void store(State& s, const Globals&) const override {
        const int wave = static_cast<int>(threadIdx.x / config::kWaveSize);
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int pid = s.pid(wave);

        // The message is already in HBM: run() wrote it as its last stage, so
        // store() only has to wait for the waves and release the pages. A future
        // SegmentSumOp consumes msg and never needs this page again.
        s.semaphores()[1].wait(0);
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.finish_page(pid, config::kConsumerWaves);
    }
};

#endif  // OPS_MODE == OPS_MODE_PERSISTENT
};

// --- host-side contract helpers -------------------------------------------
//
// The launch path (megakernel.cpp, another translation unit) has to build the
// instruction words and check that one edge's tables fit a page; both are here
// so the op owns its own encoding rather than leaving magic indices at the call
// site.

//: Bytes one wave stages for its edge.
__host__ __device__ inline size_t so2_conv_table_bytes(int dim, int red,
                                                       int layers) {
    return sizeof(float) *
           static_cast<size_t>(2 * red * dim + 2 * layers * red * red + 1);
}

#if OPS_MODE == OPS_MODE_PERSISTENT

//: Whether a shape fits the per-wave page. Checked at launch; the profiled shape
//: is also pinned below so a config change cannot silently overrun a page.
__host__ __device__ inline bool so2_conv_tables_fit(int dim, int red,
                                                    int layers) {
    return so2_conv_table_bytes(dim, red, layers) <=
           static_cast<size_t>(config::kPageSize);
}

//: The profiled checkpoint's shape: lmax = 3 -> dim 16, red 10, four layers.
//: 4,484 B of the 8,192 B page, so a wider lmax or a deeper stack still fits
//: until roughly lmax = 5.
static_assert(so2_conv_table_bytes(16, 10, 4) <= config::kPageSize,
              "the profiled SO(2) conv tables must fit one page");
static_assert(so2_conv_tables_fit(25, 13, 4),
              "lmax = 4 must still fit one page");

//: Build the instruction words the op reads. ``words`` must hold
//: config::kInstructionWidth ints; everything after word 6 is left to the
//: runtime (zeroed here so a stale slot cannot be misread).
inline void encode_so2_conv_instruction(int* words, int begin, int end, int dim,
                                        int red, int layers, int channels) {
    for (int i = 0; i < config::kInstructionWidth; ++i) words[i] = 0;
    words[0] = kSo2ConvOpcode;
    words[1] = begin;
    words[2] = end;
    words[3] = dim;
    words[4] = red;
    words[5] = layers;
    words[6] = channels;
}

#endif  // OPS_MODE == OPS_MODE_PERSISTENT

#if OPS_MODE == OPS_MODE_ORDINARY

// Ordinary mode: the same per-edge span as one plain launch.
//
// One wave per edge, lane of the wave = channel, so every HBM access is a full
// coalesced row and the span stays in registers.  `kSo2ConvEdgeWaves` edges are in
// flight per block; the loop strides by that so any edge count is covered --
// the same tail behaviour the persistent form has, where the wave count plays
// that role.
//
// The destination reduction is not part of this operator: `msg` is per edge and
// summing it onto atoms is ops::segment_sum's job.

//: Edges in flight per block: one per consumer wave of the persistent form, so
//: the two modes exercise the same amount of parallelism per block.
inline constexpr int kSo2ConvEdgeWaves = 4;

//: One wave64 per edge, `kSo2ConvEdgeWaves` waves per block.
inline constexpr int kSo2ConvBlockThreads = kSo2ConvEdgeWaves * 64;

__global__ void so2_conv_span_kernel(
    const So2ConvTensorPack pack, long edges, int dim, int red, int layers,
    int channels) {
    const int wave = static_cast<int>(threadIdx.x) / 64;
    const int lane = static_cast<int>(threadIdx.x) % 64;
    const int channel = lane;
    const bool active = channel < channels;
    const long stride = static_cast<long>(gridDim.x) * kSo2ConvEdgeWaves;

    for (long edge = static_cast<long>(blockIdx.x) * kSo2ConvEdgeWaves + wave;
         edge < edges; edge += stride) {
        float y[ops::so2::kMaxDim];
        float xl[ops::so2::kMaxRed];
        float envelope = 0.0f;
        if (active) {
            const float* xrow = pack.x + static_cast<size_t>(pack.src[edge]) *
                                             dim * channels;
            float xv[ops::so2::kMaxDim];
            for (int d = 0; d < dim; ++d)
                xv[d] = xrow[static_cast<size_t>(d) * channels + channel];
            const long rot_n = static_cast<long>(red) * dim;
            const long block = static_cast<long>(red) * red;
            envelope = pack.alpha[edge];
            ops::so2::edge_span(xv, pack.rot + static_cast<size_t>(edge) * rot_n,
                                pack.rot_t + static_cast<size_t>(edge) * rot_n,
                                pack.kc + static_cast<size_t>(edge) * layers * 2 * block,
                                dim, red, layers, y, xl);
        }
        // Idle lanes (channels < 64) write nothing, so a narrow focus cannot
        // touch the next edge's message.
        if (active) {
            float* out = pack.msg + static_cast<size_t>(edge) * dim * channels;
            for (int d = 0; d < dim; ++d)
                out[static_cast<size_t>(d) * channels + channel] = envelope * y[d];
        }
    }
}

//: Grid for an edge count, clamped so a huge edge array still launches.
__host__ inline unsigned so2_conv_span_blocks(long edges) {
    const long needed = (edges + kSo2ConvEdgeWaves - 1) / kSo2ConvEdgeWaves;
    const long limit = 65535;  // conservative grid.x bound across DCU parts
    return static_cast<unsigned>(needed < limit ? needed : limit);
}

#endif  // OPS_MODE == OPS_MODE_ORDINARY

}  // namespace ops
