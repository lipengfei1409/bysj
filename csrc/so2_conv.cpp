// The SO(2) convolution's per-edge span, as a standalone kernel.
//
// This file is the *problem-specific* half of the tree: `csrc/ops/` holds the
// generic operators that run under the persistent megakernel, while a
// convolution this specialised is written here and launched directly.  Same for
// `csrc/gie.cpp` (zonal scatter / GIE) and the other model-specific kernels.
//
//   msg[e, d, c] = alpha[e] * (R^T[e] @ stack(R[e] @ x[src[e]][:, c]))
//   stack        = layers of  xl += silu(K_l xl) * sigmoid(G_l xl)
//
// Thread mapping: one wave64 per edge, lane of the wave = channel, so every HBM
// access is a full coalesced row (256 B at C = 64) and the whole span stays in
// registers.  A block carries kEdgeWaves edges in flight and strides by that, so
// any edge count is covered.
//
// The arithmetic is in so2_conv_math.h, next to this file, because the
// reference (python/reference/so2_conv.py) and the accuracy test are written
// against exactly these stages.  The destination reduction is not here: `msg` is
// per edge, and summing it onto atoms is kernels.segment_sum's job.

#include "ops.h"
#include "checks.h"
#include "so2_conv_math.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstddef>

namespace {

//: Host-side pointers the kernel reads, in the layout the conv consumes.
struct So2ConvPack {
    const float* x;      // (n_node, dim, channels) node features
    const long* src;     // (n_edge,) source node of each edge
    const float* rot;    // (n_edge, red, dim) local-frame rotation R
    const float* rot_t;  // (n_edge, dim, red) its transpose R^T
    const float* kc;     // (n_edge, layers, 2, red, red) content then gate
    const float* alpha;  // (n_edge,) envelope / attention weight
    float* msg;          // (n_edge, dim, channels) per-edge message
};

//: Edges in flight per block: one wave each.
constexpr int kEdgeWaves = 4;

//: One wave64 per edge.
constexpr int kBlockThreads = kEdgeWaves * 64;

//: Grid for an edge count, clamped so a huge edge array still launches.
__host__ inline unsigned blocks_for(long edges) {
    const long needed = (edges + kEdgeWaves - 1) / kEdgeWaves;
    const long limit = 65535;
    return static_cast<unsigned>(needed < limit ? needed : limit);
}

__global__ void so2_conv_kernel(const So2ConvPack pack, long edges, int dim,
                                int red, int layers, int channels) {
    const int wave = static_cast<int>(threadIdx.x) / 64;
    const int lane = static_cast<int>(threadIdx.x) % 64;
    const int channel = lane;
    const bool active = channel < channels;
    const long stride = static_cast<long>(gridDim.x) * kEdgeWaves;

    for (long edge = static_cast<long>(blockIdx.x) * kEdgeWaves + wave;
         edge < edges; edge += stride) {
        float y[ops::so2::kMaxDim];
        float xl[ops::so2::kMaxRed];
        float envelope = 0.0f;
        if (active) {
            const float* xrow =
                pack.x + static_cast<size_t>(pack.src[edge]) * dim * channels;
            float xv[ops::so2::kMaxDim];
            for (int d = 0; d < dim; ++d)
                xv[d] = xrow[static_cast<size_t>(d) * channels + channel];
            const long rot_n = static_cast<long>(red) * dim;
            const long block = static_cast<long>(red) * red;
            envelope = pack.alpha[edge];
            ops::so2::edge_span(
                xv, pack.rot + static_cast<size_t>(edge) * rot_n,
                pack.rot_t + static_cast<size_t>(edge) * rot_n,
                pack.kc + static_cast<size_t>(edge) * layers * 2 * block, dim,
                red, layers, y, xl);
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

}  // namespace

Tensor so2_conv_hip(Tensor x, Tensor src, Tensor rot, Tensor rot_t,
                    Tensor kc, Tensor alpha) {
  f32(x);
  f32(rot);
  f32(rot_t);
  f32(kc);
  f32(alpha);
  TORCH_CHECK(src.scalar_type() == torch::kLong && src.is_contiguous(),
              "src must be a contiguous int64 (E,) vector");
  TORCH_CHECK(x.dim() == 3, "x must be (n_node, dim, channels)");
  TORCH_CHECK(rot.dim() == 3 && rot_t.dim() == 3,
              "rot and rot_t must be (E, red, dim) and (E, dim, red)");
  TORCH_CHECK(kc.dim() == 5, "kc must be (E, layers, 2, red, red)");

  const long edges = src.size(0);
  const int dim = static_cast<int>(rot.size(2));
  const int red = static_cast<int>(rot.size(1));
  const int layers = static_cast<int>(kc.size(1));
  const int channels = static_cast<int>(x.size(2));

  TORCH_CHECK(rot_t.size(1) == dim && rot_t.size(2) == red,
              "rot_t must be the transpose of rot");
  TORCH_CHECK(rot.size(0) == edges && rot_t.size(0) == edges &&
                  kc.size(0) == edges && alpha.size(0) == edges,
              "every per-edge tensor must have one row per edge");
  TORCH_CHECK(kc.size(2) == 2 && kc.size(3) == red && kc.size(4) == red,
              "kc must hold a content and a gate kernel of (red, red)");
  TORCH_CHECK(dim <= ops::so2::kMaxDim && red <= ops::so2::kMaxRed,
              "this kernel supports dim, red <= ", ops::so2::kMaxDim);
  TORCH_CHECK(channels <= 64,
              "one lane per channel requires channels <= 64, got ", channels);
  same_device(x, {rot, rot_t, kc, alpha});

  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto msg = torch::zeros({edges, dim, channels}, x.options());
  if (edges == 0) return msg;

  const So2ConvPack pack{x.data_ptr<float>(), src.data_ptr<long>(),
                         rot.data_ptr<float>(), rot_t.data_ptr<float>(),
                         kc.data_ptr<float>(), alpha.data_ptr<float>(),
                         msg.data_ptr<float>()};
  so2_conv_kernel<<<blocks_for(edges), kBlockThreads, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      pack, edges, dim, red, layers, channels);
  launch_check();
  return msg;
}
