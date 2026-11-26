// Ordinary (non-persistent) entry point for the SO(2) convolution.
//
// The same per-edge span also exists as a megakernel operator
// (ops/so2_conv_op.h, run under the persistent VM in dmk/). This file is the
// second way to reach it: one launch, no pages, no semaphores, the shape of
// every other kernel in csrc/. Both call ops/so2_conv_math.h, so the two paths
// cannot drift, and tests/csrc/accuracy/test_so2_conv_dmk_accuracy.py drives
// this entry point against reference/so2_conv_dmk.py.
//
//   msg[e, d, c] = alpha[e] * sum_r R^T[e][d][r] * xl[r]
//   xl           = stack of layers: xl += silu(K_l xl) * sigmoid(G_l xl),
//                  entered from xl = R[e] @ x[src[e]][:, c]
//
// Thread mapping: lane of a wave <-> channel (C <= 64). One wave handles one
// edge, so a 64-float row is a single coalesced 256 B access and the whole span
// stays in registers. A block carries kEdgeWaves edges in parallel; edges inside
// a block walk the range in strides of kEdgeWaves, which is the same tail
// behaviour the persistent op has.
//
// The destination reduction is not here either: `msg` is per edge, and summing
// it onto atoms is SegmentSumOp's job (see docs/so2-conv-dmk.md).

#include "ops.h"
#include "checks.h"
#include "ops/so2_conv_math.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

//: One wave per edge: 64 lanes == 64 channels at the profiled focus width.
constexpr int kEdgeWaves = 4;
constexpr int kBlockThreads = kEdgeWaves * 64;

__global__ void so2_conv_dmk_kernel(
    const float* __restrict__ x, const long* __restrict__ src,
    const float* __restrict__ rot, const float* __restrict__ rot_t,
    const float* __restrict__ kc, const float* __restrict__ alpha,
    float* __restrict__ msg, long edges, int dim, int red, int layers,
    int channels) {
    const int wave = static_cast<int>(threadIdx.x) / 64;
    const int lane = static_cast<int>(threadIdx.x) % 64;
    const int channel = lane;
    const bool active = channel < channels;
    const long stride = gridDim.x * static_cast<long>(kEdgeWaves);

    for (long edge = static_cast<long>(blockIdx.x) * kEdgeWaves + wave;
         edge < edges; edge += stride) {
        const long src_node = src[edge];
        const float* xrow = x + static_cast<size_t>(src_node) * dim * channels;
        float xv[dpa4::so2::kMaxDim];
        if (active) {
            for (int d = 0; d < dim; ++d)
                xv[d] = xrow[static_cast<size_t>(d) * channels + channel];
        }

        float xl[dpa4::so2::kMaxRed];
        float y[dpa4::so2::kMaxDim];
        float envelope = 0.0f;
        if (active) {
            const long rot_n = static_cast<long>(red) * dim;
            const long block = static_cast<long>(red) * red;
            const float* rot_e = rot + static_cast<size_t>(edge) * rot_n;
            const float* rot_t_e = rot_t + static_cast<size_t>(edge) * rot_n;
            const float* kc_e = kc + static_cast<size_t>(edge) * layers * 2 * block;
            envelope = alpha[edge];
            dpa4::so2::edge_span(xv, rot_e, rot_t_e, kc_e, dim, red, layers, y, xl);
        }

        // Every lane writes only its own channel; idle lanes (C < 64) write
        // nothing, which is what keeps a narrow focus from corrupting the next
        // edge's message.
        if (active) {
            float* out = msg + static_cast<size_t>(edge) * dim * channels;
            for (int d = 0; d < dim; ++d)
                out[static_cast<size_t>(d) * channels + channel] = envelope * y[d];
        }
    }
}

}  // namespace

Tensor so2_conv_dmk_hip(Tensor x, Tensor src, Tensor rot, Tensor rot_t,
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
  TORCH_CHECK(rot.size(0) == edges && rot_t.size(0) == edges && kc.size(0) == edges &&
                  alpha.size(0) == edges,
              "every per-edge tensor must have one row per edge");
  TORCH_CHECK(kc.size(2) == 2 && kc.size(3) == red && kc.size(4) == red,
              "kc must hold a content and a gate kernel of (red, red)");
  TORCH_CHECK(dim <= dpa4::so2::kMaxDim && red <= dpa4::so2::kMaxRed,
              "this kernel supports dim, red <= ", dpa4::so2::kMaxDim);
  TORCH_CHECK(channels <= 64,
              "one lane per channel requires channels <= 64, got ", channels);
  same_device(x, {rot, rot_t, kc, alpha});

  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto msg = torch::zeros({edges, dim, channels}, x.options());
  if (edges == 0) return msg;

  const long blocks_needed = (edges + kEdgeWaves - 1) / kEdgeWaves;
  const unsigned blocks = static_cast<unsigned>(blocks_needed);
  so2_conv_dmk_kernel<<<blocks, kBlockThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), src.data_ptr<long>(), rot.data_ptr<float>(),
      rot_t.data_ptr<float>(), kc.data_ptr<float>(), alpha.data_ptr<float>(),
      msg.data_ptr<float>(), edges, dim, red, layers, channels);
  launch_check();
  return msg;
}
