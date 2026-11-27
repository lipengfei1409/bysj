// Ordinary (non-persistent) entry point for the SO(2) convolution.
//
// The same per-edge span also exists as a megakernel operator
// (ops/so2_conv_op.h, run under the persistent VM in dmk/). This file is the
// second way to reach it: one launch, no pages, no semaphores, the shape of
// every other kernel in csrc/. Both call ops/so2_conv_math.h, so the two paths
// cannot drift, and tests/csrc/accuracy/test_so2_conv_accuracy.py drives
// this entry point against reference/so2_conv.py.
//
//   msg[e, d, c] = alpha[e] * sum_r R^T[e][d][r] * xl[r]
//   xl           = stack of layers: xl += silu(K_l xl) * sigmoid(G_l xl),
//                  entered from xl = R[e] @ x[src[e]][:, c]
//
// The kernel's thread mapping (one wave per edge, lane = channel) lives in the
// op header next to the persistent form, so the two modes cannot disagree about
// it.  This file selects ordinary mode and supplies the torch side.
//
// The destination reduction is not here either: `msg` is per edge, and summing
// it onto atoms is SegmentSumOp's job (see docs/so2-conv-dmk.md).

#include "ops.h"
#include "checks.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>

// The operator is in ops/so2_conv_op.h, built here in ordinary mode: the same
// header that provides the persistent megakernel form provides this kernel, and
// both call ops/so2_conv_math.h.  What is left below is the torch-facing part --
// validation, allocation, the launch and the stream.
#define OPS_MODE OPS_MODE_ORDINARY
#include "ops/so2_conv_op.h"

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
  TORCH_CHECK(rot.size(0) == edges && rot_t.size(0) == edges && kc.size(0) == edges &&
                  alpha.size(0) == edges,
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

  const ops::so2_conv::So2ConvTensorPack pack{
      x.data_ptr<float>(), src.data_ptr<long>(),
      rot.data_ptr<float>(), rot_t.data_ptr<float>(), kc.data_ptr<float>(),
      alpha.data_ptr<float>(), msg.data_ptr<float>()};
  so2_conv_span_kernel<<<ops::so2_conv::so2_conv_span_blocks(edges),
                         ops::so2_conv::kBlockThreads, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      pack, edges, dim, red, layers, channels);
  launch_check();
  return msg;
}
