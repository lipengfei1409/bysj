// C7-b: the block-diagonal mixing GEMM, one launch for the whole block set.
//
// ``SO2Linear`` contracts an activation ``x`` of shape ``(E, ROW)`` with a
// weight that is block diagonal over the ``|m|`` groups, so the dense reference
// slices the diagonal blocks and issues one ``torch.bmm`` per block, then
// concatenates. Three costs follow from that shape:
//
//   * the weight is presented as a strided view, so every block pays a
//     contiguous copy of its slice;
//   * each ``bmm`` pads the block width to a power of two, and the 192-wide
//     block wastes about a quarter of its work on the padding;
//   * the concatenation materialises a fresh output tensor.
//
// The operator takes the blocks as they are stored -- one contiguous
// ``(size_b, size_b)`` weight per diagonal block plus the column offset it acts
// on -- and writes the output once:
//
//   out[e, off_b + o] = sum_k W_b[o, k] * x[e, off_b + k]
//
// The reverse direction only needs the activation gradient (inference freezes
// the weights):  gx[e, off_b + k] = sum_o W_b[o, k] * gout[e, off_b + o].
//
// One thread per (edge, output row). The block weights are small enough to be
// L2 resident and are re-read by every edge on that block, which is exactly the
// reuse the dense slice-and-bmm form gives up.

#include "ops.h"
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <tuple>
#include <vector>

namespace {
constexpr unsigned int kThreads = 256;

__device__ __forceinline__ void contract(const float* __restrict__ src,
                                         const float* __restrict__ w,
                                         float* __restrict__ dst,
                                         int extent, int lane_extent) {
  // One output row: `extent` multiply-adds, four independent accumulators so the
  // dependent chain does not bound the issue rate.
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (int k = 0; k + 3 < extent; k += 4) {
    acc0 = fmaf(w[k + 0], src[k + 0], acc0);
    acc1 = fmaf(w[k + 1], src[k + 1], acc1);
    acc2 = fmaf(w[k + 2], src[k + 2], acc2);
    acc3 = fmaf(w[k + 3], src[k + 3], acc3);
  }
  for (int k = extent & ~3; k < extent; ++k) {
    acc0 = fmaf(w[k], src[k], acc0);
  }
  (void)lane_extent;
  dst[0] = (acc0 + acc1) + (acc2 + acc3);
}

__global__ void forward_kernel(const float* __restrict__ x,
                               const float* const* weights,
                               const long* __restrict__ offsets,
                               const int* __restrict__ sizes,
                               float* __restrict__ out,
                               long edges, int blocks, int total_rows) {
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= edges * total_rows) {
    return;
  }
  const long edge = idx / total_rows;
  const int row = static_cast<int>(idx - edge * total_rows);
  int b = 0;
  while (b + 1 < blocks && offsets[b + 1] <= row) {
    ++b;
  }
  const int extent = sizes[b];
  const int offset = static_cast<int>(offsets[b]);
  const int local = row - offset;
  if (local >= extent) {
    return;  // padding row: the block does not cover it
  }
  contract(x + edge * total_rows + offset,
           weights[b] + static_cast<long>(local) * extent,
           out + edge * total_rows + row, extent, total_rows);
}

__global__ void backward_kernel(const float* __restrict__ grad_out,
                                const float* const* weights,
                                const long* __restrict__ offsets,
                                const int* __restrict__ sizes,
                                float* __restrict__ grad_x,
                                long edges, int blocks, int total_rows) {
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= edges * total_rows) {
    return;
  }
  const long edge = idx / total_rows;
  const int row = static_cast<int>(idx - edge * total_rows);
  int b = 0;
  while (b + 1 < blocks && offsets[b + 1] <= row) {
    ++b;
  }
  const int extent = sizes[b];
  const int offset = static_cast<int>(offsets[b]);
  const int local = row - offset;
  if (local >= extent) {
    return;
  }
  // gx = W^T gout, i.e. the same contraction with the weight transposed.
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  const float* go = grad_out + edge * total_rows + offset;
  const float* w = weights[b];
  for (int o = 0; o + 3 < extent; o += 4) {
    acc0 = fmaf(w[static_cast<long>(o + 0) * extent + local], go[o + 0], acc0);
    acc1 = fmaf(w[static_cast<long>(o + 1) * extent + local], go[o + 1], acc1);
    acc2 = fmaf(w[static_cast<long>(o + 2) * extent + local], go[o + 2], acc2);
    acc3 = fmaf(w[static_cast<long>(o + 3) * extent + local], go[o + 3], acc3);
  }
  for (int o = extent & ~3; o < extent; ++o) {
    acc0 = fmaf(w[static_cast<long>(o) * extent + local], go[o], acc0);
  }
  grad_x[edge * total_rows + row] = (acc0 + acc1) + (acc2 + acc3);
}
}  // namespace

namespace {
struct BlockPlan {
  std::vector<const float*> pointers;
  std::vector<long> offsets;
  std::vector<int> sizes;
  Tensor pointers_t;
  Tensor offsets_t;
  Tensor sizes_t;
  int total_rows;
};

BlockPlan plan(Tensor& weights, const Tensor& offsets, const Tensor& x) {
  TORCH_CHECK(weights.dim() == 3 && weights.size(1) == weights.size(2),
              "each diagonal block must be (size, size)");
  TORCH_CHECK(offsets.scalar_type() == torch::kLong && offsets.dim() == 1,
              "block offsets must be an int64 (blocks + 1) vector");
  TORCH_CHECK(offsets.numel() == weights.size(0) + 1,
              "offsets must hold one more entry than there are blocks");
  const long last_offset = offsets[offsets.numel() - 1].item<long>();
  TORCH_CHECK(x.dim() == 2 && x.size(1) == last_offset,
              "the activation width must match the last offset");
  f32(weights);
  const auto count = weights.size(0);
  BlockPlan p;
  // `.item()` per block is a sync; there are two blocks in the model, so the
  // cost is irrelevant and the alternative (a device-side scan) buys nothing.
  p.total_rows = static_cast<int>(last_offset);
  p.pointers.resize(count);
  p.offsets.resize(count + 1);
  p.sizes.resize(count);
  for (int64_t b = 0; b < count; ++b) {
    p.pointers[b] = weights.data_ptr<float>() + b * weights.size(1) * weights.size(2);
    p.offsets[b] = offsets[b].item<long>();
    p.sizes[b] = static_cast<int>(weights.size(1));
    TORCH_CHECK(weights.size(1) == offsets[b + 1].item<long>() - offsets[b].item<long>(),
                "block ", b, " width does not match its offsets");
  }
  p.offsets[count] = p.total_rows;
  auto device = weights.device();
  p.pointers_t = torch::from_blob(p.pointers.data(), {count},
                                  torch::TensorOptions().dtype(torch::kLong))
                     .to(device);
  p.offsets_t = torch::from_blob(p.offsets.data(), {count + 1},
                                 torch::TensorOptions().dtype(torch::kLong))
                    .to(device);
  p.sizes_t = torch::from_blob(p.sizes.data(), {count},
                               torch::TensorOptions().dtype(torch::kInt))
                  .to(device);
  return p;
}
}  // namespace

Tensor blockdiag_gemm_hip(Tensor x, Tensor weights, Tensor offsets) {
  f32(x);
  same_device(x, {weights, offsets});
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto p = plan(weights, offsets, x);
  auto out = torch::zeros_like(x);
  const long total = x.size(0) * p.total_rows;
  if (total > 0) {
    const unsigned blocks = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    forward_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        reinterpret_cast<const float* const*>(p.pointers_t.data_ptr<long>()),
        p.offsets_t.data_ptr<long>(), p.sizes_t.data_ptr<int>(),
        out.data_ptr<float>(), x.size(0), static_cast<int>(p.sizes.size()),
        p.total_rows);
    launch_check();
  }
  return out;
}

Tensor blockdiag_gemm_backward_hip(Tensor grad_out, Tensor weights, Tensor offsets) {
  f32(grad_out);
  same_device(grad_out, {weights, offsets});
  const at::cuda::OptionalCUDAGuard guard(grad_out.device());
  auto p = plan(weights, offsets, grad_out);
  auto grad_x = torch::zeros_like(grad_out);
  const long total = grad_out.size(0) * p.total_rows;
  if (total > 0) {
    const unsigned blocks = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    backward_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_out.data_ptr<float>(),
        reinterpret_cast<const float* const*>(p.pointers_t.data_ptr<long>()),
        p.offsets_t.data_ptr<long>(), p.sizes_t.data_ptr<int>(),
        grad_x.data_ptr<float>(), grad_out.size(0),
        static_cast<int>(p.sizes.size()), p.total_rows);
    launch_check();
  }
  return grad_x;
}
