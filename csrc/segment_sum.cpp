// Destination-segmented reduction with fused zero-init and fused scaling.
//
// The traced baseline materializes ``out = 0`` (a separate ``fill_`` kernel),
// then scatters with row atomics (``index_add_`` / ``indexFunc``), then applies
// the per-node scale in a third kernel:
//
//   fill_(out, 0)                      // 0.3-1.0 ms per call, 235 calls
//   index_add_(out, dst, src)          // serializes on colliding edges
//   mul_(out, inverse_degree)          // one more full read+write pass
//
// Here one block owns one destination segment of a CSR edge list and walks the
// segment once, so the accumulator starts in a register, no atomic is issued
// and the normalization is folded into the store.
//
// Layout contract: ``dst`` is non-decreasing (the edge list is grouped by
// destination) and ``offsets`` is its exclusive prefix sum. The forward kernel
// only needs ``offsets``; the backward needs ``dst`` to map an edge back to its
// segment.

#include "ops.h"
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace {
constexpr unsigned int kMaxChannelThreads = 256;
constexpr unsigned int kMinChannelThreads = 32;

void check_offsets(const Tensor& offsets, const char* name) {
  TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt64 &&
                  offsets.is_contiguous(),
              name, " must be a contiguous DCU int64 tensor");
  TORCH_CHECK(offsets.dim() == 1 && offsets.size(0) >= 1,
              name, " must be a one-dimensional exclusive prefix sum");
}

void check_edges(const Tensor& dst, int64_t edges, const char* name) {
  TORCH_CHECK(dst.is_cuda() && dst.scalar_type() == torch::kInt64 &&
                  dst.is_contiguous(),
              name, " must be a contiguous DCU int64 tensor");
  TORCH_CHECK(dst.dim() == 1 && dst.size(0) == edges,
              name, " must have shape (", edges, ",)");
}

// One thread per channel keeps the gather over ``src`` coalesced; the block is
// rounded to a wave so no lane sits idle inside a wavefront.
unsigned int channel_threads(int64_t channels) {
  int64_t threads = (channels + 63) / 64 * 64;
  if (threads > kMaxChannelThreads) threads = kMaxChannelThreads;
  if (threads < kMinChannelThreads) threads = kMinChannelThreads;
  return static_cast<unsigned int>(threads);
}

unsigned int grid_for(int64_t blocks, const char* name) {
  TORCH_CHECK(blocks <= static_cast<int64_t>(
                  std::numeric_limits<unsigned int>::max()),
              name, " exceeds the HIP grid limit");
  return static_cast<unsigned int>(blocks);
}

template <bool kHasEdgeScale>
__global__ void segment_sum_forward_kernel(
    const float* __restrict__ src, const float* __restrict__ edge_scale,
    const int64_t* __restrict__ offsets, const float* __restrict__ node_scale,
    float* __restrict__ out, int64_t channels) {
  const int64_t segment = blockIdx.x;
  const int64_t begin = offsets[segment];
  const int64_t end = offsets[segment + 1];
  const float scale = node_scale == nullptr ? 1.0f : node_scale[segment];
  for (int64_t channel = threadIdx.x; channel < channels;
       channel += blockDim.x) {
    float accumulator = 0.0f;
    for (int64_t edge = begin; edge < end; ++edge) {
      const int64_t index = edge * channels + channel;
      accumulator += kHasEdgeScale ? src[index] * edge_scale[edge]
                                   : src[index];
    }
    out[segment * channels + channel] = accumulator * scale;
  }
}

// 16-byte loads for the common case where the channel count is a multiple of
// four. ``channels % 4 == 0`` also keeps every edge row 16-byte aligned.
template <bool kHasEdgeScale>
__global__ void segment_sum_forward_vec4_kernel(
    const float* __restrict__ src, const float* __restrict__ edge_scale,
    const int64_t* __restrict__ offsets, const float* __restrict__ node_scale,
    float* __restrict__ out, int64_t channels) {
  const int64_t segment = blockIdx.x;
  const int64_t begin = offsets[segment];
  const int64_t end = offsets[segment + 1];
  const float scale = node_scale == nullptr ? 1.0f : node_scale[segment];
  const int64_t groups = channels / 4;
  for (int64_t group = threadIdx.x; group < groups; group += blockDim.x) {
    float4 accumulator = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int64_t edge = begin; edge < end; ++edge) {
      const float4 value =
          reinterpret_cast<const float4*>(src + edge * channels)[group];
      const float weight = kHasEdgeScale ? edge_scale[edge] : 1.0f;
      accumulator.x += value.x * weight;
      accumulator.y += value.y * weight;
      accumulator.z += value.z * weight;
      accumulator.w += value.w * weight;
    }
    float4 scaled = make_float4(accumulator.x * scale, accumulator.y * scale,
                                accumulator.z * scale, accumulator.w * scale);
    reinterpret_cast<float4*>(out + segment * channels)[group] = scaled;
  }
}

// ``grad_src[e, c] = grad_out[dst[e], c] * edge_scale[e] * node_scale[dst[e]]``
// is the exact backward of the forward above; one block owns one edge so the
// destination scalar is loaded once instead of once per channel.
__global__ void segment_sum_grad_src_kernel(
    const float* __restrict__ grad_out, const float* __restrict__ edge_scale,
    const float* __restrict__ node_scale, const int64_t* __restrict__ dst,
    float* __restrict__ grad_src, int64_t channels) {
  const int64_t edge = blockIdx.x;
  const int64_t segment = dst[edge];
  float scale = edge_scale == nullptr ? 1.0f : edge_scale[edge];
  if (node_scale != nullptr) scale *= node_scale[segment];
  const float* source = grad_out + segment * channels;
  float* target = grad_src + edge * channels;
  for (int64_t channel = threadIdx.x; channel < channels;
       channel += blockDim.x)
    target[channel] = source[channel] * scale;
}

// ``d out[n, c] / d node_scale[n] = sum_{e in n} edge_scale[e] * src[e, c]``,
// so the cotangent is ``sum_c grad_out[n, c] * prefix[n, c]``. The prefix is
// recomputed rather than saved, which keeps the forward allocation-free.
__global__ void segment_sum_grad_node_scale_kernel(
    const float* __restrict__ grad_out, const float* __restrict__ src,
    const float* __restrict__ edge_scale, const int64_t* __restrict__ offsets,
    float* __restrict__ grad_scale, int64_t channels) {
  __shared__ float partial[kMaxChannelThreads];
  const int64_t segment = blockIdx.x;
  const int64_t begin = offsets[segment];
  const int64_t end = offsets[segment + 1];
  const float* upstream = grad_out + segment * channels;
  float local = 0.0f;
  for (int64_t channel = threadIdx.x; channel < channels;
       channel += blockDim.x) {
    float prefix = 0.0f;
    for (int64_t edge = begin; edge < end; ++edge) {
      const int64_t index = edge * channels + channel;
      prefix += edge_scale == nullptr ? src[index]
                                      : src[index] * edge_scale[edge];
    }
    local += upstream[channel] * prefix;
  }
  partial[threadIdx.x] = local;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) grad_scale[segment] = partial[0];
}

void check_segment_sum_inputs(const Tensor& src, const Tensor& offsets,
                              const c10::optional<Tensor>& edge_scale,
                              const c10::optional<Tensor>& node_scale) {
  f32(src);
  TORCH_CHECK(src.dim() == 2, "segment sum src must have shape (n_edges, channels)");
  const int64_t edges = src.size(0);
  const int64_t channels = src.size(1);
  TORCH_CHECK(channels > 0, "segment sum requires at least one channel");
  check_offsets(offsets, "segment sum offsets");
  const int64_t nodes = offsets.size(0) - 1;
  if (edge_scale.has_value()) {
    f32(*edge_scale);
    TORCH_CHECK(edge_scale->dim() == 1 && edge_scale->size(0) == edges,
                "segment sum edge_scale must have shape (n_edges,)");
    same_device(src, {*edge_scale});
  }
  if (node_scale.has_value()) {
    f32(*node_scale);
    TORCH_CHECK(node_scale->numel() == nodes,
                "segment sum node_scale must have one value per segment");
    same_device(src, {*node_scale});
  }
  same_device(src, {offsets});
}
}  // namespace

Tensor segment_sum_forward_hip(Tensor src, Tensor offsets,
                               c10::optional<Tensor> edge_scale,
                               c10::optional<Tensor> node_scale) {
  check_segment_sum_inputs(src, offsets, edge_scale, node_scale);
  const int64_t edges = src.size(0);
  const int64_t channels = src.size(1);
  const int64_t nodes = offsets.size(0) - 1;
  // Any in-range CSR segment reads give a valid result, so an empty edge list
  // still has to produce a zeroed output; the kernel writes every element.
  Tensor out = torch::empty({nodes, channels}, src.options());
  if (nodes == 0) return out;
  if (edges == 0) {
    out.zero_();
    return out;
  }

  const float* edge_scale_ptr =
      edge_scale.has_value() ? edge_scale->data_ptr<float>() : nullptr;
  const float* node_scale_ptr =
      node_scale.has_value() ? node_scale->data_ptr<float>() : nullptr;
  const dim3 grid(grid_for(nodes, "segment sum segment count"));
  const dim3 block(channel_threads(channels));
  c10::cuda::CUDAGuard guard(src.device());

  const bool vectorized = channels % 4 == 0;
  if (vectorized) {
    if (edge_scale_ptr == nullptr) {
      hipLaunchKernelGGL(segment_sum_forward_vec4_kernel<false>, grid, block, 0,
                         at::cuda::getCurrentCUDAStream(),
                         src.data_ptr<float>(), nullptr,
                         offsets.data_ptr<int64_t>(), node_scale_ptr,
                         out.data_ptr<float>(), channels);
    } else {
      hipLaunchKernelGGL(segment_sum_forward_vec4_kernel<true>, grid, block, 0,
                         at::cuda::getCurrentCUDAStream(),
                         src.data_ptr<float>(), edge_scale_ptr,
                         offsets.data_ptr<int64_t>(), node_scale_ptr,
                         out.data_ptr<float>(), channels);
    }
  } else if (edge_scale_ptr == nullptr) {
    hipLaunchKernelGGL(segment_sum_forward_kernel<false>, grid, block, 0,
                       at::cuda::getCurrentCUDAStream(), src.data_ptr<float>(),
                       nullptr, offsets.data_ptr<int64_t>(), node_scale_ptr,
                       out.data_ptr<float>(), channels);
  } else {
    hipLaunchKernelGGL(segment_sum_forward_kernel<true>, grid, block, 0,
                       at::cuda::getCurrentCUDAStream(), src.data_ptr<float>(),
                       edge_scale_ptr, offsets.data_ptr<int64_t>(),
                       node_scale_ptr, out.data_ptr<float>(), channels);
  }
  launch_check();
  return out;
}

std::tuple<Tensor, Tensor> segment_sum_backward_hip(
    Tensor grad_out, Tensor src, Tensor dst, Tensor offsets,
    c10::optional<Tensor> edge_scale, c10::optional<Tensor> node_scale,
    bool want_node_scale_grad) {
  check_segment_sum_inputs(src, offsets, edge_scale, node_scale);
  f32(grad_out);
  const int64_t edges = src.size(0);
  const int64_t channels = src.size(1);
  const int64_t nodes = offsets.size(0) - 1;
  TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(0) == nodes &&
                  grad_out.size(1) == channels,
              "segment sum grad_out must have shape (n_nodes, channels)");
  check_edges(dst, edges, "segment sum dst");
  same_device(src, {grad_out, dst});

  Tensor grad_src = torch::empty_like(src);
  if (edges == 0) return std::make_tuple(grad_src, Tensor());

  const float* edge_scale_ptr =
      edge_scale.has_value() ? edge_scale->data_ptr<float>() : nullptr;
  const float* node_scale_ptr =
      node_scale.has_value() ? node_scale->data_ptr<float>() : nullptr;
  c10::cuda::CUDAGuard guard(src.device());
  const dim3 block(channel_threads(channels));

  hipLaunchKernelGGL(segment_sum_grad_src_kernel,
                     dim3(grid_for(edges, "segment sum edge count")), block, 0,
                     at::cuda::getCurrentCUDAStream(), grad_out.data_ptr<float>(),
                     edge_scale_ptr, node_scale_ptr, dst.data_ptr<int64_t>(),
                     grad_src.data_ptr<float>(), channels);
  launch_check();

  if (!want_node_scale_grad || node_scale_ptr == nullptr)
    return std::make_tuple(grad_src, Tensor());

  Tensor grad_node_scale = torch::empty_like(*node_scale);
  hipLaunchKernelGGL(segment_sum_grad_node_scale_kernel,
                     dim3(grid_for(nodes, "segment sum segment count")), block,
                     0, at::cuda::getCurrentCUDAStream(),
                     grad_out.data_ptr<float>(), src.data_ptr<float>(),
                     edge_scale_ptr, offsets.data_ptr<int64_t>(),
                     grad_node_scale.data_ptr<float>(), channels);
  launch_check();
  return std::make_tuple(grad_src, grad_node_scale);
}
