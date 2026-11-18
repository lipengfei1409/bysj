// Destination-segmented softmax with the cutoff envelope folded into the logit.
//
// The traced baseline expresses one normalization as a chain of ATen kernels
// per convolution block -- a ``fill_`` of the per-channel null mass, several
// ``index_add_`` / ``scatter_reduce`` passes for the segment maximum and the
// segment sum, then the ``exp``/``sub``/``div`` surfaces:
//
//   fill_(null_mass, 0)                       // 235 fills / 75 ms overall
//   index_add_(per_channel_max, dst, logits)
//   index_add_(null_mass, dst, env_gate)
//   ...
//
// This operator owns one destination segment per block, keeps the maximum and
// the denominator in registers and writes the normalized weights once.
//
// Mathematics (per destination segment, per channel). With the effective logit
// ``l_e = logit_e + 2 ln env_e`` for edges with ``env_e > 0`` and the null logit
// ``l_0`` contributed by the segment itself,
//
//   m = max(l_0, max_e l_e),  w_e = exp(l_e - m),  w_0 = exp(l_0 - m)
//   D = w_0 + sum_e w_e,      alpha_e = w_e / D,   alpha_0 = w_0 / D
//
// Edges with a non-positive envelope carry no mass and get ``alpha_e = 0``.
// The first-order backward of the same normalization, given the output
// cotangents, is
//
//   S = sum_e alpha_e grad_alpha_e + alpha_0 grad_alpha_0
//   grad_logit_e = alpha_e (grad_alpha_e - S)
//   grad_logit_0 = alpha_0 (grad_alpha_0 - S)
//   grad_env_e   = (sum_c grad_logit_e) * 2 / env_e
//
// which needs no second read of the logits: it runs on the stored weights.

#include "ops.h"
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

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

// ``l_e = logit_e + 2 ln env_e``; a non-positive envelope excludes the edge.
__device__ inline bool effective_logit(float logit_value, float envelope,
                                       float* effective) {
  if (!(envelope > 0.0f)) return false;
  *effective = logit_value + 2.0f * logf(envelope);
  return true;
}

template <bool kHasNull>
__global__ void segment_softmax_forward_kernel(
    const float* __restrict__ logit, const float* __restrict__ env,
    const int64_t* __restrict__ offsets, const int64_t* __restrict__ order,
    const float* __restrict__ null_logit, float* __restrict__ alpha,
    float* __restrict__ null_alpha, int64_t channels) {
  const int64_t segment = blockIdx.x;
  const int64_t begin = offsets[segment];
  const int64_t end = offsets[segment + 1];
  for (int64_t channel = threadIdx.x; channel < channels;
       channel += blockDim.x) {
    float peak = -INFINITY;
    for (int64_t slot = begin; slot < end; ++slot) {
      const int64_t edge = order == nullptr ? slot : order[slot];
      float effective = 0.0f;
      if (!effective_logit(logit[edge * channels + channel], env[edge],
                           &effective))
        continue;
      peak = fmaxf(peak, effective);
    }
    const float null_value =
        kHasNull ? null_logit[segment * channels + channel] : -INFINITY;
    peak = fmaxf(peak, null_value);

    if (peak == -INFINITY) {
      for (int64_t slot = begin; slot < end; ++slot) {
        const int64_t edge = order == nullptr ? slot : order[slot];
        alpha[edge * channels + channel] = 0.0f;
      }
      if (kHasNull) null_alpha[segment * channels + channel] = 0.0f;
      continue;
    }

    float total = 0.0f;
    for (int64_t slot = begin; slot < end; ++slot) {
      const int64_t edge = order == nullptr ? slot : order[slot];
      float effective = 0.0f;
      if (!effective_logit(logit[edge * channels + channel], env[edge],
                           &effective))
        continue;
      total += expf(effective - peak);
    }
    const float null_weight = kHasNull ? expf(null_value - peak) : 0.0f;
    total += null_weight;
    const float inverse = 1.0f / total;

    for (int64_t slot = begin; slot < end; ++slot) {
      const int64_t edge = order == nullptr ? slot : order[slot];
      float effective = 0.0f;
      if (!effective_logit(logit[edge * channels + channel], env[edge],
                           &effective)) {
        alpha[edge * channels + channel] = 0.0f;
        continue;
      }
      alpha[edge * channels + channel] = expf(effective - peak) * inverse;
    }
    if (kHasNull)
      null_alpha[segment * channels + channel] = null_weight * inverse;
  }
}

template <bool kHasNull>
__global__ void segment_softmax_backward_logit_kernel(
    const float* __restrict__ alpha, const float* __restrict__ null_alpha,
    const float* __restrict__ grad_alpha,
    const float* __restrict__ grad_null_alpha,
    const int64_t* __restrict__ offsets, const int64_t* __restrict__ order,
    float* __restrict__ grad_logit, float* __restrict__ grad_null_logit,
    int64_t channels) {
  const int64_t segment = blockIdx.x;
  const int64_t begin = offsets[segment];
  const int64_t end = offsets[segment + 1];
  for (int64_t channel = threadIdx.x; channel < channels;
       channel += blockDim.x) {
    float expectation = 0.0f;
    for (int64_t slot = begin; slot < end; ++slot) {
      const int64_t edge = order == nullptr ? slot : order[slot];
      const int64_t index = edge * channels + channel;
      expectation += alpha[index] * grad_alpha[index];
    }
    if (kHasNull)
      expectation += null_alpha[segment * channels + channel] *
                     grad_null_alpha[segment * channels + channel];
    for (int64_t slot = begin; slot < end; ++slot) {
      const int64_t edge = order == nullptr ? slot : order[slot];
      const int64_t index = edge * channels + channel;
      grad_logit[index] = alpha[index] * (grad_alpha[index] - expectation);
    }
    if (kHasNull)
      grad_null_logit[segment * channels + channel] =
          null_alpha[segment * channels + channel] *
          (grad_null_alpha[segment * channels + channel] - expectation);
  }
}

// One block per edge folds the channel sum, so ``grad_env`` never needs a
// materialized ``(E, C)`` product.
__global__ void segment_softmax_backward_env_kernel(
    const float* __restrict__ grad_logit, const float* __restrict__ env,
    float* __restrict__ grad_env, int64_t channels) {
  __shared__ float partial[kMaxChannelThreads];
  const int64_t edge = blockIdx.x;
  float local = 0.0f;
  for (int64_t channel = threadIdx.x; channel < channels;
       channel += blockDim.x)
    local += grad_logit[edge * channels + channel];
  partial[threadIdx.x] = local;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float envelope = env[edge];
    grad_env[edge] = envelope > 0.0f ? partial[0] * 2.0f / envelope : 0.0f;
  }
}

void check_order(const c10::optional<Tensor>& order, int64_t edges) {
  if (!order.has_value()) return;
  const Tensor& permutation = *order;
  TORCH_CHECK(permutation.is_cuda() &&
                  permutation.scalar_type() == torch::kInt64 &&
                  permutation.is_contiguous(),
              "segment softmax order must be a contiguous DCU int64 tensor");
  TORCH_CHECK(permutation.dim() == 1 && permutation.size(0) == edges,
              "segment softmax order must have shape (n_edges,)");
}

void check_softmax_inputs(const Tensor& logit, const Tensor& env,
                          const Tensor& offsets,
                          const c10::optional<Tensor>& order) {
  f32(logit);
  f32(env);
  TORCH_CHECK(logit.dim() == 2,
              "segment softmax logit must have shape (n_edges, channels)");
  const int64_t edges = logit.size(0);
  const int64_t channels = logit.size(1);
  TORCH_CHECK(channels > 0, "segment softmax requires at least one channel");
  TORCH_CHECK(env.dim() == 1 && env.size(0) == edges,
              "segment softmax env must have shape (n_edges,)");
  check_offsets(offsets, "segment softmax offsets");
  check_order(order, edges);
  if (order.has_value()) same_device(logit, {*order});
  same_device(logit, {env, offsets});
}

void check_null(const c10::optional<Tensor>& null_tensor, int64_t nodes,
                int64_t channels, const char* name) {
  if (!null_tensor.has_value()) return;
  f32(*null_tensor);
  TORCH_CHECK(null_tensor->numel() == nodes * channels,
              name, " must have shape (n_nodes, channels)");
}
}  // namespace

std::tuple<Tensor, Tensor> segment_softmax_forward_hip(
    Tensor logit, Tensor env, Tensor offsets,
    c10::optional<Tensor> null_logit, c10::optional<Tensor> order) {
  check_softmax_inputs(logit, env, offsets, order);
  const int64_t edges = logit.size(0);
  const int64_t channels = logit.size(1);
  const int64_t nodes = offsets.size(0) - 1;
  check_null(null_logit, nodes, channels, "segment softmax null_logit");
  if (null_logit.has_value()) same_device(logit, {*null_logit});

  Tensor alpha = torch::empty({edges, channels}, logit.options());
  Tensor null_alpha;
  if (null_logit.has_value())
    null_alpha = torch::empty({nodes, channels}, logit.options());
  if (nodes == 0) return std::make_tuple(alpha, null_alpha);
  if (edges == 0) {
    if (null_alpha.defined()) null_alpha.zero_();
    return std::make_tuple(alpha, null_alpha);
  }

  c10::cuda::CUDAGuard guard(logit.device());
  const dim3 grid(grid_for(nodes, "segment softmax segment count"));
  const dim3 block(channel_threads(channels));
  const int64_t* order_ptr =
      order.has_value() ? order->data_ptr<int64_t>() : nullptr;
  const float* null_ptr =
      null_logit.has_value() ? null_logit->data_ptr<float>() : nullptr;
  if (null_ptr == nullptr) {
    hipLaunchKernelGGL(segment_softmax_forward_kernel<false>, grid, block, 0,
                       at::cuda::getCurrentCUDAStream(), logit.data_ptr<float>(),
                       env.data_ptr<float>(), offsets.data_ptr<int64_t>(),
                       order_ptr, nullptr, alpha.data_ptr<float>(), nullptr,
                       channels);
  } else {
    hipLaunchKernelGGL(segment_softmax_forward_kernel<true>, grid, block, 0,
                       at::cuda::getCurrentCUDAStream(), logit.data_ptr<float>(),
                       env.data_ptr<float>(), offsets.data_ptr<int64_t>(),
                       order_ptr, null_ptr, alpha.data_ptr<float>(),
                       null_alpha.data_ptr<float>(), channels);
  }
  launch_check();
  return std::make_tuple(alpha, null_alpha);
}

std::tuple<Tensor, Tensor, Tensor> segment_softmax_backward_hip(
    Tensor alpha, Tensor grad_alpha, Tensor env, Tensor offsets,
    c10::optional<Tensor> null_alpha,
    c10::optional<Tensor> grad_null_alpha, c10::optional<Tensor> order) {
  check_softmax_inputs(alpha, env, offsets, order);
  f32(grad_alpha);
  const int64_t edges = alpha.size(0);
  const int64_t channels = alpha.size(1);
  const int64_t nodes = offsets.size(0) - 1;
  TORCH_CHECK(grad_alpha.sizes() == alpha.sizes(),
              "segment softmax grad_alpha must match alpha");
  same_device(alpha, {grad_alpha});
  const bool has_null = null_alpha.has_value();
  TORCH_CHECK(has_null == grad_null_alpha.has_value(),
              "segment softmax requires null_alpha and grad_null_alpha together");
  if (has_null) {
    check_null(null_alpha, nodes, channels, "segment softmax null_alpha");
    check_null(grad_null_alpha, nodes, channels,
               "segment softmax grad_null_alpha");
    same_device(alpha, {*null_alpha, *grad_null_alpha});
  }

  Tensor grad_logit = torch::empty_like(alpha);
  Tensor grad_null_logit;
  if (has_null) grad_null_logit = torch::empty({nodes, channels}, alpha.options());
  Tensor grad_env = torch::empty({edges}, alpha.options());
  if (nodes == 0)
    return std::make_tuple(grad_logit, grad_null_logit, grad_env);
  if (edges == 0) {
    if (grad_null_logit.defined()) grad_null_logit.zero_();
    return std::make_tuple(grad_logit, grad_null_logit, grad_env);
  }

  const int64_t* order_ptr =
      order.has_value() ? order->data_ptr<int64_t>() : nullptr;
  c10::cuda::CUDAGuard guard(alpha.device());
  const dim3 grid(grid_for(nodes, "segment softmax segment count"));
  const dim3 block(channel_threads(channels));
  if (has_null) {
    hipLaunchKernelGGL(segment_softmax_backward_logit_kernel<true>, grid, block,
                       0, at::cuda::getCurrentCUDAStream(),
                       alpha.data_ptr<float>(),
                       null_alpha->data_ptr<float>(), grad_alpha.data_ptr<float>(),
                       grad_null_alpha->data_ptr<float>(),
                       offsets.data_ptr<int64_t>(), order_ptr,
                       grad_logit.data_ptr<float>(),
                       grad_null_logit.data_ptr<float>(), channels);
  } else {
    hipLaunchKernelGGL(segment_softmax_backward_logit_kernel<false>, grid, block,
                       0, at::cuda::getCurrentCUDAStream(),
                       alpha.data_ptr<float>(), nullptr,
                       grad_alpha.data_ptr<float>(), nullptr,
                       offsets.data_ptr<int64_t>(), order_ptr,
                       grad_logit.data_ptr<float>(), nullptr, channels);
  }
  launch_check();

  hipLaunchKernelGGL(segment_softmax_backward_env_kernel,
                     dim3(grid_for(edges, "segment softmax edge count")), block,
                     0, at::cuda::getCurrentCUDAStream(),
                     grad_logit.data_ptr<float>(), env.data_ptr<float>(),
                     grad_env.data_ptr<float>(), channels);
  launch_check();
  return std::make_tuple(grad_logit, grad_null_logit, grad_env);
}
