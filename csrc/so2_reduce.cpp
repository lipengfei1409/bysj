#include "ops.h"
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {
constexpr unsigned int kBlockSize = 256;

__device__ __forceinline__ float block_sum(float value, float* partial) {
  const unsigned int lane = threadIdx.x;
  partial[lane] = value;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) partial[lane] += partial[lane + stride];
    __syncthreads();
  }
  return partial[0];
}

void check_index_tensor(const Tensor& tensor, int64_t length, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kInt64 &&
                  tensor.is_contiguous(),
              name, " must be a contiguous DCU int64 tensor");
  TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == length,
              name, " must have shape (", length, ",)");
}

void check_so2_inputs(const Tensor& message, const Tensor& edge_weight,
                      const Tensor& dst, const Tensor& inverse) {
  f32(message);
  f32(edge_weight);
  f32(inverse);
  TORCH_CHECK(message.dim() == 3 && message.size(1) > 0 && message.size(2) > 0,
              "SO(2) message must have shape (n_edges, n_coeff, n_channels)");
  const int64_t n_edges = message.size(0);
  TORCH_CHECK(edge_weight.dim() == 1 && edge_weight.size(0) == n_edges,
              "SO(2) edge_weight must have shape (n_edges,)");
  check_index_tensor(dst, n_edges, "SO(2) dst");
  TORCH_CHECK(inverse.dim() == 1,
              "SO(2) inverse degree scale must be flattened to (n_nodes,)");
  TORCH_CHECK(inverse.numel() > 0 || n_edges == 0,
              "SO(2) edges require at least one destination node");
  same_device(message, {edge_weight, dst, inverse});
}

int64_t checked_product(int64_t left, int64_t right, const char* name) {
  TORCH_CHECK(left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
              name, " work size overflows int64");
  return left * right;
}

unsigned int grid_size(int64_t work_items, const char* name) {
  const int64_t blocks = work_items / kBlockSize +
      (work_items % kBlockSize != 0);
  TORCH_CHECK(blocks <= static_cast<int64_t>(
                  std::numeric_limits<unsigned int>::max()),
              name, " grid exceeds the HIP launch limit");
  return static_cast<unsigned int>(blocks);
}

__global__ void so2_reduce_scatter_kernel(
    const float* message, const float* edge_weight, const int64_t* dst,
    float* pre_norm, int64_t n_edges, int64_t n_coeff, int64_t n_channels,
    int64_t work_items) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t plane = n_coeff * n_channels;
  for (int64_t index = first; index < work_items; index += stride) {
    const int64_t channel_coeff = index % plane;
    const int64_t edge = index / plane;
    const int64_t node = dst[edge];
    atomicAdd(pre_norm + node * plane + channel_coeff,
              message[index] * edge_weight[edge]);
  }
}

__global__ void so2_reduce_scale_kernel(
    const float* pre_norm, const float* inverse, float* output,
    int64_t n_nodes, int64_t plane, int64_t work_items) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = first; index < work_items; index += stride) {
    const int64_t node = index / plane;
    output[index] = pre_norm[index] * inverse[node];
  }
}

__global__ void so2_reduce_edge_backward_kernel(
    const float* grad_output, const float* message, const float* edge_weight,
    const int64_t* dst, const float* inverse, float* grad_message,
    float* grad_weight, int64_t n_edges, int64_t plane) {
  __shared__ float partial[kBlockSize];
  const int64_t first_edge = static_cast<int64_t>(blockIdx.x);
  const int64_t edge_stride = static_cast<int64_t>(gridDim.x);
  for (int64_t edge = first_edge; edge < n_edges; edge += edge_stride) {
    const int64_t node = dst[edge];
    const float inv = inverse[node];
    float dot = 0.0f;
    for (int64_t inner = threadIdx.x; inner < plane; inner += blockDim.x) {
      const int64_t edge_index = edge * plane + inner;
      const int64_t node_index = node * plane + inner;
      const float upstream = grad_output[node_index];
      grad_message[edge_index] = upstream * edge_weight[edge] * inv;
      dot += upstream * message[edge_index] * inv;
    }
    const float reduced = block_sum(dot, partial);
    if (threadIdx.x == 0) grad_weight[edge] = reduced;
    __syncthreads();
  }
}

__global__ void so2_reduce_inverse_backward_kernel(
    const float* grad_output, const float* pre_norm, float* grad_inverse,
    int64_t n_nodes, int64_t plane) {
  __shared__ float partial[kBlockSize];
  const int64_t node = static_cast<int64_t>(blockIdx.x);
  if (node >= n_nodes) return;
  float dot = 0.0f;
  const int64_t offset = node * plane;
  for (int64_t inner = threadIdx.x; inner < plane; inner += blockDim.x) {
    dot += grad_output[offset + inner] * pre_norm[offset + inner];
  }
  const float reduced = block_sum(dot, partial);
  if (threadIdx.x == 0) grad_inverse[node] = reduced;
}
}  // namespace

std::tuple<Tensor, Tensor> so2_envelope_reduce_forward_hip(
    Tensor message, Tensor edge_weight, Tensor dst, Tensor inverse) {
  check_so2_inputs(message, edge_weight, dst, inverse);
  const int64_t n_edges = message.size(0);
  const int64_t n_coeff = message.size(1);
  const int64_t n_channels = message.size(2);
  const int64_t n_nodes = inverse.size(0);
  const int64_t plane = checked_product(n_coeff, n_channels, "SO(2) reduce");
  const int64_t work_items = checked_product(n_edges, plane, "SO(2) reduce");
  const int64_t output_items = checked_product(n_nodes, plane, "SO(2) output");
  Tensor pre_norm = torch::zeros({n_nodes, n_coeff, n_channels}, message.options());
  if (work_items == 0) {
    return std::make_tuple(pre_norm, pre_norm);
  }
  Tensor output = torch::empty_like(pre_norm);
  TORCH_CHECK(n_edges <= static_cast<int64_t>(
                  std::numeric_limits<unsigned int>::max()),
              "SO(2) edge count exceeds the HIP grid limit");

  c10::cuda::CUDAGuard guard(message.device());
  hipLaunchKernelGGL(so2_reduce_scatter_kernel,
      dim3(grid_size(work_items, "SO(2) reduce")), dim3(kBlockSize), 0,
      at::cuda::getCurrentCUDAStream(), message.data_ptr<float>(),
      edge_weight.data_ptr<float>(), dst.data_ptr<int64_t>(),
      pre_norm.data_ptr<float>(), n_edges, n_coeff, n_channels, work_items);
  launch_check();
  hipLaunchKernelGGL(so2_reduce_scale_kernel,
      dim3(grid_size(output_items, "SO(2) output")), dim3(kBlockSize), 0,
      at::cuda::getCurrentCUDAStream(), pre_norm.data_ptr<float>(),
      inverse.data_ptr<float>(), output.data_ptr<float>(), n_nodes, plane,
      output_items);
  launch_check();
  return std::make_tuple(output, pre_norm);
}

std::tuple<Tensor, Tensor, Tensor> so2_envelope_reduce_backward_hip(
    Tensor grad_output, Tensor message, Tensor edge_weight, Tensor dst,
    Tensor inverse, Tensor pre_norm) {
  check_so2_inputs(message, edge_weight, dst, inverse);
  f32(grad_output);
  f32(pre_norm);
  same_device(message, {grad_output, pre_norm});
  const int64_t n_edges = message.size(0);
  const int64_t n_coeff = message.size(1);
  const int64_t n_channels = message.size(2);
  const int64_t n_nodes = inverse.size(0);
  const int64_t plane = checked_product(n_coeff, n_channels, "SO(2) reduce");
  TORCH_CHECK(grad_output.sizes() == pre_norm.sizes() &&
                  grad_output.size(0) == n_nodes,
              "SO(2) grad_output must have shape (n_nodes, n_coeff, n_channels)");
  Tensor grad_message = torch::empty_like(message);
  Tensor grad_weight = torch::empty_like(edge_weight);
  Tensor grad_inverse = torch::zeros_like(inverse);
  if (n_edges == 0) {
    return std::make_tuple(grad_message, grad_weight, grad_inverse);
  }
  TORCH_CHECK(n_edges <= static_cast<int64_t>(
                  std::numeric_limits<unsigned int>::max()),
              "SO(2) edge count exceeds the HIP grid limit");
  TORCH_CHECK(n_nodes <= static_cast<int64_t>(
                  std::numeric_limits<unsigned int>::max()),
              "SO(2) node count exceeds the HIP grid limit");

  c10::cuda::CUDAGuard guard(message.device());
  hipLaunchKernelGGL(so2_reduce_edge_backward_kernel,
      dim3(static_cast<unsigned int>(n_edges)), dim3(kBlockSize), 0,
      at::cuda::getCurrentCUDAStream(), grad_output.data_ptr<float>(),
      message.data_ptr<float>(), edge_weight.data_ptr<float>(),
      dst.data_ptr<int64_t>(), inverse.data_ptr<float>(),
      grad_message.data_ptr<float>(), grad_weight.data_ptr<float>(), n_edges,
      plane);
  launch_check();
  if (n_nodes > 0) {
    hipLaunchKernelGGL(so2_reduce_inverse_backward_kernel,
        dim3(static_cast<unsigned int>(n_nodes)), dim3(kBlockSize), 0,
        at::cuda::getCurrentCUDAStream(), grad_output.data_ptr<float>(),
        pre_norm.data_ptr<float>(), grad_inverse.data_ptr<float>(), n_nodes,
        plane);
    launch_check();
  }
  return std::make_tuple(grad_message, grad_weight, grad_inverse);
}
