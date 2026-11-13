#include "ops.h"
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
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

__global__ void rmsnorm_forward_kernel(
    const float* x, const float* scale, float* output, float* inverse_rms,
    int64_t rows, int64_t width, float eps, bool has_scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) return;

  __shared__ float partial[kBlockSize];
  const int64_t offset = row * width;
  float sum_squares = 0.0f;
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = x[offset + col];
    sum_squares += value * value;
  }
  const float total = block_sum(sum_squares, partial);
  const float inv = rsqrtf(total / static_cast<float>(width) + eps);
  if (threadIdx.x == 0) inverse_rms[row] = inv;

  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    float value = x[offset + col] * inv;
    if (has_scale) value *= scale[col];
    output[offset + col] = value;
  }
}

__global__ void rmsnorm_backward_kernel(
    const float* grad_output, const float* x, const float* scale,
    const float* inverse_rms, float* grad_x, float* grad_scale,
    int64_t rows, int64_t width, bool has_scale, bool compute_scale_grad) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) return;

  __shared__ float partial[kBlockSize];
  const int64_t offset = row * width;
  float dot = 0.0f;
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    const float upstream = grad_output[offset + col] *
        (has_scale ? scale[col] : 1.0f);
    dot += x[offset + col] * upstream;
  }
  const float mean_dot = block_sum(dot, partial) / static_cast<float>(width);
  const float inv = inverse_rms[row];
  const float inv_cubed = inv * inv * inv;

  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    const float upstream = grad_output[offset + col] *
        (has_scale ? scale[col] : 1.0f);
    grad_x[offset + col] = inv * upstream -
        x[offset + col] * inv_cubed * mean_dot;
    if (compute_scale_grad) {
      const float normalized = x[offset + col] * inv;
      atomicAdd(grad_scale + col, grad_output[offset + col] * normalized);
    }
  }
}

void check_rmsnorm_inputs(const Tensor& x, const Tensor& scale) {
  f32(x);
  f32(scale);
  same_device(x, {scale});
  TORCH_CHECK(x.dim() >= 1,
              "RMSNorm input must have at least one dimension");
  TORCH_CHECK(x.size(-1) > 0,
              "RMSNorm feature dimension must be positive");
  TORCH_CHECK(scale.dim() == 1 &&
              (scale.size(0) == 0 || scale.size(0) == x.size(-1)),
              "RMSNorm scale must be an empty vector or have shape (feature_dim,)");
}

void check_grid_size(int64_t rows) {
  TORCH_CHECK(rows <= static_cast<int64_t>(
                  std::numeric_limits<unsigned int>::max()),
              "RMSNorm row count exceeds the HIP grid limit");
}
} // namespace

std::tuple<Tensor, Tensor> rmsnorm_forward_hip(
    Tensor x, Tensor scale, double eps) {
  check_rmsnorm_inputs(x, scale);
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0,
              "RMSNorm eps must be a finite positive number");

  const int64_t width = x.size(-1);
  const int64_t rows = x.numel() / width;
  Tensor output = torch::empty_like(x);
  Tensor inverse_rms = torch::empty({rows}, x.options());
  if (rows == 0) return std::make_tuple(output, inverse_rms);
  check_grid_size(rows);

  const bool has_scale = scale.numel() != 0;
  c10::cuda::CUDAGuard guard(x.device());
  hipLaunchKernelGGL(rmsnorm_forward_kernel, dim3(static_cast<unsigned int>(rows)),
      dim3(kBlockSize), 0, at::cuda::getCurrentCUDAStream(),
      x.data_ptr<float>(), scale.data_ptr<float>(), output.data_ptr<float>(),
      inverse_rms.data_ptr<float>(), rows, width, static_cast<float>(eps),
      has_scale);
  launch_check();
  return std::make_tuple(output, inverse_rms);
}

std::tuple<Tensor, Tensor> rmsnorm_backward_hip(
    Tensor grad_output, Tensor x, Tensor scale, Tensor inverse_rms,
    bool compute_scale_grad) {
  check_rmsnorm_inputs(x, scale);
  f32(grad_output);
  f32(inverse_rms);
  same_device(x, {grad_output, inverse_rms});
  TORCH_CHECK(grad_output.sizes() == x.sizes(),
              "RMSNorm grad_output must have the input shape");

  const int64_t width = x.size(-1);
  const int64_t rows = x.numel() / width;
  TORCH_CHECK(inverse_rms.dim() == 1 && inverse_rms.size(0) == rows,
              "RMSNorm saved inverse_rms has an invalid shape");
  const bool has_scale = scale.numel() != 0;
  TORCH_CHECK(!compute_scale_grad || has_scale,
              "cannot compute a scale gradient when scale is omitted");

  Tensor grad_x = torch::empty_like(x);
  Tensor grad_scale = compute_scale_grad
      ? torch::zeros({width}, x.options())
      : torch::empty({0}, x.options());
  if (rows == 0) return std::make_tuple(grad_x, grad_scale);
  check_grid_size(rows);

  c10::cuda::CUDAGuard guard(x.device());
  hipLaunchKernelGGL(rmsnorm_backward_kernel,
      dim3(static_cast<unsigned int>(rows)), dim3(kBlockSize), 0,
      at::cuda::getCurrentCUDAStream(), grad_output.data_ptr<float>(),
      x.data_ptr<float>(), scale.data_ptr<float>(),
      inverse_rms.data_ptr<float>(), grad_x.data_ptr<float>(),
      grad_scale.data_ptr<float>(), rows, width, has_scale, compute_scale_grad);
  launch_check();
  return std::make_tuple(grad_x, grad_scale);
}
