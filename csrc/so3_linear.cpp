#include "ops.h"
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>

namespace {
constexpr unsigned int kBlockSize = 256;

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

void check_so3_linear_inputs(const Tensor& x, const Tensor& weight,
                             const Tensor& bias, const Tensor& expand_index) {
  f32(x);
  f32(weight);
  f32(bias);
  TORCH_CHECK(x.dim() == 4,
              "SO3Linear input must have shape (n_nodes, n_coeff, n_focus, in_channels)");
  TORCH_CHECK(weight.dim() == 3 && weight.size(0) > 0 && weight.size(1) > 0 &&
                  weight.size(2) > 0,
              "SO3Linear weight must have shape (lmax + 1, in_channels, n_focus * out_channels)");
  TORCH_CHECK(expand_index.is_cuda() &&
                  expand_index.scalar_type() == torch::kInt64 &&
                  expand_index.is_contiguous(),
              "SO3Linear expand_index must be a contiguous DCU int64 tensor");
  const int64_t n_nodes = x.size(0);
  const int64_t n_coeff = x.size(1);
  const int64_t n_focus = x.size(2);
  const int64_t in_channels = x.size(3);
  const int64_t n_degree = weight.size(0);
  TORCH_CHECK(n_focus > 0 && in_channels == weight.size(1),
              "SO3Linear input channel/focus dimensions do not match weight");
  TORCH_CHECK(n_degree <= std::numeric_limits<int64_t>::max() / n_degree &&
                  n_coeff == n_degree * n_degree,
              "SO3Linear coefficient dimension must equal (lmax + 1)^2");
  TORCH_CHECK(weight.size(2) % n_focus == 0,
              "SO3Linear weight output width must be divisible by n_focus");
  TORCH_CHECK(expand_index.dim() == 1 && expand_index.size(0) == n_coeff,
              "SO3Linear expand_index must have shape (n_coeff,)");
  TORCH_CHECK(bias.dim() == 1 &&
                  (bias.numel() == 0 || bias.numel() == weight.size(2)),
              "SO3Linear bias must be empty or have shape (n_focus * out_channels,)");
  same_device(x, {weight, bias, expand_index});
  (void)n_nodes;
}

__global__ void so3_linear_forward_kernel(
    const float* x, const float* weight, const float* bias,
    const int64_t* expand_index, float* output, int64_t n_nodes,
    int64_t n_coeff, int64_t n_focus, int64_t in_channels,
    int64_t out_channels, int64_t output_coeffs, bool has_bias,
    int64_t work_items) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t focus_width = n_focus * out_channels;
  for (int64_t index = first; index < work_items; index += stride) {
    const int64_t out_channel = index % out_channels;
    const int64_t focus = (index / out_channels) % n_focus;
    const int64_t coeff = (index / focus_width) % output_coeffs;
    const int64_t node = index / (focus_width * output_coeffs);
    const int64_t input_coeff = coeff;
    const int64_t degree = expand_index[input_coeff];
    const int64_t input_offset =
        ((node * n_coeff + input_coeff) * n_focus + focus) * in_channels;
    const int64_t weight_offset =
        (degree * in_channels) * focus_width + focus * out_channels + out_channel;
    float value = 0.0f;
    for (int64_t channel = 0; channel < in_channels; ++channel) {
      value += x[input_offset + channel] *
          weight[weight_offset + channel * focus_width];
    }
    if (has_bias && input_coeff == 0) {
      value += bias[focus * out_channels + out_channel];
    }
    output[index] = value;
  }
}

__global__ void so3_linear_backward_input_kernel(
    const float* grad_output, const float* weight, const int64_t* expand_index,
    float* grad_x, int64_t n_nodes, int64_t n_coeff, int64_t n_focus,
    int64_t in_channels, int64_t out_channels, int64_t output_coeffs,
    bool scalar_only, int64_t work_items) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t focus_width = n_focus * in_channels;
  for (int64_t index = first; index < work_items; index += stride) {
    const int64_t input_channel = index % in_channels;
    const int64_t focus = (index / in_channels) % n_focus;
    const int64_t coeff = (index / focus_width) % n_coeff;
    const int64_t node = index / (focus_width * n_coeff);
    if (scalar_only && coeff != 0) {
      grad_x[index] = 0.0f;
      continue;
    }
    const int64_t output_coeff = scalar_only ? 0 : coeff;
    const int64_t degree = expand_index[coeff];
    const int64_t weight_offset =
        (degree * in_channels + input_channel) * (n_focus * out_channels) +
        focus * out_channels;
    const int64_t grad_offset =
        ((node * output_coeffs + output_coeff) * n_focus + focus) * out_channels;
    float value = 0.0f;
    for (int64_t out_channel = 0; out_channel < out_channels; ++out_channel) {
      value += grad_output[grad_offset + out_channel] *
          weight[weight_offset + out_channel];
    }
    grad_x[index] = value;
  }
}
}  // namespace

Tensor so3_linear_forward_hip(
    Tensor x, Tensor weight, Tensor bias, Tensor expand_index, bool scalar_only) {
  check_so3_linear_inputs(x, weight, bias, expand_index);
  const int64_t n_nodes = x.size(0);
  const int64_t n_coeff = x.size(1);
  const int64_t n_focus = x.size(2);
  const int64_t in_channels = x.size(3);
  const int64_t out_channels = weight.size(2) / n_focus;
  const int64_t output_coeffs = scalar_only ? 1 : n_coeff;
  const int64_t output_rows = checked_product(n_nodes, output_coeffs, "SO3Linear");
  const int64_t work_items = checked_product(
      checked_product(output_rows, n_focus, "SO3Linear"), out_channels,
      "SO3Linear");
  Tensor output = torch::empty(
      {n_nodes, output_coeffs, n_focus, out_channels}, x.options());
  if (work_items == 0) return output;

  c10::cuda::CUDAGuard guard(x.device());
  hipLaunchKernelGGL(so3_linear_forward_kernel,
      dim3(grid_size(work_items, "SO3Linear")), dim3(kBlockSize), 0,
      at::cuda::getCurrentCUDAStream(), x.data_ptr<float>(),
      weight.data_ptr<float>(), bias.data_ptr<float>(),
      expand_index.data_ptr<int64_t>(), output.data_ptr<float>(), n_nodes,
      n_coeff, n_focus, in_channels, out_channels, output_coeffs,
      bias.numel() != 0, work_items);
  launch_check();
  return output;
}

Tensor so3_linear_backward_input_hip(
    Tensor grad_output, Tensor x, Tensor weight, Tensor expand_index,
    bool scalar_only) {
  check_so3_linear_inputs(x, weight, torch::empty({0}, x.options()), expand_index);
  f32(grad_output);
  const int64_t n_nodes = x.size(0);
  const int64_t n_coeff = x.size(1);
  const int64_t n_focus = x.size(2);
  const int64_t in_channels = x.size(3);
  const int64_t out_channels = weight.size(2) / n_focus;
  const int64_t output_coeffs = scalar_only ? 1 : n_coeff;
  TORCH_CHECK(grad_output.dim() == 4 && grad_output.size(0) == n_nodes &&
                  grad_output.size(1) == output_coeffs &&
                  grad_output.size(2) == n_focus &&
                  grad_output.size(3) == out_channels,
              "SO3Linear grad_output has an invalid shape");
  same_device(x, {grad_output});
  const int64_t work_items = checked_product(
      checked_product(checked_product(n_nodes, n_coeff, "SO3Linear"), n_focus,
                      "SO3Linear"),
      in_channels, "SO3Linear");
  Tensor grad_x = torch::empty_like(x);
  if (work_items == 0) return grad_x;

  c10::cuda::CUDAGuard guard(x.device());
  hipLaunchKernelGGL(so3_linear_backward_input_kernel,
      dim3(grid_size(work_items, "SO3Linear backward")), dim3(kBlockSize), 0,
      at::cuda::getCurrentCUDAStream(), grad_output.data_ptr<float>(),
      weight.data_ptr<float>(), expand_index.data_ptr<int64_t>(),
      grad_x.data_ptr<float>(), n_nodes, n_coeff, n_focus, in_channels,
      out_channels, output_coeffs, scalar_only, work_items);
  launch_check();
  return grad_x;
}
