#include "../ops.h"
#include "../checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {
constexpr unsigned int kBlockSize = 256;

void check_index_tensor(const Tensor& tensor, int64_t length, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kInt64 &&
                  tensor.is_contiguous(),
              name, " must be a contiguous DCU int64 tensor");
  TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == length,
              name, " must have shape (", length, ",)");
}

void check_gie_inputs(const Tensor& zonal, const Tensor& radial,
                      const Tensor& dst, const Tensor& radial_slot,
                      const Tensor& inv_sqrt_deg) {
  f32(zonal);
  f32(radial);
  f32(inv_sqrt_deg);
  TORCH_CHECK(zonal.dim() == 2,
              "GIE zonal must have shape (n_edges, n_rows)");
  TORCH_CHECK(radial.dim() == 3,
              "GIE radial must have shape (n_edges, n_radial, channels)");
  const int64_t n_edges = zonal.size(0);
  const int64_t n_rows = zonal.size(1);
  TORCH_CHECK(radial.size(0) == n_edges,
              "GIE radial edge dimension must match zonal");
  check_index_tensor(dst, n_edges, "GIE dst");
  check_index_tensor(radial_slot, n_rows, "GIE radial_slot");
  TORCH_CHECK(inv_sqrt_deg.dim() == 1 || inv_sqrt_deg.dim() == 3,
              "GIE inv_sqrt_deg must have shape (n_nodes,) or (n_nodes, 1, 1)");
  TORCH_CHECK(inv_sqrt_deg.dim() != 3 ||
                  (inv_sqrt_deg.size(1) == 1 && inv_sqrt_deg.size(2) == 1),
              "GIE inv_sqrt_deg must have shape (n_nodes, 1, 1)");
  same_device(zonal, {radial, dst, radial_slot, inv_sqrt_deg});
}

int64_t check_work_size(int64_t n_edges, int64_t n_rows, int64_t channels) {
  const int64_t limit = std::numeric_limits<int64_t>::max();
  TORCH_CHECK(n_edges == 0 || n_rows <= limit / n_edges,
              "GIE scatter work size overflows int64");
  const int64_t edge_rows = n_edges * n_rows;
  TORCH_CHECK(edge_rows == 0 || channels <= limit / edge_rows,
              "GIE scatter work size overflows int64");
  return edge_rows * channels;
}

unsigned int grid_size(int64_t work_items) {
  const int64_t blocks = work_items / kBlockSize +
      (work_items % kBlockSize != 0);
  TORCH_CHECK(blocks <= static_cast<int64_t>(
                  std::numeric_limits<unsigned int>::max()),
              "GIE scatter grid exceeds the HIP launch limit");
  return static_cast<unsigned int>(blocks);
}

__global__ void gie_zonal_scatter_forward_kernel(
    const float* zonal, const float* radial, const int64_t* dst,
    const int64_t* radial_slot, const float* inv_sqrt_deg, float* output,
    int64_t n_edges, int64_t n_rows, int64_t n_radial, int64_t channels,
    int64_t work_items) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = first; index < work_items; index += stride) {
    const int64_t channel = index % channels;
    const int64_t row = (index / channels) % n_rows;
    const int64_t edge = index / (channels * n_rows);
    const int64_t node = dst[edge];
    const int64_t slot = radial_slot[row];
    const float value = zonal[edge * n_rows + row] *
        radial[(edge * n_radial + slot) * channels + channel] *
        inv_sqrt_deg[node];
    const int64_t output_index =
        (node * (n_rows + 1) + row + 1) * channels + channel;
    atomicAdd(output + output_index, value);
  }
}

__global__ void gie_zonal_scatter_backward_kernel(
    const float* grad_output, const float* zonal, const float* radial,
    const int64_t* dst, const int64_t* radial_slot,
    const float* inv_sqrt_deg, float* grad_zonal, float* grad_radial,
    float* grad_inv_sqrt_deg, int64_t n_edges, int64_t n_rows,
    int64_t n_radial, int64_t channels, int64_t work_items) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = first; index < work_items; index += stride) {
    const int64_t channel = index % channels;
    const int64_t row = (index / channels) % n_rows;
    const int64_t edge = index / (channels * n_rows);
    const int64_t node = dst[edge];
    const int64_t slot = radial_slot[row];
    const int64_t radial_index =
        (edge * n_radial + slot) * channels + channel;
    const int64_t output_index =
        (node * (n_rows + 1) + row + 1) * channels + channel;
    const float upstream = grad_output[output_index];
    const float zonal_value = zonal[edge * n_rows + row];
    const float radial_value = radial[radial_index];
    const float inverse = inv_sqrt_deg[node];

    atomicAdd(grad_zonal + edge * n_rows + row,
              upstream * radial_value * inverse);
    atomicAdd(grad_radial + radial_index,
              upstream * zonal_value * inverse);
    atomicAdd(grad_inv_sqrt_deg + node,
              upstream * zonal_value * radial_value);
  }
}
} // namespace

Tensor gie_zonal_scatter_forward_hip(
    Tensor zonal, Tensor radial, Tensor dst, Tensor radial_slot,
    Tensor inv_sqrt_deg) {
  check_gie_inputs(zonal, radial, dst, radial_slot, inv_sqrt_deg);
  const int64_t n_edges = zonal.size(0);
  const int64_t n_rows = zonal.size(1);
  const int64_t n_radial = radial.size(1);
  const int64_t channels = radial.size(2);
  const int64_t n_nodes = inv_sqrt_deg.size(0);
  Tensor output = torch::zeros(
      {n_nodes, n_rows + 1, channels}, zonal.options());
  const int64_t work_items = check_work_size(n_edges, n_rows, channels);
  if (work_items == 0) return output;

  c10::cuda::CUDAGuard guard(zonal.device());
  hipLaunchKernelGGL(gie_zonal_scatter_forward_kernel,
      dim3(grid_size(work_items)), dim3(kBlockSize), 0,
      at::cuda::getCurrentCUDAStream(), zonal.data_ptr<float>(),
      radial.data_ptr<float>(), dst.data_ptr<int64_t>(),
      radial_slot.data_ptr<int64_t>(), inv_sqrt_deg.data_ptr<float>(),
      output.data_ptr<float>(), n_edges, n_rows, n_radial, channels,
      work_items);
  launch_check();
  return output;
}

std::tuple<Tensor, Tensor, Tensor> gie_zonal_scatter_backward_hip(
    Tensor grad_output, Tensor zonal, Tensor radial, Tensor dst,
    Tensor radial_slot, Tensor inv_sqrt_deg) {
  check_gie_inputs(zonal, radial, dst, radial_slot, inv_sqrt_deg);
  f32(grad_output);
  same_device(zonal, {grad_output});

  const int64_t n_edges = zonal.size(0);
  const int64_t n_rows = zonal.size(1);
  const int64_t n_radial = radial.size(1);
  const int64_t channels = radial.size(2);
  const int64_t n_nodes = inv_sqrt_deg.size(0);
  TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(0) == n_nodes &&
                  grad_output.size(1) == n_rows + 1 &&
                  grad_output.size(2) == channels,
              "GIE grad_output must have shape (n_nodes, n_rows + 1, channels)");

  Tensor grad_zonal = torch::zeros_like(zonal);
  Tensor grad_radial = torch::zeros_like(radial);
  Tensor grad_inv_sqrt_deg = torch::zeros_like(inv_sqrt_deg);
  const int64_t work_items = check_work_size(n_edges, n_rows, channels);
  if (work_items == 0)
    return std::make_tuple(grad_zonal, grad_radial, grad_inv_sqrt_deg);

  c10::cuda::CUDAGuard guard(zonal.device());
  hipLaunchKernelGGL(gie_zonal_scatter_backward_kernel,
      dim3(grid_size(work_items)), dim3(kBlockSize), 0,
      at::cuda::getCurrentCUDAStream(), grad_output.data_ptr<float>(),
      zonal.data_ptr<float>(), radial.data_ptr<float>(),
      dst.data_ptr<int64_t>(), radial_slot.data_ptr<int64_t>(),
      inv_sqrt_deg.data_ptr<float>(), grad_zonal.data_ptr<float>(),
      grad_radial.data_ptr<float>(), grad_inv_sqrt_deg.data_ptr<float>(),
      n_edges, n_rows, n_radial, channels, work_items);
  launch_check();
  return std::make_tuple(grad_zonal, grad_radial, grad_inv_sqrt_deg);
}
