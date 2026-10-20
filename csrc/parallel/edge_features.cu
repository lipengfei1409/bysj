// Fused per-edge geometry, radial basis and cutoff candidate for DPA4.
#include "../include/ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include "../include/checks.h"

#include <cmath>

namespace {

__device__ void c3_envelope_pair(float r, float cutoff, int exponent,
                                 float& value, float& derivative) {
  if (exponent == 0) {
    value = 1.f;
    derivative = 0.f;
    return;
  }
  if (r >= cutoff) {
    value = derivative = 0.f;
    return;
  }
  const float x = fmaxf(0.f, r / cutoff);
  const float u = 1.f - x;
  float sum = 0.f;
  float dsum = 0.f;
  float power = 1.f;
  float previous = 1.f;
  for (int k = 0; k < exponent; ++k) {
    const float coefficient = float((k + 1) * (k + 2) * (k + 3)) / 6.f;
    sum += coefficient * power;
    if (k > 0) dsum += float(k) * coefficient * previous;
    previous = power;
    power *= x;
  }
  const float u2 = u * u;
  const float u3 = u2 * u;
  const float u4 = u2 * u2;
  value = u4 * sum;
  derivative = (r < 0.f) ? 0.f : (-4.f * u3 * sum + u4 * dsum) / cutoff;
}

__device__ void sinc_pair(float z, float& value, float& derivative) {
  if (fabsf(z) < 1.e-3f) {
    const float z2 = z * z;
    value = 1.f - z2 / 6.f + z2 * z2 / 120.f;
    derivative = -z / 3.f + z * z2 / 30.f;
  } else {
    value = sinf(z) / z;
    derivative = (z * cosf(z) - sinf(z)) / (z * z);
  }
}

__global__ void edge_features_forward_kernel(
    const float* positions, const int64_t* edge_index, const int64_t* shifts,
    const float* cells, const int64_t* edge_frame, const bool* mask,
    const float* frequencies, int64_t n_edges, int64_t n_radial,
    float cutoff, int basis_exponent, int edge_exponent, float eps,
    float* edge_vec, float* distance, float* unit, float* rbf, float* env) {
  const int64_t edge = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (edge >= n_edges) return;
  const int64_t src = edge_index[edge];
  const int64_t dst = edge_index[n_edges + edge];
  const int64_t frame = edge_frame[edge];
  float displacement[3];
  float norm2 = 0.f;
  for (int axis = 0; axis < 3; ++axis) {
    float value = positions[3 * src + axis] - positions[3 * dst + axis];
    for (int lattice = 0; lattice < 3; ++lattice) {
      value += float(shifts[3 * edge + lattice]) * cells[9 * frame + 3 * lattice + axis];
    }
    displacement[axis] = value;
    norm2 += value * value;
  }
  if (!mask[edge]) {
    displacement[0] = 0.f;
    displacement[1] = 0.f;
    displacement[2] = 1.f;
    norm2 = 1.f;
  }
  const float r = sqrtf(norm2 + eps * eps);
  const float inv_r = 1.f / r;
  edge_vec[3 * edge] = displacement[0];
  edge_vec[3 * edge + 1] = displacement[1];
  edge_vec[3 * edge + 2] = displacement[2];
  distance[edge] = r;
  unit[3 * edge] = displacement[0] * inv_r;
  unit[3 * edge + 1] = displacement[1] * inv_r;
  unit[3 * edge + 2] = displacement[2] * inv_r;

  if (!mask[edge]) {
    env[edge] = 0.f;
    for (int64_t n = 0; n < n_radial; ++n) rbf[edge * n_radial + n] = 0.f;
    return;
  }
  float basis_env, basis_derivative, edge_env, edge_derivative;
  c3_envelope_pair(r, cutoff, basis_exponent, basis_env, basis_derivative);
  c3_envelope_pair(r, cutoff, edge_exponent, edge_env, edge_derivative);
  env[edge] = edge_env;
  for (int64_t n = 0; n < n_radial; ++n) {
    float sinc, sinc_derivative;
    sinc_pair(r * frequencies[n], sinc, sinc_derivative);
    rbf[edge * n_radial + n] = frequencies[n] * sinc * basis_env;
  }
}

__global__ void edge_features_backward_kernel(
    const float* grad_vec, const float* grad_distance, const float* grad_unit,
    const float* grad_rbf, const float* grad_env, const float* edge_vec,
    const float* distance, const float* unit, const int64_t* edge_index,
    const int64_t* shifts, const float* cells, const int64_t* edge_frame,
    const bool* mask, const float* frequencies, int64_t n_edges,
    int64_t n_radial, float cutoff, int basis_exponent, int edge_exponent,
    int64_t n_atoms, int64_t n_frames, float* grad_positions,
    float* grad_cells, float* grad_frequencies) {
  const int64_t edge = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (edge >= n_edges || !mask[edge]) return;
  const int64_t src = edge_index[edge];
  const int64_t dst = edge_index[n_edges + edge];
  const int64_t frame = edge_frame[edge];
  const float r = distance[edge];
  float dot = 0.f;
  for (int axis = 0; axis < 3; ++axis) dot += grad_unit[3 * edge + axis] * unit[3 * edge + axis];
  float grad_r = grad_distance[edge];
  float basis_env, basis_derivative, edge_env, edge_derivative;
  c3_envelope_pair(r, cutoff, basis_exponent, basis_env, basis_derivative);
  c3_envelope_pair(r, cutoff, edge_exponent, edge_env, edge_derivative);
  grad_r += grad_env[edge] * edge_derivative;
  for (int64_t n = 0; n < n_radial; ++n) {
    const float w = frequencies[n];
    float sinc, sinc_derivative;
    sinc_pair(r * w, sinc, sinc_derivative);
    const float grad = grad_rbf[edge * n_radial + n];
    grad_r += grad * (w * w * sinc_derivative * basis_env + w * sinc * basis_derivative);
    atomicAdd(grad_frequencies + n,
              grad * (sinc + r * w * sinc_derivative) * basis_env);
  }
  const float inv_r = 1.f / r;
  for (int axis = 0; axis < 3; ++axis) {
    const float u = unit[3 * edge + axis];
    const float grad = grad_vec[3 * edge + axis]
        + grad_r * u
        + (grad_unit[3 * edge + axis] - u * dot) * inv_r;
    atomicAdd(grad_positions + 3 * src + axis, grad);
    atomicAdd(grad_positions + 3 * dst + axis, -grad);
    for (int lattice = 0; lattice < 3; ++lattice) {
      atomicAdd(grad_cells + 9 * frame + 3 * lattice + axis,
                float(shifts[3 * edge + lattice]) * grad);
    }
  }
}

void check_edge_feature_inputs(const Tensor& positions, const Tensor& edge_index,
                               const Tensor& shifts, const Tensor& cells,
                               const Tensor& edge_frame, const Tensor& mask,
                               const Tensor& frequencies, double cutoff,
                               int64_t basis_exponent, int64_t edge_exponent,
                               double eps) {
  f32(positions); f32(cells); f32(frequencies);
  i64(edge_index); i64(shifts); i64(edge_frame);
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "edge mask must be bool");
  same_device(positions, {edge_index, shifts, cells, edge_frame, mask, frequencies});
  TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "positions must be [N,3]");
  TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2, "edge_index must be [2,E]");
  const int64_t n_edges = edge_index.size(1);
  TORCH_CHECK(shifts.dim() == 2 && shifts.size(0) == n_edges && shifts.size(1) == 3, "shifts must be [E,3]");
  TORCH_CHECK(edge_frame.dim() == 1 && edge_frame.numel() == n_edges, "edge_frame must be [E]");
  TORCH_CHECK(mask.dim() == 1 && mask.numel() == n_edges, "mask must be [E]");
  TORCH_CHECK(cells.dim() == 3 && cells.size(0) > 0 && cells.size(1) == 3 && cells.size(2) == 3, "cells must be [F,3,3], F>0");
  TORCH_CHECK(frequencies.dim() == 2 && frequencies.size(0) == 1 && frequencies.size(1) > 0,
              "frequencies must be [1,R]");
  TORCH_CHECK(cutoff > 0.0 && eps > 0.0 && basis_exponent >= 0 && edge_exponent >= 0
              && basis_exponent <= 64 && edge_exponent <= 64, "invalid edge feature parameters");
}

}  // namespace

std::vector<Tensor> edge_features_hip(Tensor positions, Tensor edge_index,
    Tensor shifts, Tensor cells, Tensor edge_frame, Tensor mask, Tensor frequencies,
    double cutoff, int64_t basis_exponent, int64_t edge_exponent, double eps) {
  check_edge_feature_inputs(positions, edge_index, shifts, cells, edge_frame, mask,
                            frequencies, cutoff, basis_exponent, edge_exponent, eps);
  c10::cuda::CUDAGuard guard(positions.device());
  const int64_t n_edges = edge_index.size(1), n_radial = frequencies.size(1);
  auto vec = torch::empty({n_edges, 3}, positions.options());
  auto distance = torch::empty({n_edges, 1}, positions.options());
  auto unit = torch::empty({n_edges, 3}, positions.options());
  auto rbf = torch::empty({n_edges, n_radial}, positions.options());
  auto env = torch::empty({n_edges, 1}, positions.options());
  if (n_edges) {
    hipLaunchKernelGGL(edge_features_forward_kernel, dim3(blocks(n_edges)), dim3(256), 0,
        at::cuda::getCurrentCUDAStream(), positions.data_ptr<float>(), edge_index.data_ptr<int64_t>(),
        shifts.data_ptr<int64_t>(), cells.data_ptr<float>(), edge_frame.data_ptr<int64_t>(),
        mask.data_ptr<bool>(), frequencies.data_ptr<float>(), n_edges, n_radial,
        float(cutoff), int(basis_exponent), int(edge_exponent), float(eps), vec.data_ptr<float>(),
        distance.data_ptr<float>(), unit.data_ptr<float>(), rbf.data_ptr<float>(), env.data_ptr<float>());
    launch_check();
  }
  return {vec, distance, unit, rbf, env};
}

std::vector<Tensor> edge_features_backward_hip(Tensor grad_vec, Tensor grad_distance,
    Tensor grad_unit, Tensor grad_rbf, Tensor grad_env, Tensor edge_vec, Tensor distance,
    Tensor unit, Tensor edge_index, Tensor shifts, Tensor cells, Tensor edge_frame,
    Tensor mask, Tensor frequencies, int64_t n_atoms, double cutoff,
    int64_t basis_exponent, int64_t edge_exponent, double eps) {
  f32(grad_vec); f32(grad_distance); f32(grad_unit); f32(grad_rbf); f32(grad_env);
  f32(edge_vec); f32(distance); f32(unit); f32(cells); f32(frequencies);
  i64(edge_index); i64(shifts); i64(edge_frame);
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "edge mask must be bool");
  same_device(grad_vec, {grad_distance, grad_unit, grad_rbf, grad_env, edge_vec, distance, unit,
                         edge_index, shifts, cells, edge_frame, mask, frequencies});
  const int64_t n_edges = distance.numel(), n_radial = frequencies.size(1);
  TORCH_CHECK(grad_vec.sizes() == edge_vec.sizes() && grad_unit.sizes() == unit.sizes(),
              "edge vector gradient shapes disagree");
  TORCH_CHECK(grad_distance.numel() == n_edges && grad_env.numel() == n_edges
              && grad_rbf.dim() == 2 && grad_rbf.size(0) == n_edges && grad_rbf.size(1) == n_radial,
              "edge radial gradient shapes disagree");
  TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2 && edge_index.size(1) == n_edges
              && shifts.dim() == 2 && shifts.size(0) == n_edges && shifts.size(1) == 3
              && edge_frame.numel() == n_edges && mask.numel() == n_edges,
              "edge topology shapes disagree");
  TORCH_CHECK(cells.dim() == 3 && cells.size(1) == 3 && cells.size(2) == 3
              && cells.size(0) > 0 && n_atoms >= 0 && cutoff > 0.0 && eps > 0.0
              && basis_exponent >= 0 && edge_exponent >= 0
              && basis_exponent <= 64 && edge_exponent <= 64,
              "invalid edge backward parameters");
  c10::cuda::CUDAGuard guard(edge_vec.device());
  auto grad_positions = torch::zeros({n_atoms, 3}, edge_vec.options());
  auto grad_cells = torch::zeros_like(cells);
  auto grad_frequencies = torch::zeros_like(frequencies);
  if (n_edges) {
    hipLaunchKernelGGL(edge_features_backward_kernel, dim3(blocks(n_edges)), dim3(256), 0,
        at::cuda::getCurrentCUDAStream(), grad_vec.data_ptr<float>(), grad_distance.data_ptr<float>(),
        grad_unit.data_ptr<float>(), grad_rbf.data_ptr<float>(), grad_env.data_ptr<float>(),
        edge_vec.data_ptr<float>(), distance.data_ptr<float>(), unit.data_ptr<float>(),
        edge_index.data_ptr<int64_t>(), shifts.data_ptr<int64_t>(), cells.data_ptr<float>(),
        edge_frame.data_ptr<int64_t>(), mask.data_ptr<bool>(), frequencies.data_ptr<float>(),
        n_edges, n_radial, float(cutoff), int(basis_exponent), int(edge_exponent),
        n_atoms, cells.size(0), grad_positions.data_ptr<float>(), grad_cells.data_ptr<float>(),
        grad_frequencies.data_ptr<float>());
    launch_check();
  }
  return {grad_positions, grad_cells, grad_frequencies};
}
