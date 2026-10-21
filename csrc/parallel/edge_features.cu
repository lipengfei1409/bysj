// Fused per-edge geometry, radial basis and cutoff candidate for DPA4.
#include "../include/ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include "../include/checks.h"
#include "../include/megakernel/edge_features.cuh"

#include <cmath>

namespace {

__global__ void edge_features_forward_kernel(dcu::megakernel::EdgeFeaturesArgs args) {
  const int64_t edge = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (edge < args.n_edges) dcu::megakernel::edge_features_one(args, edge);
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
  dcu::megakernel::c3_envelope_pair(r, cutoff, basis_exponent, basis_env, basis_derivative);
  dcu::megakernel::c3_envelope_pair(r, cutoff, edge_exponent, edge_env, edge_derivative);
  grad_r += grad_env[edge] * edge_derivative;
  for (int64_t n = 0; n < n_radial; ++n) {
    const float w = frequencies[n];
    float sinc, sinc_derivative;
    dcu::megakernel::sinc_pair(r * w, sinc, sinc_derivative);
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
    dcu::megakernel::EdgeFeaturesArgs args{
        positions.data_ptr<float>(), edge_index.data_ptr<int64_t>(), shifts.data_ptr<int64_t>(),
        cells.data_ptr<float>(), edge_frame.data_ptr<int64_t>(), mask.data_ptr<bool>(),
        frequencies.data_ptr<float>(), n_edges, n_radial, float(cutoff),
        int(basis_exponent), int(edge_exponent), float(eps), vec.data_ptr<float>(),
        distance.data_ptr<float>(), unit.data_ptr<float>(), rbf.data_ptr<float>(), env.data_ptr<float>()};
    hipLaunchKernelGGL(edge_features_forward_kernel, dim3(blocks(n_edges)), dim3(256), 0,
        at::cuda::getCurrentCUDAStream(), args);
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
