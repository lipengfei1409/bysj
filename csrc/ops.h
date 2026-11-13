#pragma once
#include <torch/extension.h>
#include <tuple>

using Tensor = torch::Tensor;

std::tuple<Tensor, Tensor> rmsnorm_forward_hip(
    Tensor x, Tensor scale, double eps);
std::tuple<Tensor, Tensor> rmsnorm_backward_hip(
    Tensor grad_output, Tensor x, Tensor scale, Tensor inverse_rms,
    bool compute_scale_grad);
Tensor gie_zonal_scatter_forward_hip(
    Tensor zonal, Tensor radial, Tensor dst, Tensor radial_slot,
    Tensor inv_sqrt_deg);
std::tuple<Tensor, Tensor, Tensor> gie_zonal_scatter_backward_hip(
    Tensor grad_output, Tensor zonal, Tensor radial, Tensor dst,
    Tensor radial_slot, Tensor inv_sqrt_deg);
