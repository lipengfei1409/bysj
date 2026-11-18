#pragma once
#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <tuple>

using Tensor = torch::Tensor;

std::tuple<Tensor, Tensor> rmsnorm_forward_hip(Tensor x, Tensor scale, double eps);

std::tuple<Tensor, Tensor> rmsnorm_backward_hip(Tensor grad_output, Tensor x, Tensor scale, Tensor inverse_rms, bool compute_scale_grad);

Tensor gie_zonal_scatter_forward_hip(Tensor zonal, Tensor radial, Tensor dst, Tensor radial_slot, Tensor inv_sqrt_deg);

std::tuple<Tensor, Tensor, Tensor> gie_zonal_scatter_backward_hip(Tensor grad_output, Tensor zonal, Tensor radial, Tensor dst, Tensor radial_slot, Tensor inv_sqrt_deg);

std::tuple<Tensor, Tensor> so2_envelope_reduce_forward_hip(Tensor message, Tensor edge_weight, Tensor dst, Tensor inverse);

std::tuple<Tensor, Tensor, Tensor> so2_envelope_reduce_backward_hip(Tensor grad_output, Tensor message, Tensor edge_weight, Tensor dst, Tensor inverse, Tensor pre_norm);

// --- Fused segmented operators (see docs/task2-hip-fusion-plan.md) ---------
//
// All three take a destination-grouped edge list: ``dst`` non-decreasing and
// ``offsets`` its exclusive prefix sum. The forward kernels never zero-init
// their output and never use atomics.

Tensor segment_sum_forward_hip(Tensor src, Tensor offsets,
                               c10::optional<Tensor> edge_scale,
                               c10::optional<Tensor> node_scale);

std::tuple<Tensor, Tensor> segment_sum_backward_hip(
    Tensor grad_out, Tensor src, Tensor dst, Tensor offsets,
    c10::optional<Tensor> edge_scale, c10::optional<Tensor> node_scale,
    bool want_node_scale_grad);

std::tuple<Tensor, Tensor> segment_softmax_forward_hip(
    Tensor logit, Tensor env, Tensor offsets, c10::optional<Tensor> null_logit);

std::tuple<Tensor, Tensor, Tensor> segment_softmax_backward_hip(
    Tensor alpha, Tensor grad_alpha, Tensor env, Tensor offsets,
    c10::optional<Tensor> null_alpha, c10::optional<Tensor> grad_null_alpha);

std::tuple<Tensor, Tensor> force_assembly_hip(Tensor grad_edge,
                                             Tensor edge_vec,
                                             Tensor dst_offsets,
                                             Tensor src_offsets,
                                             Tensor src_order);
