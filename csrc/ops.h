#pragma once
#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <tuple>
#include <vector>

using Tensor = torch::Tensor;

std::tuple<Tensor, Tensor> rmsnorm_forward_hip(Tensor x, Tensor scale, double eps);

std::tuple<Tensor, Tensor> rmsnorm_backward_hip(Tensor grad_output, Tensor x, Tensor scale, Tensor inverse_rms, bool compute_scale_grad);

Tensor gie_zonal_scatter_forward_hip(Tensor zonal, Tensor radial, Tensor dst, Tensor radial_slot, Tensor inv_sqrt_deg);

std::tuple<Tensor, Tensor, Tensor> gie_zonal_scatter_backward_hip(Tensor grad_output, Tensor zonal, Tensor radial, Tensor dst, Tensor radial_slot, Tensor inv_sqrt_deg);

// --- Fused segmented operators (see docs/task2-hip-fusion-plan.md) ---------
//
// All three take a destination-grouped edge list: ``dst`` non-decreasing and
// ``offsets`` its exclusive prefix sum. The forward kernels never zero-init
// their output and never use atomics.

Tensor segment_sum_forward_hip(Tensor src, Tensor offsets,
                               c10::optional<Tensor> edge_scale,
                               c10::optional<Tensor> node_scale,
                               c10::optional<Tensor> order);

std::tuple<Tensor, Tensor> segment_sum_backward_hip(
    Tensor grad_out, Tensor src, Tensor dst, Tensor offsets,
    c10::optional<Tensor> edge_scale, c10::optional<Tensor> node_scale,
    bool want_node_scale_grad, c10::optional<Tensor> order);

std::tuple<Tensor, Tensor> segment_softmax_forward_hip(
    Tensor logit, Tensor env, Tensor offsets, c10::optional<Tensor> null_logit,
    c10::optional<Tensor> order);

std::tuple<Tensor, Tensor, Tensor> segment_softmax_backward_hip(
    Tensor alpha, Tensor grad_alpha, Tensor env, Tensor offsets,
    c10::optional<Tensor> null_alpha, c10::optional<Tensor> grad_null_alpha,
    c10::optional<Tensor> order);

std::tuple<Tensor, Tensor> force_assembly_hip(Tensor grad_edge,
                                             Tensor edge_vec,
                                             Tensor dst_offsets,
                                             Tensor src_offsets,
                                             Tensor src_order,
                                             c10::optional<Tensor> dst_order);

// --- Fused edge-stream linear combination (see docs/optimization-status.md, C6) --
//
// out[e, r, c] = sum_t coef_t * term_t[e, ...] in one pass; each term
// broadcasts over the channel axis, the row axis, both, or is a scalar.
Tensor edge_combine_hip(std::vector<Tensor> factors,
                        std::vector<int64_t> modes,
                        std::vector<double> coefficients,
                        std::vector<int64_t> term_starts,
                        std::vector<int64_t> term_counts, int64_t edges,
                        int64_t rows, int64_t channels);

// --- Slot-axis operators (see docs/optimization-status.md, C1/C2) ----------

Tensor slot_pack_scale_hip(Tensor src, Tensor slot, c10::optional<Tensor> scale);

Tensor slot_reduce_hip(Tensor src, c10::optional<Tensor> weight);

// --- C7-a/C7-b: monomial runs and the block-diagonal mixing GEMM ----------

Tensor monomial_runs_hip(Tensor q, Tensor exps, Tensor coeff);

Tensor monomial_runs_backward_hip(Tensor grad_runs, Tensor q, Tensor dexp,
                                  Tensor dcoeff);

Tensor blockdiag_gemm_hip(Tensor x, Tensor weights, Tensor offsets);

Tensor blockdiag_gemm_backward_hip(Tensor grad_out, Tensor weights, Tensor offsets);

// --- C8: the SO(2) convolution, ordinary entry point ------------------------
//
// `csrc/so2_conv_dmk.cpp`. The persistent op that mirrors it
// (`ops/so2_conv_op.h`) is source-only for now, so this is the only way to
// reach ops/so2_conv_math.h from Python.
Tensor so2_conv_dmk_hip(Tensor x, Tensor src, Tensor rot, Tensor rot_t,
                        Tensor kc, Tensor alpha);
