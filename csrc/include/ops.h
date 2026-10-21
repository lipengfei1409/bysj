#pragma once
#include <torch/extension.h>
#include <vector>

using Tensor = torch::Tensor;
std::vector<Tensor> edge_geometry_hip(Tensor, Tensor, Tensor, Tensor, Tensor);
std::vector<Tensor> edge_geometry_backward_hip(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, int64_t);
std::vector<Tensor> radial_hip(Tensor, Tensor, double, int64_t, int64_t);
std::vector<Tensor> radial_backward_hip(Tensor, Tensor, Tensor, Tensor, double, int64_t, int64_t);
std::vector<Tensor> edge_features_hip(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, double, int64_t, int64_t, double);
std::vector<Tensor> edge_features_backward_hip(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, double, int64_t, int64_t, double);
std::vector<Tensor> edge_features_persistent_hip(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    double, int64_t, int64_t, double, std::vector<int64_t>, std::vector<int64_t>,
    std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
    int64_t, int64_t, int64_t, int64_t);
Tensor segment_sum_hip(Tensor, Tensor, int64_t);
std::vector<Tensor> force_virial_hip(Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, int64_t);
Tensor rmsnorm_hip(Tensor, Tensor, double);
std::vector<Tensor> rmsnorm_backward_hip(Tensor, Tensor, Tensor, double);
Tensor linear_hip(Tensor, Tensor, Tensor);
Tensor blas_gemm_hip(Tensor, Tensor, Tensor, bool, bool, bool, double, double);
