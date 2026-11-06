#pragma once
#include <torch/extension.h>

using Tensor = torch::Tensor;
Tensor gemm_dumma_hip(Tensor a, Tensor b);
