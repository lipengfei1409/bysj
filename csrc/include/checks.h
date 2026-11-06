#pragma once
#include "ops.h"
#include <hip/hip_runtime.h>
#include <initializer_list>

inline void f32(const Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 &&
              tensor.is_contiguous(),
              "expected contiguous DCU float32 tensor");
}

inline void same_device(const Tensor& tensor,
                        std::initializer_list<Tensor> others) {
  for (const auto& other : others)
    TORCH_CHECK(other.device() == tensor.device(),
                "all tensors must share a device");
}

inline void launch_check() {
  const auto status = hipGetLastError();
  TORCH_CHECK(status == hipSuccess, "HIP kernel launch failed: ",
              hipGetErrorString(status));
}
