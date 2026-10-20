#pragma once
#include <hip/hip_runtime.h>
#include <initializer_list>
#include <limits>

inline void f32(const Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type()==torch::kFloat32 && t.is_contiguous(),"expected contiguous DCU float32 tensor");
}
inline void i64(const Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type()==torch::kInt64 && t.is_contiguous(),"expected contiguous DCU int64 tensor");
}
inline void same_device(const Tensor& t, std::initializer_list<Tensor> others) {
  for(const auto& x:others) TORCH_CHECK(x.device()==t.device(),"all tensors must share a device");
}
inline unsigned blocks(int64_t count) {
  TORCH_CHECK(count>=0 && (count+255)/256<=std::numeric_limits<unsigned>::max(),"tensor too large for launch grid");
  return unsigned((count+255)/256);
}
inline void launch_check() {
  auto status=hipGetLastError();
  TORCH_CHECK(status==hipSuccess,"HIP kernel launch failed: ",hipGetErrorString(status));
}
