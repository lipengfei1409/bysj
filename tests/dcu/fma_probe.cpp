// FFMA throughput probe for the fused SO(2) convolution decision.
//
// The fused ``dpa4_so2_conv`` trades device traffic for float32 SIMT arithmetic:
// one mixing layer costs ``M0^2 + M1^2 + Cf*GATE`` multiply-adds per edge per
// focus stream (225.3 kFMA at the DPA4-Air shape).  Whether porting it to DCU is
// worth anything turns on the sustained ffma rate of that pattern, which is a
// device property, not a checkpoint property.  This kernel measures it with a
// pure register FMA chain -- no memory traffic, so the number is the arithmetic
// ceiling the port could reach.
//
// ``iters``/``blocks``/``threads``/``nacc`` are all runtime arguments so the
// Python side can sweep occupancy and ILP without a recompile.

#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

namespace {

template <int NACC>
__global__ void fma_kernel(float *out, int iters, float w, float c) {
  float a[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) {
    a[i] = 0.5f * static_cast<float>(i + 1) + 1e-7f * static_cast<float>(threadIdx.x);
  }
  for (int t = 0; t < iters; ++t) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) {
      a[i] = fmaf(a[i], w, c);
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < NACC; ++i) {
    sum += a[i];
  }
  // Compare against a runtime-unreachable value: keeps the chain alive without
  // paying a store on the measured path.
  if (sum == 12345.678f) {
    out[0] = sum;
  }
}

void launch(float *out, int blocks, int threads, int iters, int nacc) {
  const dim3 grid(blocks);
  const dim3 block(threads);
  const float w = 0.999999f;
  const float c = 1e-6f;
  switch (nacc) {
    case 1:
      fma_kernel<1><<<grid, block>>>(out, iters, w, c);
      break;
    case 4:
      fma_kernel<4><<<grid, block>>>(out, iters, w, c);
      break;
    case 8:
      fma_kernel<8><<<grid, block>>>(out, iters, w, c);
      break;
    case 16:
      fma_kernel<16><<<grid, block>>>(out, iters, w, c);
      break;
    default:
      TORCH_CHECK(false, "nacc must be one of 1, 4, 8, 16");
  }
  TORCH_CHECK(hipGetLastError() == hipSuccess, "fma_probe kernel launch failed");
}

void run(torch::Tensor out, int64_t blocks, int64_t threads, int64_t iters,
         int64_t nacc) {
  TORCH_CHECK(out.is_cuda(), "out must be a device tensor");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  launch(out.data_ptr<float>(), static_cast<int>(blocks),
         static_cast<int>(threads), static_cast<int>(iters),
         static_cast<int>(nacc));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "run an FFMA throughput probe");
}
