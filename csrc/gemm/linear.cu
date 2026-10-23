// Correctness baseline. Replace with hipBLASLt or MFMA tiles after profiling.
#include "../include/ops.h"
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../include/checks.h"
#include "ops/ops.cuh"
#include <limits>

__global__ void linear_kernel(const float* x,const float* w,const float* bias,int64_t rows,int64_t kin,int64_t kout,float* y) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=rows*kout)return;
  int64_t row=i/kout,col=i%kout;float value=bias?bias[col]:0;
  for(int64_t k=0;k<kin;++k)value+=x[row*kin+k]*w[k*kout+col];y[i]=value;
}

#if DPA4_ENABLE_DUMMA
// One complete wave owns a 16x16 output tile. Loads are cooperative and
// synchronous; each K slice is staged in LDS, consumed by DUMMA, then reused.
template<typename Op>
__global__ void linear_dumma_kernel(const float* x, const float* w,
                                    const float* bias, int64_t rows,
                                    int64_t kin, int64_t kout, float* y) {
  using shape = dcu::gemm_tile_shape<Op::m, Op::n, Op::k>;
  static_assert(Op::wave_threads == 64, "DUMMA linear requires one wave64 CTA");
  __shared__ dcu::shared_tile<float, shape::m, shape::k> lhs;
  __shared__ dcu::shared_tile<float, shape::k, shape::n> rhs;
  __shared__ dcu::shared_tile<float, shape::m, shape::n> output;

  const int64_t row0 = int64_t(blockIdx.y) * shape::m;
  const int64_t col0 = int64_t(blockIdx.x) * shape::n;
  const dcu::matrix_view<const float> x_view{x, rows, kin, kin};
  const dcu::matrix_view<const float> w_view{w, kin, kout, kout};
  typename Op::a_fragment a;
  typename Op::b_fragment b;
  typename Op::accumulator accum0, accum1;
  typename Op::accumulator* current = &accum0;
  typename Op::accumulator* next = &accum1;
  Op::fill(*current, 0.f);

  for (int64_t k0 = 0; k0 < kin; k0 += shape::k) {
    // Producer phase: masked tail loads zero-pad the physical DUMMA tile.
    dcu::group::load(lhs, x_view, row0, k0);
    dcu::group::load(rhs, w_view, k0, col0);
    dcu::group::block_sync();

    // Consumer phase: all lanes execute the same opaque DUMMA fragment calls.
    Op::load_a(a, lhs.data, shape::k);
    Op::load_b(b, rhs.data, shape::n);
    dcu::group::block_sync(); // LDS may be reused only after collective loads finish.
    Op::mma(*next, a, b, *current);
    dcu::group::block_sync(); // No async MMA is assumed by this pipeline.
    auto* previous = current;
    current = next;
    next = previous;
  }

  // Storer phase uses an LDS tile so incomplete M/N tiles remain masked.
  Op::store(output.data, *current, shape::n);
  dcu::group::block_sync();
  for (unsigned i = dcu::group::thread_rank(); i < shape::m * shape::n;
       i += dcu::group::block_threads()) {
    const int64_t row = row0 + i / shape::n;
    const int64_t col = col0 + i % shape::n;
    if (row < rows && col < kout) {
      float value = output.data[i];
      if (bias) value += bias[col];
      y[row * kout + col] = value;
    }
  }
}
#endif

Tensor linear_hip(Tensor x,Tensor w,Tensor bias) {
  f32(x);f32(w);f32(bias);same_device(x,{w,bias});
  TORCH_CHECK(x.dim()==2 && w.dim()==2 && x.size(1)==w.size(0) && bias.dim()==1 && (bias.numel()==0 || bias.numel()==w.size(1)),"linear requires x [N,K], weight [K,C], optional bias [C]");
  c10::cuda::CUDAGuard guard(x.device());auto y=torch::empty({x.size(0),w.size(1)},x.options());
  if(y.numel()){
#if DPA4_ENABLE_DUMMA
    constexpr int tile_m = dcu::group::mma_f32_16x16x4::m;
    constexpr int tile_n = dcu::group::mma_f32_16x16x4::n;
    const int64_t grid_m = (x.size(0) + tile_m - 1) / tile_m;
    const int64_t grid_n = (w.size(1) + tile_n - 1) / tile_n;
    TORCH_CHECK(grid_m <= std::numeric_limits<unsigned>::max() &&
                grid_n <= std::numeric_limits<unsigned>::max(),
                "linear DUMMA tile grid is too large");
    hipLaunchKernelGGL((linear_dumma_kernel<dcu::group::mma_f32_16x16x4>),
        dim3(static_cast<unsigned>(grid_n), static_cast<unsigned>(grid_m)),
        dim3(dcu::group::mma_f32_16x16x4::wave_threads), 0,
        at::cuda::getCurrentCUDAStream(), x.data_ptr<float>(), w.data_ptr<float>(),
        bias.numel()?bias.data_ptr<float>():nullptr, x.size(0), x.size(1),
        w.size(1), y.data_ptr<float>());
#else
    hipLaunchKernelGGL(linear_kernel,dim3(blocks(y.numel())),dim3(256),0,at::cuda::getCurrentCUDAStream(),x.data_ptr<float>(),w.data_ptr<float>(),bias.numel()?bias.data_ptr<float>():nullptr,x.size(0),x.size(1),w.size(1),y.data_ptr<float>());
#endif
    launch_check();
  }
  return y;
}
