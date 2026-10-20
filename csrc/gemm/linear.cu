// Correctness baseline. Replace with hipBLASLt or MFMA tiles after profiling.
#include "../include/ops.h"
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../include/checks.h"

__global__ void linear_kernel(const float* x,const float* w,const float* bias,int64_t rows,int64_t kin,int64_t kout,float* y) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=rows*kout)return;
  int64_t row=i/kout,col=i%kout;float value=bias?bias[col]:0;
  for(int64_t k=0;k<kin;++k)value+=x[row*kin+k]*w[k*kout+col];y[i]=value;
}

Tensor linear_hip(Tensor x,Tensor w,Tensor bias) {
  f32(x);f32(w);f32(bias);same_device(x,{w,bias});
  TORCH_CHECK(x.dim()==2 && w.dim()==2 && x.size(1)==w.size(0) && bias.dim()==1 && (bias.numel()==0 || bias.numel()==w.size(1)),"linear requires x [N,K], weight [K,C], optional bias [C]");
  c10::cuda::CUDAGuard guard(x.device());auto y=torch::empty({x.size(0),w.size(1)},x.options());
  if(y.numel()){hipLaunchKernelGGL(linear_kernel,dim3(blocks(y.numel())),dim3(256),0,at::cuda::getCurrentCUDAStream(),x.data_ptr<float>(),w.data_ptr<float>(),bias.numel()?bias.data_ptr<float>():nullptr,x.size(0),x.size(1),w.size(1),y.data_ptr<float>());launch_check();}
  return y;
}
