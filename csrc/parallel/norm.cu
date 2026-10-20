#include "../include/ops.h"
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../include/checks.h"

__global__ void norm_kernel(const float* x,const float* w,int64_t rows,int64_t width,float eps,float* y) {
  int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(row>=rows)return;
  float sum=0;for(int64_t j=0;j<width;++j)sum+=x[row*width+j]*x[row*width+j];
  float inv=rsqrtf(sum/float(width)+eps);
  for(int64_t j=0;j<width;++j)y[row*width+j]=x[row*width+j]*inv*w[j];
}

__global__ void norm_backward_kernel(const float* g,const float* x,const float* w,int64_t rows,int64_t width,float eps,float* gx,float* gw) {
  int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=rows)return;
  float sum=0,dot=0;for(int64_t j=0;j<width;++j){float v=x[row*width+j];sum+=v*v;dot+=g[row*width+j]*w[j]*v;}
  float inv=rsqrtf(sum/float(width)+eps), correction=inv*inv*inv*dot/float(width);
  for(int64_t j=0;j<width;++j){int64_t i=row*width+j;gx[i]=inv*g[i]*w[j]-x[i]*correction;atomicAdd(gw+j,g[i]*x[i]*inv);}
}

Tensor rmsnorm_hip(Tensor x,Tensor w,double eps) {
  f32(x);f32(w);same_device(x,{w});
  TORCH_CHECK(x.dim()==2 && w.dim()==1 && w.numel()==x.size(1) && x.size(1)>0 && eps>0,"RMSNorm requires x [N,C], scale [C], eps>0");
  c10::cuda::CUDAGuard guard(x.device());auto y=torch::empty_like(x);
  if(x.size(0)){hipLaunchKernelGGL(norm_kernel,dim3(blocks(x.size(0))),dim3(256),0,at::cuda::getCurrentCUDAStream(),x.data_ptr<float>(),w.data_ptr<float>(),x.size(0),x.size(1),float(eps),y.data_ptr<float>());launch_check();}
  return y;
}

std::vector<Tensor> rmsnorm_backward_hip(Tensor g,Tensor x,Tensor w,double eps) {
  f32(g);f32(x);f32(w);same_device(x,{g,w});
  TORCH_CHECK(g.sizes()==x.sizes() && x.dim()==2 && w.dim()==1 && w.numel()==x.size(1) && x.size(1)>0 && eps>0,"RMSNorm backward shapes disagree");
  c10::cuda::CUDAGuard guard(x.device());auto gx=torch::empty_like(x),gw=torch::zeros_like(w);
  if(x.size(0)){hipLaunchKernelGGL(norm_backward_kernel,dim3(blocks(x.size(0))),dim3(256),0,at::cuda::getCurrentCUDAStream(),g.data_ptr<float>(),x.data_ptr<float>(),w.data_ptr<float>(),x.size(0),x.size(1),float(eps),gx.data_ptr<float>(),gw.data_ptr<float>());launch_check();}
  return {gx,gw};
}
