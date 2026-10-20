#include "../include/ops.h"
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../include/checks.h"

__global__ void segment_kernel(const float* x,const int64_t* idx,int64_t rows,int64_t width,float* out) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<rows*width) atomicAdd(out+idx[i/width]*width+i%width,x[i]);
}

__global__ void force_kernel(const float* g,const float* d,const int64_t* edge,const int64_t* frame,
    const bool* mask,int64_t ecount,float* force,float* virial) {
  int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(e>=ecount || !mask[e])return;
  for(int k=0;k<3;++k) {
    atomicAdd(force+3*edge[e]+k,-g[3*e+k]);
    atomicAdd(force+3*edge[ecount+e]+k,g[3*e+k]);
    for(int j=0;j<3;++j) atomicAdd(virial+9*frame[e]+3*k+j,-g[3*e+k]*d[3*e+j]);
  }
}

Tensor segment_sum_hip(Tensor x,Tensor idx,int64_t n) {
  f32(x);i64(idx);same_device(x,{idx});
  TORCH_CHECK(x.dim()==2 && idx.dim()==1 && idx.size(0)==x.size(0) && n>=0,"segment_sum requires x [E,K], index [E], size>=0");
  c10::cuda::CUDAGuard guard(x.device());auto out=torch::zeros({n,x.size(1)},x.options());
  if(x.numel()) {
    hipLaunchKernelGGL(segment_kernel,dim3(blocks(x.numel())),dim3(256),0,at::cuda::getCurrentCUDAStream(),
      x.data_ptr<float>(),idx.data_ptr<int64_t>(),x.size(0),x.size(1),out.data_ptr<float>());
    launch_check();
  }
  return out;
}

std::vector<Tensor> force_virial_hip(Tensor g,Tensor d,Tensor edge,Tensor frame,Tensor mask,int64_t n,int64_t f) {
  f32(g);f32(d);i64(edge);i64(frame);same_device(g,{d,edge,frame,mask});
  auto e=g.size(0);
  TORCH_CHECK(g.dim()==2 && g.size(1)==3 && d.sizes()==g.sizes() && edge.dim()==2 && edge.size(0)==2 && edge.size(1)==e && frame.dim()==1 && frame.numel()==e && mask.dim()==1 && mask.numel()==e && mask.scalar_type()==torch::kBool && mask.is_contiguous() && n>=0 && f>0,"force/virial shapes disagree");
  c10::cuda::CUDAGuard guard(g.device());auto force=torch::zeros({n,3},g.options()),virial=torch::zeros({f,3,3},g.options());
  if(e) {
    hipLaunchKernelGGL(force_kernel,dim3(blocks(e)),dim3(256),0,at::cuda::getCurrentCUDAStream(),
      g.data_ptr<float>(),d.data_ptr<float>(),edge.data_ptr<int64_t>(),frame.data_ptr<int64_t>(),mask.data_ptr<bool>(),e,
      force.data_ptr<float>(),virial.data_ptr<float>());
    launch_check();
  }
  return {force,virial};
}
