#include "../include/ops.h"
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../include/checks.h"

__device__ void envelope(float r, float cutoff, int p, float& value, float& derivative) {
  if(p==0) {value=1;derivative=0;return;}
  if(r>=cutoff) {value=derivative=0;return;}
  float x=fmaxf(0,r/cutoff),u=1-x;
  float sum=0,ds=0,power=1,previous=1;
  for(int k=0;k<p;++k) {
    float coefficient=float((k+1)*(k+2)*(k+3))/6.f;
    sum+=coefficient*power;
    if(k) ds+=float(k)*coefficient*previous;
    previous=power;power*=x;
  }
  value=u*u*u*u*sum;
  // Match torch.clamp's inclusive derivative at r=0 (p=1 is nonzero there).
  derivative=r<0?0:(-4*u*u*u*sum+u*u*u*u*ds)/cutoff;
}

__device__ void sinc_pair(float z,float& s,float& ds) {
  if(fabsf(z)<1e-3f) {
    float z2=z*z;s=1-z2/6+z2*z2/120;ds=-z/3+z*z2/30;
  } else {s=sinf(z)/z;ds=(z*cosf(z)-sinf(z))/(z*z);}
}

__global__ void radial_kernel(const float* r,const float* w,int64_t ecount,int64_t nr,
    float cutoff,int bp,int ep,float* rbf,float* env) {
  int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(e>=ecount)return;
  float bv,bd,ev,ed; envelope(r[e],cutoff,bp,bv,bd);envelope(r[e],cutoff,ep,ev,ed);env[e]=ev;
  for(int64_t j=0;j<nr;++j) {float s,ds;sinc_pair(r[e]*w[j],s,ds);rbf[e*nr+j]=w[j]*s*bv;}
}

__global__ void radial_backward_kernel(const float* g,const float* ge,const float* r,const float* w,
    int64_t ecount,int64_t nr,float cutoff,int bp,int ep,float* gr,float* gw) {
  int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(e>=ecount)return;
  float bv,bd,ev,ed;envelope(r[e],cutoff,bp,bv,bd);envelope(r[e],cutoff,ep,ev,ed);
  float dr=ge[e]*ed;
  for(int64_t j=0;j<nr;++j) {
    float s,ds,z=r[e]*w[j];sinc_pair(z,s,ds);
    float go=g[e*nr+j]; dr+=go*(w[j]*w[j]*ds*bv+w[j]*s*bd);
    atomicAdd(gw+j,go*(s+z*ds)*bv);
  }
  gr[e]=dr;
}

std::vector<Tensor> radial_hip(Tensor r,Tensor w,double cutoff,int64_t bp,int64_t ep) {
  f32(r);f32(w);same_device(r,{w});
  TORCH_CHECK(r.dim()==2 && r.size(1)==1 && w.dim()==2 && w.size(0)==1,"r must be [E,1], w [1,R]");
  TORCH_CHECK(cutoff>0 && bp>=0 && ep>=0 && bp<=64 && ep<=64 && w.numel()>0,"invalid radial parameters");
  c10::cuda::CUDAGuard guard(r.device());auto e=r.size(0),nr=w.numel();
  auto rbf=torch::empty({e,nr},r.options()), env=torch::empty_like(r);
  if(e) {
    hipLaunchKernelGGL(radial_kernel,dim3(blocks(e)),dim3(256),0,at::cuda::getCurrentCUDAStream(),
      r.data_ptr<float>(),w.data_ptr<float>(),e,nr,float(cutoff),int(bp),int(ep),rbf.data_ptr<float>(),env.data_ptr<float>());
    launch_check();
  }
  return {rbf,env};
}

std::vector<Tensor> radial_backward_hip(Tensor g,Tensor ge,Tensor r,Tensor w,double cutoff,int64_t bp,int64_t ep) {
  f32(g);f32(ge);f32(r);f32(w);same_device(r,{g,ge,w});
  TORCH_CHECK(r.dim()==2 && r.size(1)==1 && w.dim()==2 && w.size(0)==1 && g.dim()==2 && g.size(0)==r.size(0) && g.size(1)==w.numel() && ge.sizes()==r.sizes(),"radial backward shapes disagree");
  TORCH_CHECK(cutoff>0 && bp>=0 && ep>=0 && bp<=64 && ep<=64,"invalid radial parameters");
  c10::cuda::CUDAGuard guard(r.device());auto e=r.size(0),nr=w.numel();
  auto gr=torch::empty_like(r),gw=torch::zeros_like(w);
  if(e) {
    hipLaunchKernelGGL(radial_backward_kernel,dim3(blocks(e)),dim3(256),0,at::cuda::getCurrentCUDAStream(),
      g.data_ptr<float>(),ge.data_ptr<float>(),r.data_ptr<float>(),w.data_ptr<float>(),e,nr,float(cutoff),int(bp),int(ep),gr.data_ptr<float>(),gw.data_ptr<float>());
    launch_check();
  }
  return {gr,gw};
}
