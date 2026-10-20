// CUDAExtension HIPifies this translation unit on ROCm/DTK.
#include "../include/ops.h"
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../include/checks.h"

__global__ void geometry_kernel(const float* pos, const int64_t* edge, const int64_t* shift,
    const float* cell, const int64_t* frame, int64_t ecount, float* vec, float* distance, float* unit) {
  int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= ecount) return;
  int64_t src = edge[e], dst = edge[ecount + e], f = frame[e];
  float d[3], norm2 = 0;
  for (int j=0; j<3; ++j) {
    d[j] = pos[3*src+j] - pos[3*dst+j];
    for (int k=0; k<3; ++k) d[j] += float(shift[3*e+k]) * cell[9*f+3*k+j];
    vec[3*e+j] = d[j]; norm2 += d[j]*d[j];
  }
  float r = sqrtf(norm2); distance[e] = r;
  for (int j=0; j<3; ++j) unit[3*e+j] = r > 0 ? d[j]/r : 0;
}

__global__ void geometry_backward_kernel(const float* gv, const float* gr, const float* gu,
    const float* r, const float* u, const int64_t* edge, const int64_t* shifts,
    const int64_t* frame, int64_t ecount, float* gp, float* gc) {
  int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(e>=ecount) return;
  float dot=0;
  for(int j=0;j<3;++j) dot+=gu[3*e+j]*u[3*e+j];
  for(int j=0;j<3;++j) {
    float g=gv[3*e+j]+gr[e]*u[3*e+j];
    if(r[e]>0) g+=(gu[3*e+j]-u[3*e+j]*dot)/r[e];
    atomicAdd(gp+3*edge[e]+j,g);
    atomicAdd(gp+3*edge[ecount+e]+j,-g);
    for(int k=0;k<3;++k) atomicAdd(gc+9*frame[e]+3*k+j,float(shifts[3*e+k])*g);
  }
}

std::vector<Tensor> edge_geometry_hip(Tensor pos, Tensor edge, Tensor shifts, Tensor cells, Tensor frame) {
  f32(pos); f32(cells); i64(edge); i64(shifts); i64(frame);
  same_device(pos,{edge,shifts,cells,frame});
  TORCH_CHECK(pos.dim()==2 && pos.size(1)==3,"positions must be [N,3]");
  TORCH_CHECK(edge.dim()==2 && edge.size(0)==2,"edge_index must be [2,E]");
  auto e=edge.size(1);
  TORCH_CHECK(shifts.dim()==2 && shifts.size(0)==e && shifts.size(1)==3 && frame.dim()==1 && frame.numel()==e,"edge shapes disagree");
  TORCH_CHECK(cells.dim()==3 && cells.size(0)>0 && cells.size(1)==3 && cells.size(2)==3,"cells must be [F,3,3], F>0");
  c10::cuda::CUDAGuard guard(pos.device());
  auto vec=torch::empty({e,3},pos.options()), r=torch::empty({e,1},pos.options()), unit=torch::empty({e,3},pos.options());
  if(e) {
    hipLaunchKernelGGL(geometry_kernel,dim3(blocks(e)),dim3(256),0,at::cuda::getCurrentCUDAStream(),
      pos.data_ptr<float>(),edge.data_ptr<int64_t>(),shifts.data_ptr<int64_t>(),cells.data_ptr<float>(),frame.data_ptr<int64_t>(),e,
      vec.data_ptr<float>(),r.data_ptr<float>(),unit.data_ptr<float>());
    launch_check();
  }
  return {vec,r,unit};
}

std::vector<Tensor> edge_geometry_backward_hip(Tensor gv, Tensor gr, Tensor gu, Tensor r, Tensor u,
    Tensor edge, Tensor shifts, Tensor frame, int64_t n, int64_t f) {
  f32(gv); f32(gr); f32(gu); f32(r); f32(u); i64(edge); i64(shifts); i64(frame);
  same_device(gv,{gr,gu,r,u,edge,shifts,frame});
  auto e=r.numel();
  TORCH_CHECK(n>=0 && f>0 && gv.numel()==3*e && gu.numel()==3*e && u.numel()==3*e && gr.numel()==e && edge.numel()==2*e && shifts.numel()==3*e && frame.numel()==e,"backward shapes disagree");
  c10::cuda::CUDAGuard guard(gv.device());
  auto gp=torch::zeros({n,3},gv.options()), gc=torch::zeros({f,3,3},gv.options());
  if(e) {
    hipLaunchKernelGGL(geometry_backward_kernel,dim3(blocks(e)),dim3(256),0,at::cuda::getCurrentCUDAStream(),
      gv.data_ptr<float>(),gr.data_ptr<float>(),gu.data_ptr<float>(),r.data_ptr<float>(),u.data_ptr<float>(),edge.data_ptr<int64_t>(),
      shifts.data_ptr<int64_t>(),frame.data_ptr<int64_t>(),e,gp.data_ptr<float>(),gc.data_ptr<float>());
    launch_check();
  }
  return {gp,gc};
}
