// C7-a: fused quaternion monomial basis and its contraction with the run table.
//
// The dense reference builds the Wigner run matrix in four passes:
//
//   pow  ->  (4, P+1, E) power table          (2.9 MB at 3000 atoms, lmax=3)
//   gather/prod -> (E, M) monomial design      (three (4, P+1, E) intermediates)
//   bmm  ->  runs = M @ coeff^T                (E, NW)
//
// and its backward differentiates the product tree, which lowers to a `cumprod`
// scan pair. Fused here:
//
//   runs[e, nw] = sum_m M[e, m] * coeff[nw, m]
//   M[e, m]     = prod_i q[e, i] ^ exp[m, i]        (exponents from a table)
//
//   g_quat[e, i] = sum_m' dM[e, m'] * sum_nw g_runs[e, nw] * dcoeff[nw, i, m']
//
// No power table, no design matrix, no scan: the exponents are a table the
// caller owns, so a different degree is a different table rather than a
// different kernel, and the monomials are evaluated in registers.
//
// One thread per (edge, nw) forward and per (edge, i) backward. The tables are
// small enough to stay in L2 (coeff NW x M, dcoeff NW x 4 x M'), so the kernels
// are limited by the runs tensor, not by recompute.

#include "ops.h"
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <tuple>

namespace {
constexpr unsigned int kThreads = 256;

__device__ __forceinline__ float monomial(const float* q, const signed char* exps) {
  float value = 1.0f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int e = exps[c];
    const float base = q[c];
    for (int p = 0; p < e; ++p) {
      value *= base;
    }
  }
  return value;
}

__global__ void runs_kernel(const float* __restrict__ q,
                            const signed char* __restrict__ exps,
                            const float* __restrict__ coeff,
                            float* __restrict__ runs,
                            long edges, int n_mono, int n_runs) {
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= edges * n_runs) {
    return;
  }
  const long edge = idx / n_runs;
  const int nw = static_cast<int>(idx - edge * n_runs);
  const float* qe = q + edge * 4;
  const float* row = coeff + static_cast<long>(nw) * n_mono;
  float acc = 0.0f;
  for (int m = 0; m < n_mono; ++m) {
    acc = fmaf(monomial(qe, exps + m * 4), row[m], acc);
  }
  runs[idx] = acc;
}

__global__ void quat_grad_kernel(const float* __restrict__ g_runs,
                                 const float* __restrict__ q,
                                 const signed char* __restrict__ dexp,
                                 const float* __restrict__ dcoeff,
                                 float* __restrict__ g_quat,
                                 long edges, int n_mono_d, int n_runs) {
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= edges * 4) {
    return;
  }
  const long edge = idx / 4;
  const int i = static_cast<int>(idx - edge * 4);
  const float* qe = q + edge * 4;
  const float* gr = g_runs + edge * n_runs;

  // partial[m'] = sum_nw g_runs[nw] * dcoeff[nw, i, m']
  // computed one m' at a time so the run cotangent is read once per thread.
  float acc = 0.0f;
  for (int m = 0; m < n_mono_d; ++m) {
    // Strided access: dcoeff is (n_runs, 4, n_mono_d), so the runs are the
    // outer axis and the m-contiguous read is per (nw, i).
    const float* col = dcoeff + (static_cast<long>(i) * n_mono_d) + m;
    float partial = 0.0f;
    for (int nw = 0; nw < n_runs; ++nw) {
      partial = fmaf(gr[nw], col[static_cast<long>(nw) * 4 * n_mono_d], partial);
    }
    acc = fmaf(partial, monomial(qe, dexp + m * 4), acc);
  }
  g_quat[idx] = acc;
}
}  // namespace

Tensor monomial_runs_hip(Tensor q, Tensor exps, Tensor coeff) {
  f32(q);
  f32(coeff);
  TORCH_CHECK(exps.scalar_type() == torch::kChar && exps.dim() == 2 &&
                  exps.size(1) == 4,
              "monomial exponents must be an int8 (M, 4) table");
  TORCH_CHECK(q.dim() == 2 && q.size(1) == 4, "quaternions must be (E, 4)");
  TORCH_CHECK(coeff.dim() == 2 && coeff.size(1) == exps.size(0),
              "the run coefficient table must be (NW, M)");
  TORCH_CHECK(exps.is_contiguous() && coeff.is_contiguous(),
              "the monomial tables must be contiguous");
  same_device(q, {coeff});
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const long edges = q.size(0);
  const int n_mono = static_cast<int>(exps.size(0));
  const int n_runs = static_cast<int>(coeff.size(0));
  auto runs = torch::empty({edges, n_runs}, q.options());
  const long total = edges * n_runs;
  if (total > 0) {
    const unsigned blocks = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    runs_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), exps.data_ptr<signed char>(), coeff.data_ptr<float>(),
        runs.data_ptr<float>(), edges, n_mono, n_runs);
    launch_check();
  }
  return runs;
}

Tensor monomial_runs_backward_hip(Tensor grad_runs, Tensor q, Tensor dexp,
                                  Tensor dcoeff) {
  f32(grad_runs);
  f32(q);
  f32(dcoeff);
  TORCH_CHECK(dexp.scalar_type() == torch::kChar && dexp.dim() == 2 &&
                  dexp.size(1) == 4,
              "derivative exponents must be an int8 (M', 4) table");
  TORCH_CHECK(dcoeff.dim() == 3 && dcoeff.size(0) == grad_runs.size(1) &&
                  dcoeff.size(1) == 4 && dcoeff.size(2) == dexp.size(0),
              "the derivative coefficients must be (NW, 4, M')");
  TORCH_CHECK(grad_runs.size(0) == q.size(0), "one run cotangent row per edge");
  TORCH_CHECK(dexp.is_contiguous() && dcoeff.is_contiguous(),
              "the derivative tables must be contiguous");
  same_device(q, {grad_runs, dcoeff});
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const long edges = q.size(0);
  const int n_runs = static_cast<int>(grad_runs.size(1));
  const int n_mono_d = static_cast<int>(dexp.size(0));
  auto g_quat = torch::empty_like(q);
  const long total = edges * 4;
  if (total > 0) {
    const unsigned blocks = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    quat_grad_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_runs.data_ptr<float>(), q.data_ptr<float>(),
        dexp.data_ptr<signed char>(), dcoeff.data_ptr<float>(),
        g_quat.data_ptr<float>(), edges, n_mono_d, n_runs);
    launch_check();
  }
  return g_quat;
}
