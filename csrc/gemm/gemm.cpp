#include "../include/ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include "../include/checks.h"
#include "ops/group/memory.h"
#include "ops/group/mma.h"
#include "ops/group/sync.h"
#include <limits>

#if !DPA4_ENABLE_DUMMA
#error "The only native operator is DUMMA GEMM; build with DPA4_ENABLE_DUMMA=1"
#endif

namespace {
template<typename Op>
__global__ void gemm_dumma_kernel(const float* a, const float* b, float* c,
                                  int64_t m, int64_t n, int64_t k) {
  using shape = dmk::gemm_tile_shape<Op::m, Op::n, Op::k>;
  static_assert(Op::wave_threads == 64, "DUMMA GEMM requires one complete wave64 CTA");
  __shared__ dmk::shared_tile<float, shape::m, shape::k> lhs;
  __shared__ dmk::shared_tile<float, shape::k, shape::n> rhs;
  __shared__ dmk::shared_tile<float, shape::m, shape::n> output;

  const int64_t row0 = int64_t(blockIdx.y) * shape::m;
  const int64_t col0 = int64_t(blockIdx.x) * shape::n;
  const dmk::matrix_view<const float> a_view{a, m, k, k};
  const dmk::matrix_view<const float> b_view{b, k, n, n};
  typename Op::a_fragment a_fragment;
  typename Op::b_fragment b_fragment;
  typename Op::accumulator accum0, accum1;
  typename Op::accumulator* current = &accum0;
  typename Op::accumulator* next = &accum1;
  Op::fill(*current, 0.f);

  for (int64_t k0 = 0; k0 < k; k0 += shape::k) {
    dmk::group::load(lhs, a_view, row0, k0);
    dmk::group::load(rhs, b_view, k0, col0);
    dmk::group::block_sync();

    Op::load_a(a_fragment, lhs.data, shape::k);
    Op::load_b(b_fragment, rhs.data, shape::n);
    dmk::group::block_sync();
    Op::mma(*next, a_fragment, b_fragment, *current);
    dmk::group::block_sync();
    auto* previous = current;
    current = next;
    next = previous;
  }

  Op::store(output.data, *current, shape::n);
  dmk::group::block_sync();
  for (unsigned i = dmk::group::thread_rank(); i < shape::m * shape::n;
       i += dmk::group::block_threads()) {
    const int64_t row = row0 + i / shape::n;
    const int64_t col = col0 + i % shape::n;
    if (row < m && col < n) c[row * n + col] = output.data[i];
  }
}
} // namespace

Tensor gemm_dumma_hip(Tensor a, Tensor b) {
  f32(a);
  f32(b);
  same_device(a, {b});
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
              "DUMMA GEMM requires two 2D tensors A[M,K] and B[K,N]");
  TORCH_CHECK(a.size(1) == b.size(0), "DUMMA GEMM inner dimensions must match");

  const int64_t m = a.size(0), k = a.size(1), n = b.size(1);
  Tensor c = torch::empty({m, n}, a.options());
  if (m != 0 && n != 0) {
    constexpr int tile_m = dmk::group::mma_f32_16x16x4::m;
    constexpr int tile_n = dmk::group::mma_f32_16x16x4::n;
    const int64_t grid_m = m / tile_m + (m % tile_m != 0);
    const int64_t grid_n = n / tile_n + (n % tile_n != 0);
    TORCH_CHECK(grid_m <= std::numeric_limits<unsigned>::max() &&
                grid_n <= std::numeric_limits<unsigned>::max(),
                "DUMMA GEMM tile grid is too large");
    c10::cuda::CUDAGuard guard(a.device());
    hipLaunchKernelGGL((gemm_dumma_kernel<dmk::group::mma_f32_16x16x4>),
        dim3(static_cast<unsigned>(grid_n), static_cast<unsigned>(grid_m)),
        dim3(dmk::group::mma_f32_16x16x4::wave_threads), 0,
        at::cuda::getCurrentCUDAStream(), a.data_ptr<float>(), b.data_ptr<float>(),
        c.data_ptr<float>(), m, n, k);
    launch_check();
  }
  return c;
}
