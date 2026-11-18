// Segmented force and per-atom virial assembly from per-edge energy gradients.
//
// The traced baseline assembles the two endpoints with row-atomic scatters and
// a materialized ``(E, 9)`` outer product:
//
//   fill_(force, 0); index_add_(force, dst, grad)     // atomics serialize
//   index_add_(force, src, -grad)
//   outer = -(grad[:, :, None] * edge_vec[:, None, :]).reshape(E, 9)
//   fill_(virial, 0); index_add_(virial, src, outer)  // materialized (E, 9)
//
// With ~60 colliding edges per atom the row atomics serialize: the profile
// shows ``index_add_`` at 4.2 ms per call for a 0.75 GB gather, i.e. ~177 GB/s
// of an achievable ~630 GB/s.
//
// Here one block owns five atoms and walks two CSR segment lists in
// double-precision registers -- the destination order (positive contribution)
// and the source order (negative contribution plus the virial outer product).
// No atomic, no materialized outer product, deterministic summation order.
//
// Layout contract: the natural edge order is grouped by destination, so
// ``dst_offsets`` addresses ``grad_edge`` directly; the source side needs the
// explicit permutation ``src_order`` whose segment ranges are ``src_offsets``.
//
// Mathematics, with ``g_e = dE / d(edge_vec_e)``:
//
//   F_k = sum_{dst(e)=k} g_e - sum_{src(e)=k} g_e
//   W_k = sum_{src(e)=k} (-g_e (x) edge_vec_e)
//
// ``F_k`` matches the canonical DeePMD force convention; ``W_k`` attributes the
// per-edge outer product in full to the source endpoint (atom-virial form).

#include "ops.h"
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {
constexpr unsigned int kLanesPerAtom = 12;  // 3 force + 9 virial components
// Five atoms fill one 64-lane wavefront (60 active lanes), so a launch owns
// five atoms instead of one and 3000 atoms cost 600 blocks.
constexpr unsigned int kAtomsPerBlock = 5;
constexpr unsigned int kBlockThreads = kAtomsPerBlock * kLanesPerAtom;

void check_offsets(const Tensor& offsets, const char* name) {
  TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt64 &&
                  offsets.is_contiguous(),
              name, " must be a contiguous DCU int64 tensor");
  TORCH_CHECK(offsets.dim() == 1 && offsets.size(0) >= 1,
              name, " must be a one-dimensional exclusive prefix sum");
}

void check_permutation(const Tensor& order, int64_t edges, const char* name) {
  TORCH_CHECK(order.is_cuda() && order.scalar_type() == torch::kInt64 &&
                  order.is_contiguous(),
              name, " must be a contiguous DCU int64 tensor");
  TORCH_CHECK(order.dim() == 1 && order.size(0) == edges,
              name, " must have shape (n_edges,)");
}

__global__ void force_assembly_kernel(
    const float* __restrict__ grad_edge, const float* __restrict__ edge_vec,
    const int64_t* __restrict__ src_order,
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ dst_offsets, float* __restrict__ force,
    float* __restrict__ virial, int64_t nodes) {
  const unsigned int lane = threadIdx.x % kLanesPerAtom;
  const int64_t atom = static_cast<int64_t>(blockIdx.x) * kAtomsPerBlock +
                       threadIdx.x / kLanesPerAtom;
  if (atom >= nodes) return;

  if (lane < 3) {
    const int component = static_cast<int>(lane);
    double accumulator = 0.0;
    for (int64_t edge = dst_offsets[atom]; edge < dst_offsets[atom + 1]; ++edge)
      accumulator += grad_edge[edge * 3 + component];
    for (int64_t slot = src_offsets[atom]; slot < src_offsets[atom + 1]; ++slot)
      accumulator -= grad_edge[src_order[slot] * 3 + component];
    force[atom * 3 + component] = static_cast<float>(accumulator);
    return;
  }

  const int component = static_cast<int>(lane) - 3;
  const int row = component / 3;
  const int column = component % 3;
  double accumulator = 0.0;
  for (int64_t slot = src_offsets[atom]; slot < src_offsets[atom + 1]; ++slot) {
    const int64_t edge = src_order[slot];
    accumulator -= static_cast<double>(grad_edge[edge * 3 + row]) *
                   edge_vec[edge * 3 + column];
  }
  virial[atom * 9 + component] = static_cast<float>(accumulator);
}

unsigned int grid_for(int64_t blocks, const char* name) {
  TORCH_CHECK(blocks <= static_cast<int64_t>(
                  std::numeric_limits<unsigned int>::max()),
              name, " exceeds the HIP grid limit");
  return static_cast<unsigned int>(blocks);
}
}  // namespace

std::tuple<Tensor, Tensor> force_assembly_hip(Tensor grad_edge, Tensor edge_vec,
                                             Tensor dst_offsets,
                                             Tensor src_offsets,
                                             Tensor src_order) {
  f32(grad_edge);
  f32(edge_vec);
  TORCH_CHECK(grad_edge.dim() == 2 && grad_edge.size(1) == 3,
              "force assembly grad_edge must have shape (n_edges, 3)");
  const int64_t edges = grad_edge.size(0);
  TORCH_CHECK(edge_vec.sizes() == grad_edge.sizes(),
              "force assembly edge_vec must match grad_edge");
  check_offsets(dst_offsets, "force assembly dst_offsets");
  check_offsets(src_offsets, "force assembly src_offsets");
  check_permutation(src_order, edges, "force assembly src_order");
  TORCH_CHECK(dst_offsets.size(0) == src_offsets.size(0),
              "force assembly offsets must describe the same atom count");
  const int64_t nodes = dst_offsets.size(0) - 1;
  TORCH_CHECK(nodes > 0, "force assembly requires at least one atom");
  same_device(grad_edge, {edge_vec, dst_offsets, src_offsets, src_order});

  Tensor force = torch::empty({nodes, 3}, grad_edge.options());
  Tensor virial = torch::empty({nodes, 9}, grad_edge.options());
  if (edges == 0) {
    force.zero_();
    virial.zero_();
    return std::make_tuple(force, virial);
  }

  c10::cuda::CUDAGuard guard(grad_edge.device());
  const int64_t blocks = (nodes + kAtomsPerBlock - 1) / kAtomsPerBlock;
  hipLaunchKernelGGL(force_assembly_kernel,
                     dim3(grid_for(blocks, "force assembly atom count")),
                     dim3(kBlockThreads), 0, at::cuda::getCurrentCUDAStream(),
                     grad_edge.data_ptr<float>(), edge_vec.data_ptr<float>(),
                     src_order.data_ptr<int64_t>(),
                     src_offsets.data_ptr<int64_t>(),
                     dst_offsets.data_ptr<int64_t>(), force.data_ptr<float>(),
                     virial.data_ptr<float>(), nodes);
  launch_check();
  return std::make_tuple(force, virial);
}
