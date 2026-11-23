// C6: fused edge-stream linear combinations.
//
// The profiled forward/backward spends 488 ms on E-major elementwise chains of
// the shape ``[1, E, R, C]``: a residual add, a broadcast multiply, a copy and
// a zero fill, each a separate pass over the same bytes. Every pass already runs
// at the memory roofline, so the only saving is the number of passes.
//
//   edge_combine   out[e, r, c] = sum_t coef_t * term_t[e, ...]
//
// One pass, one kernel, one output write; each term may broadcast over the
// channel axis (E, R, 1), the row axis (E, 1, C), both (E, 1, 1) or be a scalar.
// That covers "residual + gate + bias" shapes without a graph compiler.
//
// All tensors are contiguous float32 on one DCU and the channel axis is a
// multiple of four, so the inner loop is one float4 per thread.

#include "ops.h"
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <vector>

namespace {

constexpr int kMaxTerms = 8;
constexpr unsigned int kBlockThreads = 256;
constexpr int64_t kMaxGridBlocks = 65535;

// Broadcast mode of one term relative to the (E, R, C) output.
constexpr int kModeFull = 0;       // (E, R, C)
constexpr int kModePerRow = 1;     // (E, R, 1)
constexpr int kModePerChannel = 2; // (E, 1, C)
constexpr int kModePerEdge = 3;    // (E, 1, 1)
constexpr int kModeScalar = 4;     // (1,)

struct Params {
  const float* ptr[kMaxTerms];
  float coef[kMaxTerms];
  int mode[kMaxTerms];
  int terms;
  int64_t edges;
  int64_t rows;
  int64_t channels;
  int64_t groups;
};

__device__ __forceinline__ float4 load_term(const Params& p, int term, int64_t edge,
                                            int64_t row, int64_t group) {
  const float* base = p.ptr[term];
  switch (p.mode[term]) {
  case kModeFull:
    return reinterpret_cast<const float4*>(
        base + (edge * p.rows + row) * p.channels + group * 4)[0];
  case kModePerRow: {
    const float value = base[edge * p.rows + row];
    return make_float4(value, value, value, value);
  }
  case kModePerChannel:
    return reinterpret_cast<const float4*>(base + edge * p.channels + group * 4)[0];
  case kModePerEdge: {
    const float value = base[edge];
    return make_float4(value, value, value, value);
  }
  default: {
    const float value = base[0];
    return make_float4(value, value, value, value);
  }
  }
}

__global__ void edge_combine_kernel(Params p, float* out) {
  const int64_t work = p.edges * p.rows * p.groups;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < work; index += stride) {
    const int64_t group = index % p.groups;
    const int64_t row = (index / p.groups) % p.rows;
    const int64_t edge = index / (p.groups * p.rows);
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int term = 0; term < p.terms; ++term) {
      const float4 value = load_term(p, term, edge, row, group);
      const float coef = p.coef[term];
      acc.x += coef * value.x;
      acc.y += coef * value.y;
      acc.z += coef * value.z;
      acc.w += coef * value.w;
    }
    reinterpret_cast<float4*>(out + (edge * p.rows + row) * p.channels + group * 4)[0] = acc;
  }
}

int64_t groups_for(int64_t channels) { return channels / 4; }

unsigned int grid_for(int64_t work) {
  const int64_t blocks = (work + kBlockThreads - 1) / kBlockThreads;
  const int64_t capped = blocks < kMaxGridBlocks ? blocks : kMaxGridBlocks;
  return static_cast<unsigned int>(capped < 1 ? 1 : capped);
}

void check_term(const Tensor& term, int mode, int64_t edges, int64_t rows,
                int64_t channels) {
  f32(term);
  const int64_t numel = term.numel();
  switch (mode) {
  case kModeFull:
    TORCH_CHECK(term.dim() == 3 && term.size(0) == edges &&
                    term.size(1) == rows && term.size(2) == channels,
                "edge combine full term must have shape (E, R, C)");
    break;
  case kModePerRow:
    TORCH_CHECK(term.dim() == 3 && term.size(0) == edges &&
                    term.size(1) == rows && term.size(2) == 1,
                "edge combine per-row term must have shape (E, R, 1)");
    break;
  case kModePerChannel:
    TORCH_CHECK(term.dim() == 3 && term.size(0) == edges &&
                    term.size(1) == 1 && term.size(2) == channels,
                "edge combine per-channel term must have shape (E, 1, C)");
    break;
  case kModePerEdge:
    TORCH_CHECK(term.dim() == 3 && term.size(0) == edges &&
                    term.size(1) == 1 && term.size(2) == 1,
                "edge combine per-edge term must have shape (E, 1, 1)");
    break;
  default:
    TORCH_CHECK(numel == 1, "edge combine scalar term must have one element");
    break;
  }
}

} // namespace

Tensor edge_combine_hip(std::vector<Tensor> terms,
                        std::vector<double> coefficients,
                        std::vector<int64_t> modes, int64_t edges, int64_t rows,
                        int64_t channels) {
  TORCH_CHECK(!terms.empty(), "edge_combine needs at least one term");
  TORCH_CHECK(terms.size() <= kMaxTerms,
              "edge_combine supports at most ", kMaxTerms, " terms");
  TORCH_CHECK(coefficients.empty() || coefficients.size() == terms.size(),
              "edge_combine coefficients must match the term count");
  TORCH_CHECK(modes.size() == terms.size(),
              "edge_combine modes must match the term count");
  TORCH_CHECK(edges > 0 && rows > 0, "edge_combine needs a non-empty edge stream");
  TORCH_CHECK(channels > 0 && channels % 4 == 0,
              "edge_combine channels must be a positive multiple of 4");

  const auto device = terms.front().device();
  Params params{};
  params.terms = static_cast<int>(terms.size());
  params.edges = edges;
  params.rows = rows;
  params.channels = channels;
  params.groups = groups_for(channels);
  for (size_t index = 0; index < terms.size(); ++index) {
    const auto& term = terms[index];
    TORCH_CHECK(term.device() == device, "edge_combine terms must share a device");
    const int mode = static_cast<int>(modes[index]);
    TORCH_CHECK(mode >= kModeFull && mode <= kModeScalar,
                "edge_combine got an unknown broadcast mode ", mode);
    check_term(term, mode, edges, rows, channels);
    params.ptr[index] = term.data_ptr<float>();
    params.coef[index] = static_cast<float>(
        coefficients.empty() ? 1.0 : coefficients[index]);
    params.mode[index] = mode;
  }

  Tensor out = torch::empty({edges, rows, channels}, terms.front().options());
  c10::cuda::CUDAGuard guard(device);
  const int64_t work = edges * rows * params.groups;
  const dim3 grid(grid_for(work));
  const dim3 block(kBlockThreads);
  hipLaunchKernelGGL(edge_combine_kernel, grid, block, 0,
                     at::cuda::getCurrentCUDAStream(), params,
                     out.data_ptr<float>());
  launch_check();
  return out;
}
