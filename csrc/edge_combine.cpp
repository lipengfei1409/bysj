// C6: fused edge-stream elementwise chains.
//
// The profiled forward/backward spends 488 ms on E-major chains of the shape
// ``[1, E, R, C]``: a residual add, a broadcast multiply, a copy and a zero
// fill, each a separate pass over the same bytes. Every pass already runs at
// the memory roofline, so the only saving is the number of passes.
//
//   edge_combine   out[e, r, c] = sum_t coef_t * prod_f factor_{t,f}[e, ...]
//
// One pass, one kernel, one output write. Each factor broadcasts over the
// channel axis (E, R, 1), the row axis (E, 1, C), both (E, 1, 1), over the edge
// axis (1, R, 1) / (1, 1, C), or is a scalar. A term with one factor is an
// ordinary linear term; a term with two is a product (``x * gate``), which is
// what makes "residual + gate" chains expressible without a graph compiler.
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

constexpr int kMaxTerms = 4;
constexpr int kMaxFactors = 8;
constexpr unsigned int kBlockThreads = 256;
constexpr int64_t kMaxGridBlocks = 65535;

// Broadcast mode of one factor relative to the (E, R, C) output.
constexpr int kModeFull = 0;         // (E, R, C)
constexpr int kModePerRow = 1;       // (E, R, 1)
constexpr int kModePerChannel = 2;   // (E, 1, C)
constexpr int kModePerEdge = 3;      // (E, 1, 1)
constexpr int kModeRowConst = 4;     // (1, R, 1)
constexpr int kModeChannelConst = 5; // (1, 1, C)
constexpr int kModeScalar = 6;       // (1,)

struct Params {
  const float* ptr[kMaxFactors];
  int mode[kMaxFactors];
  int term_start[kMaxTerms];
  int term_count[kMaxTerms];
  float coef[kMaxTerms];
  int terms;
  int64_t edges;
  int64_t rows;
  int64_t channels;
  int64_t groups;
};

__device__ __forceinline__ float4 load_factor(const Params& p, int factor,
                                              int64_t edge, int64_t row,
                                              int64_t group) {
  const float* base = p.ptr[factor];
  switch (p.mode[factor]) {
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
  case kModeRowConst: {
    const float value = base[row];
    return make_float4(value, value, value, value);
  }
  case kModeChannelConst:
    return reinterpret_cast<const float4*>(base + group * 4)[0];
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
      float4 product = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
      const int begin = p.term_start[term];
      for (int factor = begin; factor < begin + p.term_count[term]; ++factor) {
        const float4 value = load_factor(p, factor, edge, row, group);
        product.x *= value.x;
        product.y *= value.y;
        product.z *= value.z;
        product.w *= value.w;
      }
      const float coef = p.coef[term];
      acc.x += coef * product.x;
      acc.y += coef * product.y;
      acc.z += coef * product.z;
      acc.w += coef * product.w;
    }
    reinterpret_cast<float4*>(out + (edge * p.rows + row) * p.channels + group * 4)[0] = acc;
  }
}

unsigned int grid_for(int64_t work) {
  const int64_t blocks = (work + kBlockThreads - 1) / kBlockThreads;
  const int64_t capped = blocks < kMaxGridBlocks ? blocks : kMaxGridBlocks;
  return static_cast<unsigned int>(capped < 1 ? 1 : capped);
}

void check_factor(const Tensor& factor, int mode, int64_t edges, int64_t rows,
                  int64_t channels) {
  f32(factor);
  const bool rank3 = factor.dim() == 3;
  const int64_t numel = factor.numel();
  switch (mode) {
  case kModeFull:
    TORCH_CHECK(rank3 && factor.size(0) == edges && factor.size(1) == rows &&
                    factor.size(2) == channels,
                "edge combine full factor must have shape (E, R, C)");
    break;
  case kModePerRow:
    TORCH_CHECK(rank3 && factor.size(0) == edges && factor.size(1) == rows &&
                    factor.size(2) == 1,
                "edge combine per-row factor must have shape (E, R, 1)");
    break;
  case kModePerChannel:
    TORCH_CHECK(rank3 && factor.size(0) == edges && factor.size(1) == 1 &&
                    factor.size(2) == channels,
                "edge combine per-channel factor must have shape (E, 1, C)");
    break;
  case kModePerEdge:
    TORCH_CHECK(rank3 && factor.size(0) == edges && factor.size(1) == 1 &&
                    factor.size(2) == 1,
                "edge combine per-edge factor must have shape (E, 1, 1)");
    break;
  case kModeRowConst:
    TORCH_CHECK(rank3 && factor.size(0) == 1 && factor.size(1) == rows &&
                    factor.size(2) == 1,
                "edge combine row-constant factor must have shape (1, R, 1)");
    break;
  case kModeChannelConst:
    TORCH_CHECK(rank3 && factor.size(0) == 1 && factor.size(1) == 1 &&
                    factor.size(2) == channels,
                "edge combine channel-constant factor must have shape (1, 1, C)");
    break;
  default:
    TORCH_CHECK(numel == 1, "edge combine scalar factor must have one element");
    break;
  }
}

} // namespace

Tensor edge_combine_hip(std::vector<Tensor> factors,
                        std::vector<int64_t> modes,
                        std::vector<double> coefficients,
                        std::vector<int64_t> term_starts,
                        std::vector<int64_t> term_counts, int64_t edges,
                        int64_t rows, int64_t channels) {
  const size_t terms = coefficients.size();
  TORCH_CHECK(terms > 0, "edge_combine needs at least one term");
  TORCH_CHECK(terms <= kMaxTerms, "edge_combine supports at most ", kMaxTerms,
              " terms");
  TORCH_CHECK(!factors.empty(), "edge_combine needs at least one factor");
  TORCH_CHECK(factors.size() <= kMaxFactors, "edge_combine supports at most ",
              kMaxFactors, " factors");
  TORCH_CHECK(modes.size() == factors.size(),
              "edge_combine modes must match the factor count");
  TORCH_CHECK(term_starts.size() == terms && term_counts.size() == terms,
              "edge_combine term tables must match the term count");
  TORCH_CHECK(edges > 0 && rows > 0, "edge_combine needs a non-empty edge stream");
  TORCH_CHECK(channels > 0 && channels % 4 == 0,
              "edge_combine channels must be a positive multiple of 4");

  const auto device = factors.front().device();
  Params params{};
  params.terms = static_cast<int>(terms);
  params.edges = edges;
  params.rows = rows;
  params.channels = channels;
  params.groups = channels / 4;
  for (size_t index = 0; index < factors.size(); ++index) {
    const auto& factor = factors[index];
    TORCH_CHECK(factor.device() == device,
                "edge_combine factors must share a device");
    const int mode = static_cast<int>(modes[index]);
    TORCH_CHECK(mode >= kModeFull && mode <= kModeScalar,
                "edge_combine got an unknown broadcast mode ", mode);
    check_factor(factor, mode, edges, rows, channels);
    params.ptr[index] = factor.data_ptr<float>();
    params.mode[index] = mode;
  }
  for (size_t term = 0; term < terms; ++term) {
    const int64_t start = term_starts[term];
    const int64_t count = term_counts[term];
    TORCH_CHECK(start >= 0 && count > 0 &&
                    start + count <= static_cast<int64_t>(factors.size()),
                "edge_combine term ", term, " references missing factors");
    params.term_start[term] = static_cast<int>(start);
    params.term_count[term] = static_cast<int>(count);
    params.coef[term] = static_cast<float>(coefficients[term]);
  }

  Tensor out = torch::empty({edges, rows, channels}, factors.front().options());
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
