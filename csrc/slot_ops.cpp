// Small-axis operators over the DPA4 "slot" axis (S = 9/10 rows per edge).
//
// The profiled forward spends 75.7 ms on four passes over the same edge tensor
// to build the SO(2) input -- `index_select` to expand the three l-rows into
// nine, an elementwise `mul` for the radial scale, a `copy_` into a slice of the
// (E, 10, C) buffer, and a separate `fill_` for the reserved row. It spends
// another 24.9 ms on three `sum` calls over the ten-row axis at an effective
// 56 GB/s, because a ten-element axis goes through the generic reduction and
// vectorises badly.
//
//   slot_pack_scale  out[e, r, c] = slot[r] < 0 ? 0
//                                 : src[e, slot[r], c] * scale[e, r, c]
//   slot_reduce      out[e, c]    = sum_r weight[e, r] * src[e, r, c]
//
// Both are pure gather/scale/reduce: no atomics, no intermediate buffer, and the
// output is written once in the layout the consumer wants.

#include "ops.h"
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {
constexpr unsigned int kBlockThreads = 256;
constexpr unsigned int kMaxChannelThreads = 256;

unsigned int grid_for(int64_t blocks, const char* name) {
  TORCH_CHECK(blocks <= static_cast<int64_t>(
                  std::numeric_limits<unsigned int>::max()),
              name, " exceeds the HIP grid limit");
  return static_cast<unsigned int>(blocks);
}

unsigned int channel_threads(int64_t channels) {
  int64_t threads = (channels + 63) / 64 * 64;
  if (threads > kMaxChannelThreads) threads = kMaxChannelThreads;
  if (threads < 32) threads = 32;
  return static_cast<unsigned int>(threads);
}

// One thread handles one float4 group of one slot row; the index arithmetic is
// two 64-bit div/mods per 16 bytes, which the loads dominate.
template <bool kHasScale>
__global__ void slot_pack_scale_vec4_kernel(
    const float* __restrict__ src, const int64_t* __restrict__ slot,
    const float* __restrict__ scale, float* __restrict__ out,
    int64_t edges, int64_t rows_in, int64_t rows_out, int64_t groups,
    int64_t work_items) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < work_items; index += stride) {
    const int64_t group = index % groups;
    const int64_t row = (index / groups) % rows_out;
    const int64_t edge = index / (groups * rows_out);
    float4* target = reinterpret_cast<float4*>(out + (edge * rows_out + row) * (groups * 4));
    const int64_t source_row = slot[row];
    if (source_row < 0) {
      target[group] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      continue;
    }
    const float4 value =
        reinterpret_cast<const float4*>(src + (edge * rows_in + source_row) * (groups * 4))[group];
    if (kHasScale) {
      const float4 factor =
          reinterpret_cast<const float4*>(scale + (edge * rows_out + row) * (groups * 4))[group];
      target[group] = make_float4(value.x * factor.x, value.y * factor.y,
                                  value.z * factor.z, value.w * factor.w);
    } else {
      target[group] = value;
    }
  }
}

template <bool kHasScale>
__global__ void slot_pack_scale_kernel(
    const float* __restrict__ src, const int64_t* __restrict__ slot,
    const float* __restrict__ scale, float* __restrict__ out,
    int64_t rows_in, int64_t rows_out, int64_t channels, int64_t work_items) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < work_items; index += stride) {
    const int64_t channel = index % channels;
    const int64_t row = (index / channels) % rows_out;
    const int64_t edge = index / (channels * rows_out);
    const int64_t source_row = slot[row];
    const int64_t target = (edge * rows_out + row) * channels + channel;
    if (source_row < 0) {
      out[target] = 0.0f;
      continue;
    }
    const float value = src[(edge * rows_in + source_row) * channels + channel];
    out[target] = kHasScale
        ? value * scale[(edge * rows_out + row) * channels + channel]
        : value;
  }
}

// One block per edge: the whole slot axis fits in a few registers, so the
// reduction never touches memory twice.
template <bool kHasWeight>
__global__ void slot_reduce_kernel(
    const float* __restrict__ src, const float* __restrict__ weight,
    float* __restrict__ out, int64_t rows, int64_t channels) {
  const int64_t edge = blockIdx.x;
  const float* base = src + edge * rows * channels;
  float* target = out + edge * channels;
  for (int64_t channel = threadIdx.x; channel < channels;
       channel += blockDim.x) {
    float accumulator = 0.0f;
    for (int64_t row = 0; row < rows; ++row) {
      const float value = base[row * channels + channel];
      accumulator += kHasWeight ? weight[edge * rows + row] * value : value;
    }
    target[channel] = accumulator;
  }
}

void check_slot(const Tensor& slot, int64_t rows_out) {
  TORCH_CHECK(slot.is_cuda() && slot.scalar_type() == torch::kInt64 &&
                  slot.is_contiguous(),
              "slot pack slot must be a contiguous DCU int64 tensor");
  TORCH_CHECK(slot.dim() == 1 && slot.size(0) == rows_out,
              "slot pack slot must have shape (rows_out,)");
}
}  // namespace

Tensor slot_pack_scale_hip(Tensor src, Tensor slot,
                           c10::optional<Tensor> scale) {
  f32(src);
  TORCH_CHECK(src.dim() == 3,
              "slot pack src must have shape (n_edges, rows_in, channels)");
  const int64_t edges = src.size(0);
  const int64_t rows_in = src.size(1);
  const int64_t channels = src.size(2);
  TORCH_CHECK(channels > 0, "slot pack requires at least one channel");
  const int64_t rows_out = slot.defined() ? slot.size(0) : 0;
  check_slot(slot, rows_out);
  TORCH_CHECK(rows_out > 0, "slot pack requires at least one output row");
  if (scale.has_value()) {
    f32(*scale);
    TORCH_CHECK(scale->dim() == 3 && scale->size(0) == edges &&
                    scale->size(1) == rows_out && scale->size(2) == channels,
                "slot pack scale must have shape (n_edges, rows_out, channels)");
    same_device(src, {*scale});
  }
  same_device(src, {slot});

  Tensor out = torch::empty({edges, rows_out, channels}, src.options());
  const int64_t work_items = edges * rows_out * channels;
  if (work_items == 0) return out;

  const float* scale_ptr = scale.has_value() ? scale->data_ptr<float>() : nullptr;
  c10::cuda::CUDAGuard guard(src.device());
  const bool vectorized = channels % 4 == 0;
  const int64_t items = vectorized ? work_items / 4 : work_items;
  const dim3 grid(grid_for((items + kBlockThreads - 1) / kBlockThreads,
                           "slot pack work items"));
  if (vectorized) {
    if (scale_ptr == nullptr) {
      hipLaunchKernelGGL(slot_pack_scale_vec4_kernel<false>, grid,
                         dim3(kBlockThreads), 0,
                         at::cuda::getCurrentCUDAStream(),
                         src.data_ptr<float>(), slot.data_ptr<int64_t>(), nullptr,
                         out.data_ptr<float>(), edges, rows_in, rows_out,
                         channels / 4, items);
    } else {
      hipLaunchKernelGGL(slot_pack_scale_vec4_kernel<true>, grid,
                         dim3(kBlockThreads), 0,
                         at::cuda::getCurrentCUDAStream(),
                         src.data_ptr<float>(), slot.data_ptr<int64_t>(),
                         scale_ptr, out.data_ptr<float>(), edges, rows_in,
                         rows_out, channels / 4, items);
    }
  } else if (scale_ptr == nullptr) {
    hipLaunchKernelGGL(slot_pack_scale_kernel<false>, grid, dim3(kBlockThreads),
                       0, at::cuda::getCurrentCUDAStream(),
                       src.data_ptr<float>(), slot.data_ptr<int64_t>(), nullptr,
                       out.data_ptr<float>(), rows_in, rows_out, channels,
                       work_items);
  } else {
    hipLaunchKernelGGL(slot_pack_scale_kernel<true>, grid, dim3(kBlockThreads),
                       0, at::cuda::getCurrentCUDAStream(),
                       src.data_ptr<float>(), slot.data_ptr<int64_t>(),
                       scale_ptr, out.data_ptr<float>(), rows_in, rows_out,
                       channels, work_items);
  }
  launch_check();
  return out;
}

Tensor slot_reduce_hip(Tensor src, c10::optional<Tensor> weight) {
  f32(src);
  TORCH_CHECK(src.dim() == 3,
              "slot reduce src must have shape (n_edges, rows, channels)");
  const int64_t edges = src.size(0);
  const int64_t rows = src.size(1);
  const int64_t channels = src.size(2);
  TORCH_CHECK(rows > 0, "slot reduce requires at least one row");
  TORCH_CHECK(channels > 0, "slot reduce requires at least one channel");
  if (weight.has_value()) {
    f32(*weight);
    TORCH_CHECK(weight->dim() == 2 && weight->size(0) == edges &&
                    weight->size(1) == rows,
                "slot reduce weight must have shape (n_edges, rows)");
    same_device(src, {*weight});
  }

  Tensor out = torch::empty({edges, channels}, src.options());
  if (edges == 0) return out;

  const float* weight_ptr = weight.has_value() ? weight->data_ptr<float>() : nullptr;
  c10::cuda::CUDAGuard guard(src.device());
  const dim3 grid(grid_for(edges, "slot reduce edge count"));
  const dim3 block(channel_threads(channels));
  if (weight_ptr == nullptr) {
    hipLaunchKernelGGL(slot_reduce_kernel<false>, grid, block, 0,
                       at::cuda::getCurrentCUDAStream(), src.data_ptr<float>(),
                       nullptr, out.data_ptr<float>(), rows, channels);
  } else {
    hipLaunchKernelGGL(slot_reduce_kernel<true>, grid, block, 0,
                       at::cuda::getCurrentCUDAStream(), src.data_ptr<float>(),
                       weight_ptr, out.data_ptr<float>(), rows, channels);
  }
  launch_check();
  return out;
}
