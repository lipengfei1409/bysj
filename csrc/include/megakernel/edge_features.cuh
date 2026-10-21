#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace dcu::megakernel {

struct EdgeFeaturesArgs {
  const float* positions;
  const int64_t* edge_index;
  const int64_t* shifts;
  const float* cells;
  const int64_t* edge_frame;
  const bool* mask;
  const float* frequencies;
  int64_t n_edges;
  int64_t n_radial;
  float cutoff;
  int basis_exponent;
  int edge_exponent;
  float eps;
  float* edge_vec;
  float* distance;
  float* unit;
  float* rbf;
  float* env;
};

__device__ __forceinline__ void c3_envelope_pair(
    float r, float cutoff, int exponent, float& value, float& derivative) {
  if (exponent == 0) {
    value = 1.f;
    derivative = 0.f;
    return;
  }
  if (r >= cutoff) {
    value = derivative = 0.f;
    return;
  }
  const float x = fmaxf(0.f, r / cutoff);
  const float u = 1.f - x;
  float sum = 0.f;
  float dsum = 0.f;
  float power = 1.f;
  float previous = 1.f;
  for (int k = 0; k < exponent; ++k) {
    const float coefficient = float((k + 1) * (k + 2) * (k + 3)) / 6.f;
    sum += coefficient * power;
    if (k > 0) dsum += float(k) * coefficient * previous;
    previous = power;
    power *= x;
  }
  const float u2 = u * u;
  const float u3 = u2 * u;
  const float u4 = u2 * u2;
  value = u4 * sum;
  derivative = (r < 0.f) ? 0.f : (-4.f * u3 * sum + u4 * dsum) / cutoff;
}

__device__ __forceinline__ void sinc_pair(
    float z, float& value, float& derivative) {
  if (fabsf(z) < 1.e-3f) {
    const float z2 = z * z;
    value = 1.f - z2 / 6.f + z2 * z2 / 120.f;
    derivative = -z / 3.f + z * z2 / 30.f;
  } else {
    value = sinf(z) / z;
    derivative = (z * cosf(z) - sinf(z)) / (z * z);
  }
}

__device__ __forceinline__ void edge_features_one(
    const EdgeFeaturesArgs& args, int64_t edge) {
  const int64_t src = args.edge_index[edge];
  const int64_t dst = args.edge_index[args.n_edges + edge];
  const int64_t frame = args.edge_frame[edge];
  float displacement[3];
  float norm2 = 0.f;
  for (int axis = 0; axis < 3; ++axis) {
    float value = args.positions[3 * src + axis] - args.positions[3 * dst + axis];
    for (int lattice = 0; lattice < 3; ++lattice) {
      value += float(args.shifts[3 * edge + lattice])
          * args.cells[9 * frame + 3 * lattice + axis];
    }
    displacement[axis] = value;
    norm2 += value * value;
  }
  if (!args.mask[edge]) {
    displacement[0] = 0.f;
    displacement[1] = 0.f;
    displacement[2] = 1.f;
    norm2 = 1.f;
  }
  const float r = sqrtf(norm2 + args.eps * args.eps);
  const float inv_r = 1.f / r;
  args.edge_vec[3 * edge] = displacement[0];
  args.edge_vec[3 * edge + 1] = displacement[1];
  args.edge_vec[3 * edge + 2] = displacement[2];
  args.distance[edge] = r;
  args.unit[3 * edge] = displacement[0] * inv_r;
  args.unit[3 * edge + 1] = displacement[1] * inv_r;
  args.unit[3 * edge + 2] = displacement[2] * inv_r;

  if (!args.mask[edge]) {
    args.env[edge] = 0.f;
    for (int64_t n = 0; n < args.n_radial; ++n) {
      args.rbf[edge * args.n_radial + n] = 0.f;
    }
    return;
  }
  float basis_env, basis_derivative, edge_env, edge_derivative;
  c3_envelope_pair(r, args.cutoff, args.basis_exponent,
                   basis_env, basis_derivative);
  c3_envelope_pair(r, args.cutoff, args.edge_exponent,
                   edge_env, edge_derivative);
  args.env[edge] = edge_env;
  for (int64_t n = 0; n < args.n_radial; ++n) {
    float sinc, sinc_derivative;
    sinc_pair(r * args.frequencies[n], sinc, sinc_derivative);
    args.rbf[edge * args.n_radial + n] = args.frequencies[n] * sinc * basis_env;
  }
}

}  // namespace dcu::megakernel
