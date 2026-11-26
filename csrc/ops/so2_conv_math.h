#pragma once

// The SO(2) convolution's per-edge math, shared by both entry points:
//
//   * the persistent operator (`ops/so2_conv_op.h`, running under the
//     megakernel VM), and
//   * the ordinary kernel (`csrc/so2_conv_dmk.cpp`, launched like every other
//     kernel in this tree).
//
// One copy of the arithmetic means the two paths cannot drift: a change in the
// gate, the residual placement or the rotation order shows up in both, and the
// accuracy test compares both against `reference/so2_conv_dmk.py`.
//
// Layouts (row-major, all float32):
//   rot    (red, dim)   local-frame rotation R
//   rot_t  (dim, red)   its transpose, the inverse rotation
//   kc     (layers, 2, red, red)  content kernel then gate kernel
//
// One edge, one channel:
//   xl[r]    = sum_d R[r][d] x[d]
//   layer:  z[r] = sum_s K[r][s] xl[s];  g[r] = sum_s G[r][s] xl[s]
//           xl[r] += silu(z[r]) * sigmoid(g[r])
//   y[d]     = sum_r R^T[d][r] xl[r]
//
// `dim` and `red` are runtime values here, so the same code serves any lmax.
// Templating on (dim, red) would let the compiler unroll the two inner loops;
// that is deliberate future work, not an oversight -- the first version is
// measured before it is specialised.

#include <hip/hip_runtime.h>

namespace dpa4::so2 {

//: Upper bound on (lmax + 1)^2 and 3 * lmax + 1 for lmax <= 4.
inline constexpr int kMaxDim = 32;
inline constexpr int kMaxRed = 32;

//: silu(content) * sigmoid(gate), the gated layer's residual term.
__device__ __forceinline__ float gated_term(float content, float gate) {
    const float sigmoid_content = 1.0f / (1.0f + __expf(-content));
    const float sigmoid_gate = 1.0f / (1.0f + __expf(-gate));
    return content * sigmoid_content * sigmoid_gate;
}

//: xl = R @ x, one lane per channel.
__device__ __forceinline__ void rotate_to_local(const float* rot, const float* x,
                                                float* xl, int dim, int red) {
    for (int r = 0; r < red; ++r) {
        const float* row = rot + static_cast<long>(r) * dim;
        float acc = 0.0f;
        for (int d = 0; d < dim; ++d) acc = fmaf(row[d], x[d], acc);
        xl[r] = acc;
    }
}

//: One gated layer, in place: xl += silu(K @ xl) * sigmoid(G @ xl).
__device__ __forceinline__ void gated_layer(const float* content,
                                            const float* gate, float* xl,
                                            int red) {
    float z[kMaxRed];
    float g[kMaxRed];
    for (int r = 0; r < red; ++r) {
        const float* crow = content + static_cast<long>(r) * red;
        const float* grow = gate + static_cast<long>(r) * red;
        float a = 0.0f;
        float b = 0.0f;
        for (int s = 0; s < red; ++s) {
            const float v = xl[s];
            a = fmaf(crow[s], v, a);
            b = fmaf(grow[s], v, b);
        }
        z[r] = a;
        g[r] = b;
    }
    for (int r = 0; r < red; ++r) xl[r] += gated_term(z[r], g[r]);
}

//: y = R^T @ xl.
__device__ __forceinline__ void rotate_back(const float* rot_t, const float* xl,
                                            float* y, int dim, int red) {
    for (int d = 0; d < dim; ++d) {
        const float* row = rot_t + static_cast<long>(d) * red;
        float acc = 0.0f;
        for (int r = 0; r < red; ++r) acc = fmaf(row[r], xl[r], acc);
        y[d] = acc;
    }
}

//: The whole span for one edge and one channel: rotate, stack, rotate back.
//
// ``x`` is the source atom's feature column (``dim`` floats, already gathered),
// ``out`` receives ``dim`` floats before the envelope weight is applied -- the
// caller multiplies, so the weight's placement stays visible at the call site.
__device__ __forceinline__ void edge_span(const float* x, const float* rot,
                                          const float* rot_t, const float* kc,
                                          int dim, int red, int layers,
                                          float* out, float* scratch_xl) {
    float* xl = scratch_xl;
    rotate_to_local(rot, x, xl, dim, red);
    const long block = static_cast<long>(red) * red;
    for (int layer = 0; layer < layers; ++layer) {
        const float* content = kc + 2 * block * layer;
        gated_layer(content, content + block, xl, red);
    }
    rotate_back(rot_t, xl, out, dim, red);
}

}  // namespace dpa4::so2
