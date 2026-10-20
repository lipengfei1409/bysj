#pragma once
#include "../../common/tile.cuh"
#include "math.cuh"

namespace dcu::thread {
// Explicit scalar reference: D[M,N] = A[M,K] B[K,N] + C[M,N].
// D may alias C, but must NOT alias A or B. Not a hardware MMA instruction.
template<typename T, int M, int N, int K>
DCU_HD void mma_reference(register_tile<T, M, N>& d,
                           const register_tile<T, M, K>& a,
                           const register_tile<T, K, N>& b,
                           const register_tile<T, M, N>& c) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "reference MMA requires float or double");
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        T value = c(m, n);
        for (int k = 0; k < K; ++k) value = thread::fma(a(m, k), b(k, n), value);
        d(m, n) = value;
    }
}
} // namespace dcu::thread
