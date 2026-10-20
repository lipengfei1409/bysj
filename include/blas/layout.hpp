#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace dcu::blas {
struct gemm_plan {
    int m, n, k, lda, ldb, ldc;
    bool trans_a, trans_b;
};
inline int checked_int(std::int64_t value) {
    if (value < 0) throw std::invalid_argument("BLAS dimensions must be nonnegative");
    if (value > std::numeric_limits<int>::max())
        throw std::overflow_error("rocBLAS 32-bit dimension limit exceeded");
    return static_cast<int>(value);
}
// C_row^T = op(B_row)^T op(A_row)^T, without moving/transposing storage.
// Returned column-major call uses B pointer first, then A pointer.
inline gemm_plan row_major_plan(std::int64_t m, std::int64_t n, std::int64_t k,
                                bool ta, bool tb, std::int64_t lda,
                                std::int64_t ldb, std::int64_t ldc) {
    const int mi = checked_int(m), ni = checked_int(n), ki = checked_int(k);
    const int ai = checked_int(lda), bi = checked_int(ldb), ci = checked_int(ldc);
    if (ai < std::max(1, ta ? mi : ki) || bi < std::max(1, tb ? ki : ni) ||
        ci < std::max(1, ni))
        throw std::invalid_argument("invalid row-major BLAS leading dimension");
    return {ni, mi, ki, bi, ai, ci, tb, ta};
}
} // namespace dcu::blas
