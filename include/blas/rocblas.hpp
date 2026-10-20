#pragma once
#include "layout.hpp"

#ifndef DPA4_ENABLE_ROCBLAS
#define DPA4_ENABLE_ROCBLAS 0
#endif

#if DPA4_ENABLE_ROCBLAS
#if !__has_include(<rocblas/rocblas.h>)
#error "rocBLAS requested but rocblas/rocblas.h is unavailable"
#else
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cmath>
#include <initializer_list>
#include <string>
#include <type_traits>

namespace dcu::blas {
// HOST API, not callable from __device__ code. Library launches run on stream.
inline void check(rocblas_status status, const char* operation) {
    if (status != rocblas_status_success)
        throw std::runtime_error(std::string(operation) + " rocBLAS status=" +
                                 std::to_string(static_cast<int>(status)));
}
inline void check_hip(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

class handle {
    rocblas_handle value_ = nullptr;
    int device_ = 0;
    hipStream_t stream_ = nullptr;
    bool bound_ = false;
public:
    handle() {
        check_hip(hipGetDevice(&device_));
        check(rocblas_create_handle(&value_), "create_handle");
    }
    handle(const handle&) = delete;
    handle& operator=(const handle&) = delete;
    ~handle() noexcept {
        int prior = 0;
        // Thread/process teardown may occur after the HIP runtime is gone.
        if (!value_ || hipGetDevice(&prior) != hipSuccess) return;
        if (prior != device_ && hipSetDevice(device_) != hipSuccess) return;
        rocblas_destroy_handle(value_);
        if (prior != device_) hipSetDevice(prior);
    }
    rocblas_handle prepare(hipStream_t stream) {
        int current = 0;
        check_hip(hipGetDevice(&current));
        if (current != device_) throw std::invalid_argument("rocBLAS handle device mismatch");
        if (bound_ && stream != stream_)
            throw std::invalid_argument("rocBLAS handle is bound to another stream; use one handle per stream");
        check(rocblas_set_stream(value_, stream), "set_stream");
        check(rocblas_set_pointer_mode(value_, rocblas_pointer_mode_host), "set_pointer_mode");
        check(rocblas_set_atomics_mode(value_, rocblas_atomics_not_allowed), "set_atomics_mode");
        stream_ = stream;
        bound_ = true;
        return value_;
    }
};

inline rocblas_operation operation(bool transpose) {
    return transpose ? rocblas_operation_transpose : rocblas_operation_none;
}
template<typename T>
inline void scalars(T alpha, T beta) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "rocBLAS wrapper supports float/double only");
    if (!std::isfinite(alpha) || !std::isfinite(beta))
        throw std::invalid_argument("BLAS alpha/beta must be finite");
}

// In-place C = alpha*op(A)*op(B) + beta*C; all storage is row-major.
// Caller owns device buffers, alignment/capacity/lifetime and stream ordering.
// No physical transpose. A/B must not overlap C. For K=0 use caller epilogue.
template<typename T>
inline void gemm(handle& owner, hipStream_t stream,
                 std::int64_t m, std::int64_t n, std::int64_t k,
                 const T* a, std::int64_t lda, bool ta,
                 const T* b, std::int64_t ldb, bool tb,
                 T* c, std::int64_t ldc, T alpha = T(1), T beta = T(0)) {
    const auto p = row_major_plan(m, n, k, ta, tb, lda, ldb, ldc);
    scalars(alpha, beta);
    if (m == 0 || n == 0) return;
    if (k == 0) throw std::invalid_argument("low-level K=0 requires a separate scale epilogue");
    if (!a || !b || !c) throw std::invalid_argument("BLAS device pointer is null");
    const auto native = owner.prepare(stream);
    if constexpr (std::is_same_v<T, float>)
        check(rocblas_sgemm(native, operation(p.trans_a), operation(p.trans_b),
                           p.m, p.n, p.k, &alpha, b, p.lda, a, p.ldb, &beta, c, p.ldc), "sgemm");
    else
        check(rocblas_dgemm(native, operation(p.trans_a), operation(p.trans_b),
                           p.m, p.n, p.k, &alpha, b, p.lda, a, p.ldb, &beta, c, p.ldc), "dgemm");
}

template<typename T>
inline void gemm_strided_batched(handle& owner, hipStream_t stream,
                                 std::int64_t m, std::int64_t n, std::int64_t k,
                                 const T* a, std::int64_t lda, std::int64_t stride_a, bool ta,
                                 const T* b, std::int64_t ldb, std::int64_t stride_b, bool tb,
                                 T* c, std::int64_t ldc, std::int64_t stride_c,
                                 std::int64_t batches, T alpha = T(1), T beta = T(0)) {
    const auto p = row_major_plan(m, n, k, ta, tb, lda, ldb, ldc);
    const int batch_count = checked_int(batches);
    scalars(alpha, beta);
    if (stride_a < 0 || stride_b < 0 || stride_c < 0)
        throw std::invalid_argument("BLAS batch strides must be nonnegative");
    if (batches == 0 || m == 0 || n == 0) return;
    if (k == 0) throw std::invalid_argument("low-level K=0 requires a separate scale epilogue");
    if (!a || !b || !c) throw std::invalid_argument("BLAS device pointer is null");
    if (batches > 1) {
        if ((stride_a != 0 && stride_a < (ta ? k : m) * lda) ||
            (stride_b != 0 && stride_b < (tb ? n : k) * ldb) || stride_c < m * ldc)
            throw std::invalid_argument("overlapping BLAS batch storage");
        for (auto stride : {stride_a, stride_b, stride_c})
            if (stride > std::numeric_limits<std::int64_t>::max() / batches)
                throw std::overflow_error("BLAS batched address span overflow");
    }
    const auto native = owner.prepare(stream);
    // Swap both pointers AND their leading dimensions/batch strides.
    if constexpr (std::is_same_v<T, float>)
        check(rocblas_sgemm_strided_batched(native, operation(p.trans_a), operation(p.trans_b),
              p.m, p.n, p.k, &alpha, b, p.lda, stride_b, a, p.ldb, stride_a,
              &beta, c, p.ldc, stride_c, batch_count), "sgemm_strided_batched");
    else
        check(rocblas_dgemm_strided_batched(native, operation(p.trans_a), operation(p.trans_b),
              p.m, p.n, p.k, &alpha, b, p.lda, stride_b, a, p.ldb, stride_a,
              &beta, c, p.ldc, stride_c, batch_count), "dgemm_strided_batched");
}
} // namespace dcu::blas
#endif // header available
#endif // DPA4_ENABLE_ROCBLAS
