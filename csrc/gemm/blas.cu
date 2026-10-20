#include "../include/ops.h"
#include "../include/checks.h"
#include "blas/rocblas.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <map>
#include <memory>

static dcu::blas::handle& current_handle(hipStream_t stream) {
    // Separate handles/workspace for concurrently running streams and threads.
    using stream_handles = std::map<hipStream_t, std::unique_ptr<dcu::blas::handle>,
                                    std::less<hipStream_t>>;
    thread_local std::map<int, stream_handles> handles;
    int device = 0;
    dcu::blas::check_hip(hipGetDevice(&device));
    auto& owner = handles[device][stream];
    if (!owner) owner = std::make_unique<dcu::blas::handle>();
    return *owner;
}

static void blas_tensor(const Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                (x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64),
                "rocBLAS requires contiguous DCU float32/float64 tensor");
}

template<typename T>
__global__ void scale_output(T* output, const T* c, int64_t count, T beta) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = beta == T(0) ? T(0) : beta * c[i];
}

Tensor blas_gemm_hip(Tensor a, Tensor b, Tensor c, bool has_c, bool ta, bool tb,
                     double alpha, double beta) {
    blas_tensor(a); blas_tensor(b); blas_tensor(c); same_device(a, {b, c});
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(),
                "rocBLAS inputs must have matching dtype");
    TORCH_CHECK((a.dim() == 2 || a.dim() == 3) && b.dim() == a.dim(), "GEMM requires 2D/3D inputs");
    const int64_t m = a.size(ta ? -1 : -2), k = a.size(ta ? -2 : -1);
    const int64_t kb = b.size(tb ? -1 : -2), n = b.size(tb ? -2 : -1);
    TORCH_CHECK(k == kb, "GEMM inner dimensions disagree");
    const int64_t batches = a.dim() == 3 ? a.size(0) : 1;
    TORCH_CHECK(a.dim() == 2 || batches == b.size(0), "GEMM batch sizes disagree");
    auto shape = a.sizes().vec(); shape[shape.size() - 2] = m; shape.back() = n;
    TORCH_CHECK(has_c ? c.sizes().vec() == shape : c.numel() == 0, "GEMM C shape mismatch");
    TORCH_CHECK(has_c || beta == 0, "nonzero beta requires C");
    TORCH_CHECK(std::isfinite(alpha) && std::isfinite(beta), "GEMM alpha/beta must be finite");
    if (a.scalar_type() == torch::kFloat32)
        TORCH_CHECK(std::abs(alpha) <= std::numeric_limits<float>::max() &&
                    std::abs(beta) <= std::numeric_limits<float>::max(), "FP32 alpha/beta overflow");
    const int64_t lda = std::max<int64_t>(1, a.size(-1));
    const int64_t ldb = std::max<int64_t>(1, b.size(-1));
    const int64_t ldc = std::max<int64_t>(1, n);
    dcu::blas::row_major_plan(m, n, k, ta, tb, lda, ldb, ldc);
    dcu::blas::checked_int(batches);
    c10::cuda::CUDAGuard guard(a.device());
    auto out = torch::empty(shape, a.options());
    if (out.numel() == 0) return out;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (k == 0 || alpha == 0) {
        if (a.scalar_type() == torch::kFloat32)
            hipLaunchKernelGGL(scale_output<float>, dim3(blocks(out.numel())), dim3(256), 0, stream,
                              out.data_ptr<float>(), has_c ? c.data_ptr<float>() : nullptr,
                              out.numel(), static_cast<float>(beta));
        else
            hipLaunchKernelGGL(scale_output<double>, dim3(blocks(out.numel())), dim3(256), 0, stream,
                              out.data_ptr<double>(), has_c ? c.data_ptr<double>() : nullptr,
                              out.numel(), beta);
        launch_check();
        return out;
    }
    if (has_c && beta != 0) out.copy_(c);
    auto& owner = current_handle(stream);
    const int64_t stride_a = a.dim() == 3 ? a.stride(0) : 0;
    const int64_t stride_b = b.dim() == 3 ? b.stride(0) : 0;
    const int64_t stride_c = out.dim() == 3 ? out.stride(0) : 0;
    if (a.scalar_type() == torch::kFloat32) {
        if (a.dim() == 2)
            dcu::blas::gemm(owner, stream, m, n, k, a.data_ptr<float>(), lda, ta,
                            b.data_ptr<float>(), ldb, tb, out.data_ptr<float>(), ldc,
                            static_cast<float>(alpha), static_cast<float>(beta));
        else
            dcu::blas::gemm_strided_batched(owner, stream, m, n, k,
                a.data_ptr<float>(), lda, stride_a, ta, b.data_ptr<float>(), ldb, stride_b, tb,
                out.data_ptr<float>(), ldc, stride_c, batches, static_cast<float>(alpha), static_cast<float>(beta));
    } else {
        if (a.dim() == 2)
            dcu::blas::gemm(owner, stream, m, n, k, a.data_ptr<double>(), lda, ta,
                            b.data_ptr<double>(), ldb, tb, out.data_ptr<double>(), ldc, alpha, beta);
        else
            dcu::blas::gemm_strided_batched(owner, stream, m, n, k,
                a.data_ptr<double>(), lda, stride_a, ta, b.data_ptr<double>(), ldb, stride_b, tb,
                out.data_ptr<double>(), ldc, stride_c, batches, alpha, beta);
    }
    return out;
}
