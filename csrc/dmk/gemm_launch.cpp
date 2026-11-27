// Persistent GEMM: the megakernel with a GEMM-only op pack.
//
// This is the smallest end-to-end path through the persistent runtime -- an
// instruction stream in, a dense product out -- and the only operator the host
// driver (`python/dmk`) can launch today.  The arithmetic inside is DUMMA
// (`ops/gemm.h` -> `ops/dumma.h`), so the kernel is limited by the matrix unit
// rather than by an FFMA loop.
//
// Layout matches `ops/gemm.h`: one instruction computes a 16x16 output tile of
// `C = A @ B` by walking `k_tiles` k-steps of 16,
//
//     C[m0 : m0+16, n0 : n0+16] = sum_k A[m0 : m0+16, 16k : 16k+16]
//                                     @ B[16k : 16k+16, n0 : n0+16]
//
// so M, N and K must all be multiples of 16: the operator has no edge path, and
// this launcher rejects a shape that would need one rather than producing a
// silently wrong tile.

// The megakernel headers come first: `ops.h` defines a global `Tensor` alias for
// torch::Tensor, and including it ahead of `dmk/megakernel.h` makes the
// `tensor::Tensor` template used inside `dmk::mk` ambiguous to the eye (and, in
// the first build, unresolvable).
#include "dmk/megakernel.h"
#include "ops/gemm.h"
#include "ops/noop.h"

#include "checks.h"
#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace dmk::mk {
namespace {

//: Globals for a GEMM-only kernel.  `MkGlobals` also carries the demo's 1-D
//: input/output, whose validity check a GEMM launch cannot satisfy.
struct GemmGlobals {
    Tensor<const int, 3, MemorySpace::HBM> instructions;  // (blocks, len, 32)
    Tensor<const float, 2, MemorySpace::HBM> matrix_a;    // (M, K)
    Tensor<const float, 2, MemorySpace::HBM> matrix_b;    // (K, N)
    Tensor<float, 2, MemorySpace::HBM> matrix_c;          // (M, N)
    int num_instructions = 0;

    DMK_HOST_DEVICE bool valid() const {
        if (num_instructions <= 0) return false;
        if (instructions.data == nullptr || matrix_a.data == nullptr ||
            matrix_b.data == nullptr || matrix_c.data == nullptr)
            return false;
        if (instructions.shape.dims[2] != config::kInstructionWidth) return false;
        const size_t m = matrix_c.shape.dims[0];
        const size_t n = matrix_c.shape.dims[1];
        const size_t k = matrix_a.shape.dims[1];
        return matrix_a.shape.dims[0] == m && matrix_b.shape.dims[0] == k &&
               matrix_b.shape.dims[1] == n;
    }
};

__global__ void gemm_mk(const GemmGlobals g) {
    mk_internal<GemmGlobals, NoOp<GemmGlobals>, GemmOp<GemmGlobals>>(g);
}

bool report(const char* message) {
    std::fprintf(stderr, "dmk::gemm: %s\n", message);
    return false;
}

bool request_dynamic_lds() {
    hipFuncAttributes attributes{};
    if (hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(gemm_mk)) !=
        hipSuccess)
        return report("hipFuncGetAttributes failed");
    if (static_cast<size_t>(attributes.sharedSizeBytes) >= config::kLdsBytesPerCu)
        return report("static LDS already fills the CU's 64 KiB");
    const auto status = hipFuncSetAttribute(
        reinterpret_cast<const void*>(gemm_mk),
        hipFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kLaunchBytes));
    if (status != hipSuccess) {
        std::fprintf(stderr,
                     "dmk::gemm: cannot reserve %zu B of dynamic LDS: %s\n",
                     kLaunchBytes, hipGetErrorString(status));
        return false;
    }
    return true;
}

}  // namespace
}  // namespace dmk::mk

namespace {

using namespace dmk;

//: Whether the shapes can be tiled by 16 with no edge path.
bool tileable(int64_t m, int64_t n, int64_t k) {
    return m % 16 == 0 && n % 16 == 0 && k % 16 == 0;
}

}  // namespace

Tensor dmk_gemm_hip(Tensor instructions, Tensor matrix_a, Tensor matrix_b,
                    Tensor matrix_c) {
    TORCH_CHECK(instructions.scalar_type() == torch::kInt &&
                    instructions.is_contiguous() && instructions.dim() == 3 &&
                    instructions.size(2) == config::kInstructionWidth,
                "instructions must be a contiguous int32 (blocks, length, 32) tensor");
    f32(matrix_a);
    f32(matrix_b);
    f32(matrix_c);
    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2 && matrix_c.dim() == 2,
                "matrices must be 2-D");
    const int64_t m = matrix_c.size(0);
    const int64_t n = matrix_c.size(1);
    const int64_t k = matrix_a.size(1);
    TORCH_CHECK(matrix_a.size(0) == m && matrix_b.size(0) == k &&
                    matrix_b.size(1) == n,
                "expected A (M,K), B (K,N), C (M,N)");
    TORCH_CHECK(tileable(m, n, k),
                "M, N and K must be multiples of 16: the persistent GEMM walks "
                "16x16 tiles and has no edge path (got ", m, "x", n, "x", k, ")");
    TORCH_CHECK(instructions.size(0) > 0 && instructions.size(1) > 0,
                "the instruction stream must have at least one block and one instruction");
    same_device(matrix_a, {matrix_b, matrix_c});

    const at::cuda::OptionalCUDAGuard guard(matrix_a.device());
    if (!dmk::mk::request_dynamic_lds()) {
        TORCH_CHECK(false, "the persistent GEMM cannot reserve its LDS");
    }

    dmk::mk::GemmGlobals g{};
    // `from_torch` takes a mutable handle (it reads shape/strides from it), so the
    // parameters are copied rather than bound by reference to the caller's.
    at::Tensor instructions_handle = instructions;
    at::Tensor a_handle = matrix_a;
    at::Tensor b_handle = matrix_b;
    at::Tensor c_handle = matrix_c;
    ops::tensor::from_torch(instructions_handle, g.instructions);
    ops::tensor::from_torch(a_handle, g.matrix_a);
    ops::tensor::from_torch(b_handle, g.matrix_b);
    ops::tensor::from_torch(c_handle, g.matrix_c);
    g.num_instructions = static_cast<int>(instructions.size(1));
    TORCH_CHECK(g.valid(), "the persistent GEMM globals are not valid");

    const uint32_t blocks = static_cast<uint32_t>(instructions.size(0));
    dmk::mk::gemm_mk<<<blocks, dmk::mk::config::kThreads, dmk::mk::kLaunchBytes,
                       at::cuda::getCurrentCUDAStream()>>>(g);
    launch_check();
    return matrix_c;
}
