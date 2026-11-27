// The shipped persistent kernel, and the host entry that launches it.
//
// One `__global__` instantiation per op pack -- this file is that instantiation,
// not a launch implementation: the launch itself is `dmk/launch.h`, and the
// globals are `MkGlobals` from `dmk/megakernel.h`, shared with every other pack.
// Adding an operator to the pack below is the whole change needed to make it
// reachable from the host; adding a *model* (a different pack) means a file like
// this one, not a copy of the launch code.
//
// Layout per `ops/gemm.h`: one instruction covers `kConsumerWaves` output tiles
// along N, one per consumer wave,
//
//     wave w:  C[m0 : m0+16, n0 + 16w : n0 + 16w + 16] =
//              sum_k A[m0 : m0+16, 16k : 16k+16] @ B[16k : 16k+16, n0 + 16w ...]
//
// so M, N and K must all be multiples of 16: the operator has no edge path, and
// the entry rejects a shape that would need one instead of producing a silently
// wrong tile.

#include "dmk/launch.h"
#include "dmk/megakernel.h"
#include "ops/gemm.h"
#include "ops/noop.h"

#include "checks.h"
#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace dmk::mk {

__global__ void mk_harness(const MkGlobals g) {
    mk_internal<MkGlobals, ops::NoOp<MkGlobals>, ops::GemmOp<MkGlobals>>(g);
}

}  // namespace dmk::mk

#ifndef __HIP_DEVICE_COMPILE__
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
                    instructions.size(2) == ops::config::kInstructionWidth,
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
                "M, N and K must be multiples of 16: the persistent GEMM works in "
                "16x16 tiles and has no edge path (got ", m, "x", n, "x", k, ")");
    TORCH_CHECK(instructions.size(0) > 0 && instructions.size(1) > 0,
                "the instruction stream must have at least one block and one "
                "instruction");
    same_device(matrix_a, {matrix_b, matrix_c});

    const at::cuda::OptionalCUDAGuard guard(matrix_a.device());

    mk::MkGlobals g{};
    // `from_torch` takes a mutable handle (it reads shape and strides from it).
    at::Tensor instructions_handle = instructions;
    at::Tensor a_handle = matrix_a;
    at::Tensor b_handle = matrix_b;
    at::Tensor c_handle = matrix_c;
    ops::tensor::from_torch(instructions_handle, g.instructions);
    ops::tensor::from_torch(a_handle, g.matrix_a);
    ops::tensor::from_torch(b_handle, g.matrix_b);
    ops::tensor::from_torch(c_handle, g.matrix_c);
    g.num_instructions = static_cast<int>(instructions.size(1));

    const uint32_t blocks = static_cast<uint32_t>(instructions.size(0));
    TORCH_CHECK(g.valid_instructions(blocks),
                "the instruction stream does not match the globals");
    size_t gm = 0, gn = 0, gk = 0;
    TORCH_CHECK(g.gemm_shapes(gm, gn, gk), "the GEMM operands are not consistent");

    TORCH_CHECK(mk::launch_kernel(mk::mk_harness, g, blocks,
                                  at::cuda::getCurrentCUDAStream()),
                "the persistent GEMM launch failed");
    launch_check();
    return matrix_c;
}
#endif  // __HIP_DEVICE_COMPILE__
