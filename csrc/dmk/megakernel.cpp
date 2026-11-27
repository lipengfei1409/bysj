// The one persistent kernel, and the launch everything goes through.
//
// There is a single `__global__` here on purpose.  `mk_internal` is a template,
// and a template that was never instantiated is not a symbol -- so a launch needs
// at least one concrete instantiation, but it does *not* need one per operator:
// the pack below holds every operator this tree has, and `make_op` dispatches on
// the opcode at runtime.  Adding an operator therefore means adding a header and a
// type to this list, not a new kernel, a new file or a new entry point.
//
// The cost of the superset is kernel code size: an op that a given instruction
// stream never uses still occupies instruction cache.  Each op is a few hundred
// bytes and the controller constructs one op per instruction, so the dispatch
// chain (one compare per opcode) is negligible next to the work it launches.
//
// The launch mechanics themselves live in `dmk/launch.h`: reserve the dynamic LDS
// pages, one block per instruction row, current stream.

#include "dmk/launch.h"
#include "dmk/megakernel.h"

#include "ops/elementwise.h"
#include "ops/embedding.h"
#include "ops/gemm.h"
#include "ops/noop.h"

#include "checks.h"
#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace dmk::mk {

// The block is `kThreads` = (4 consumer + loader + storer + controller) * 64.
// Without the bound the compiler assumes hipcc's default maximum and the launch is
// clamped to 256 threads -- the last three waves then do not exist, which is
// invisible for a single-tile shape (wave 0 does everything) and silently wrong for
// every shape that needs the loader, the storer or the controller.
__global__ void __launch_bounds__(config::kThreads) mk_all(const MkGlobals g) {
    mk_internal<MkGlobals, ops::NoOp<MkGlobals>, ops::ElementwiseOp<MkGlobals>,
                ops::GemmOp<MkGlobals>, ops::EmbeddingOp<MkGlobals>>(g);
}

bool launch(const MkGlobals& g, uint32_t blocks, hipStream_t stream) {
    // Only the instruction stream is required here: which operands an instruction
    // needs is the op's business, and an op whose tensor is missing traps in its
    // own hook rather than in a validity check that would have to know every op.
    if (!g.valid_instructions(blocks)) return false;
    return launch_kernel(mk_all, g, blocks, stream);
}

}  // namespace dmk::mk

#ifndef __HIP_DEVICE_COMPILE__
namespace {

using namespace dmk;

//: Whether a dense GEMM of this shape can be tiled by 16 with no edge path.
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

    size_t gm = 0, gn = 0, gk = 0;
    TORCH_CHECK(g.gemm_shapes(gm, gn, gk), "the GEMM operands are not consistent");

    const uint32_t blocks = static_cast<uint32_t>(instructions.size(0));
    TORCH_CHECK(mk::launch(g, blocks, at::cuda::getCurrentCUDAStream()),
                "the persistent GEMM launch failed");
    launch_check();
    return matrix_c;
}
#endif  // __HIP_DEVICE_COMPILE__
