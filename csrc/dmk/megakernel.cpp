#include "dmk/launch.h"
#include "dmk/megakernel.h"

#include "ops/elementwise.h"
#include "ops/embedding.h"
#include "ops/gemm.h"
#include "ops/noop.h"

#include <cstdint>
#include <cstdio>

namespace dmk::mk {
namespace {

// The second shipped op pack: everything this tree has, used by the standalone
// LDS/pipeline test (`tests/csrc/test_dmk_lds_megakernel.cpp`), which builds
// instruction streams of every op.  The product path ships a narrower pack in
// `mk_harness.cpp`.
__global__ void mk_demo(const MkGlobals g) {
    mk_internal<MkGlobals, ops::NoOp<MkGlobals>, ops::ElementwiseOp<MkGlobals>,
                ops::GemmOp<MkGlobals>, ops::EmbeddingOp<MkGlobals>>(g);
}

}  // namespace

bool launch_mk(const MkGlobals& g, uint32_t blocks, hipStream_t stream) {
    // The launch itself (LDS reservation, grid, stream) is shared with every
    // other kernel: see dmk/launch.h.
    if (!g.valid(blocks)) return false;
    return launch_kernel(mk_demo, g, blocks, stream);
}

}  // namespace dmk::mk
