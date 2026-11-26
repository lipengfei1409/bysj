#include "dmk/megakernel.h"

#include "ops/demo_op.h"
#include "ops/embedding.h"
#include "ops/gemm.h"
#include "ops/noop.h"

#include <cstdint>
#include <cstdio>

namespace dmk::mk {
namespace {

// Concrete op pack for the demo. A real model edits this list (the reference
// implementation does the same in llama.cu / its pybind instantiation).
__global__ void demo_kernel(MkGlobals g) {
    mk_internal<MkGlobals, NoOp<MkGlobals>,
                ElementwiseOp<MkGlobals>, GemmOp<MkGlobals>,
                EmbeddingOp<MkGlobals>>(g);
}

bool report(const char* message) {
    std::fprintf(stderr, "dmk::mk: %s\n", message);
    return false;
}

bool device_supports_launch() {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return report("hipGetDevice failed");

    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device) != hipSuccess)
        return report("hipGetDeviceProperties failed");
    if (properties.warpSize != config::kWaveSize)
        return report("role partitioning requires 64-lane wavefronts");
    if (properties.maxThreadsPerBlock < config::kThreads)
        return report("device cannot host all role waves in one block");

    hipFuncAttributes attributes{};
    const bool have_attributes =
        hipFuncGetAttributes(&attributes,
                             reinterpret_cast<const void*>(&demo_kernel)) ==
        hipSuccess;
    if (!have_attributes) {
        std::fprintf(stderr,
                     "dmk::mk: warning: hipFuncGetAttributes failed; "
                     "checking the device limit only\n");
    }

    const size_t static_bytes =
        have_attributes ? attributes.sharedSizeBytes : kStaticBytes;
    if (properties.sharedMemPerBlock < static_bytes)
        return report("static LDS usage already exceeds the per-block limit");
    const size_t available = properties.sharedMemPerBlock - static_bytes;
    if (kLaunchBytes > available)
        return report("planned dynamic LDS exceeds the per-block limit");

    const bool needs_optin =
        !have_attributes ||
        kLaunchBytes > static_cast<size_t>(attributes.maxDynamicSharedSizeBytes);
    if (needs_optin) {
        if (hipFuncSetAttribute(
                reinterpret_cast<const void*>(&demo_kernel),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(kLaunchBytes)) != hipSuccess) {
            std::fprintf(stderr,
                         "dmk::mk: warning: hipFuncSetAttribute for dynamic "
                         "LDS failed; the launch may fail\n");
        }
    }
    return true;
}

}  // namespace

bool launch_mk(const MkGlobals& g, uint32_t blocks, hipStream_t stream) {
    if (blocks == 0) return report("at least one block is required");
    if (!g.valid(blocks)) return report("globals are inconsistent");

    if (!device_supports_launch()) return false;

    hipLaunchKernelGGL(demo_kernel, dim3(blocks), dim3(config::kThreads),
                       static_cast<uint32_t>(kLaunchBytes), stream, g);
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        std::fprintf(stderr, "dmk::mk: launch failed: %s\n",
                     hipGetErrorString(status));
        return false;
    }
    return true;
}

}  // namespace dmk::mk
