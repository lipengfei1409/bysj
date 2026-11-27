#pragma once

// Host-side launch for a persistent kernel.
//
// `mk_internal` is a template over (Globals, op pack), so a launch always needs
// one concrete `__global__` instantiation to point at -- a kernel cannot be
// launched through a template that was never instantiated.  Everything *around*
// that pointer is the same for every kernel, which is what this header holds:
//
//   * ask the driver how much static LDS the instantiation already uses,
//   * reserve the dynamic remainder (the page array) as one request,
//   * launch with one block per CU row and the current stream.
//
// The per-kernel source file is then just the instantiation and the entry point;
// a second op pack (a model, a test harness) does not copy any of this.

#include "dmk/megakernel.h"
#include "ops/config.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace dmk::mk {

namespace launch_detail {

inline bool fail(const char* what, const char* why = nullptr) {
    if (why != nullptr)
        std::fprintf(stderr, "dmk::launch: %s: %s\n", what, why);
    else
        std::fprintf(stderr, "dmk::launch: %s\n", what);
    return false;
}

}  // namespace launch_detail

//: Reserve the dynamic LDS this kernel needs.  Returns false with a message on
//: stderr when the instantiation cannot fit, which is a build/hardware mismatch
//: rather than a caller error.
inline bool reserve_dynamic_lds(const void* kernel) {
    hipFuncAttributes attributes{};
    if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess)
        return launch_detail::fail("hipFuncGetAttributes failed");
    if (static_cast<size_t>(attributes.sharedSizeBytes) >=
        config::kLdsBytesPerCu)
        return launch_detail::fail("static LDS already fills the CU's 64 KiB");
    const auto status =
        hipFuncSetAttribute(kernel, hipFuncAttributeMaxDynamicSharedMemorySize,
                            static_cast<int>(kLaunchBytes));
    if (status != hipSuccess)
        return launch_detail::fail("cannot reserve the dynamic LDS pages",
                                   hipGetErrorString(status));
    return true;
}

//: Run `kernel` once per instruction row.
//:
//: `kernel` is the instantiation's function name (a __global__ function, not a
//: `const void*`: the launch syntax needs the symbol, and the driver attribute
//: calls take the address), `blocks` the number of CU rows.
template <typename Kernel, typename Globals>
inline bool launch_kernel(Kernel kernel, const Globals& globals, uint32_t blocks,
                          hipStream_t stream) {
    if (blocks == 0) return launch_detail::fail("no instruction rows to run");
    if (!reserve_dynamic_lds(reinterpret_cast<const void*>(kernel)))
        return false;
    kernel<<<blocks, config::kThreads, kLaunchBytes, stream>>>(globals);
    return true;
}

}  // namespace dmk::mk
