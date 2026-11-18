#include <future>
#include <hip/hip_runtime.h>
#include <util.h>
#include <config.h>
#include <cstdint>

namespace dmk {

constexpr uint32_t kRoleCount = 4;
constexpr uint32_t kMaxWaveSize = 64;

__device__ void controller() {

}

// Launch with blockDim.x = 4 * device.warpSize and one block per independent tile.
// This small data flow makes each role observable; model operators can replace it later.
template<const uint WARP_SZIE>
__global__ void megakernel() {
    const uint wid = threadIdx.x / WARP_SZIE;
    if (wid == 0) {
        // controller
    }
    // else if ()

}


void launch_mk() {

}

}  // namespace dmk
