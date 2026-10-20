#pragma once
#include "../../common/config.cuh"
#if defined(__HIPCC__)
namespace dcu::group {
DCU_DEVICE unsigned thread_rank() {
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}
DCU_DEVICE unsigned block_threads() { return blockDim.x * blockDim.y * blockDim.z; }
DCU_DEVICE unsigned lane_id() { return thread_rank() % warpSize; }
DCU_DEVICE unsigned wave_id() { return thread_rank() / warpSize; }

// Full, converged hardware wave ONLY; all lanes must participate.
// Do not invoke these in a divergent branch or a partial last wave.
DCU_DEVICE float shuffle(float value, int source_lane) {
    return __shfl(value, source_lane, warpSize);
}
DCU_DEVICE float shuffle_down(float value, unsigned delta) {
    return __shfl_down(value, delta, warpSize);
}
DCU_DEVICE float wave_sum(float value) {
    for (unsigned offset = warpSize / 2; offset != 0; offset >>= 1)
        value += shuffle_down(value, offset);
    return shuffle(value, 0); // Result is broadcast to EVERY lane.
}
DCU_DEVICE float wave_max(float value) {
    for (unsigned offset = warpSize / 2; offset != 0; offset >>= 1)
        value = ::fmaxf(value, shuffle_down(value, offset));
    return shuffle(value, 0);
}
} // namespace dcu::group
#endif
