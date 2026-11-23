// HIP compatibility shims for the translated fused SO(2) convolution.
//
// The CUDA source is written for a 32-lane warp: every output column is
// ``j * 32 + lane`` and ``kWarp == 32`` splits the channel width.  A ROCm
// wavefront is 64 lanes, so the port keeps the 32-lane unit and runs two of
// them per wave.  That choice is what the shuffle width below encodes: with
// ``width == 32`` HIP shuffles inside each 32-lane segment, which is exactly
// the CUDA semantics, at the cost of running the kernel at half wavefront
// occupancy.  Widening to 64 lanes is an algorithm change, not a translation.
#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>

/// Warp-level barrier for divergent regions.  ``__syncthreads()`` would
/// deadlock where the CUDA source calls ``__syncwarp()`` inside ``if (active)``;
/// an AMD wave barrier is the conservative superset (it also orders the other
/// 32-lane half, which is harmless).
__device__ __forceinline__ void dpa4_wave_barrier() {
  __builtin_amdgcn_wave_barrier();
}
#endif
