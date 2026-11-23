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

// Keep the per-edge degree kernel (`kc`, 1 KB per edge) in global memory instead
// of staging it in shared memory.  At BE=8 that staging is 8 KB of the 24.2 KB
// budget, and 21.3 KB is what a third resident block costs.  Off by default so
// the verified binary is untouched.
#ifndef DPA4_KC_GLOBAL
#define DPA4_KC_GLOBAL 0
#endif
