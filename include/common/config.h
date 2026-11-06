#pragma once

// No Torch/CUDA dependency. HIP-only operations are absent in host builds.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_NVIDIA__)
#error "DCU instruction headers require the HIP AMD/Hygon backend"
#endif
#define DCU_HD __host__ __device__ __forceinline__
#define DCU_DEVICE __device__ __forceinline__
#else
#define DCU_HD inline
#define DCU_DEVICE inline
#endif

#ifndef DPA4_ENABLE_DUMMA
#define DPA4_ENABLE_DUMMA 0
#endif

namespace dmk {
enum class architecture { unknown, gfx926, gfx928, gfx936, gfx938 };

// The host pass cannot infer the GPU architecture. Never guess from CPU ISA.
#if defined(__gfx926__)
inline constexpr architecture compiled_arch = architecture::gfx926;
#elif defined(__gfx928__)
inline constexpr architecture compiled_arch = architecture::gfx928;
#elif defined(__gfx936__)
inline constexpr architecture compiled_arch = architecture::gfx936;
#elif defined(__gfx938__)
inline constexpr architecture compiled_arch = architecture::gfx938;
#else
inline constexpr architecture compiled_arch = architecture::unknown;
#endif

// The first GEMM uses true FP32 DUMMA with K=4, not TF32.
template<architecture Arch, typename InputT, int K>
inline constexpr bool dumma_supported = false;
template<architecture Arch>
inline constexpr bool dumma_supported<Arch, float, 4> =
    Arch == architecture::gfx926 || Arch == architecture::gfx928 ||
    Arch == architecture::gfx936 || Arch == architecture::gfx938;
} // namespace dmk
