#pragma once

// Vector accessors for the HBM/LDS copies performed by op load()/store().
//
// This header is intentionally tiny. Which elements get copied, how the lanes
// of a wave are strided over them, what happens to a partial tile and whether
// a page has to be zero-filled are all decided in the op that owns the tile;
// only the two 16-byte accesses live here.
//
// A vector access is legal only when the address is aligned to alignof(Vec) and
// the span stays inside the allocation, which the caller checks with `aligned`:
//
//   constexpr size_t kVector = sizeof(float4) / sizeof(float);   // 4 elements
//   if (copy::aligned<float4>(dst) && copy::aligned<float4>(src)) {
//       const size_t vectors = count / kVector;
//       for (size_t v = lane; v < vectors; v += kWaveSize)
//           copy::store_vec<float4>(dst + v * kVector,
//                                   copy::load_vec<float4>(src + v * kVector));
//   }
//
// Lane i then touches element i * kVector, so a wave of 64 lanes covers 1 KiB
// of consecutive addresses per iteration.

#include <hip/hip_runtime.h>

#include <cstdint>

#ifndef DMK_HOST_DEVICE
#if defined(__HIPCC__)
#define DMK_HOST_DEVICE __host__ __device__
#else
#define DMK_HOST_DEVICE
#endif
#endif

namespace ops::copy {

template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool aligned(const T* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & (alignof(Vec) - 1)) == 0;
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline Vec load_vec(const T* pointer) {
    static_assert(sizeof(Vec) % sizeof(T) == 0,
                  "the vector must be a whole number of elements");
    return *reinterpret_cast<const Vec*>(pointer);
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline void store_vec(T* pointer, Vec value) {
    static_assert(sizeof(Vec) % sizeof(T) == 0,
                  "the vector must be a whole number of elements");
    *reinterpret_cast<Vec*>(pointer) = value;
}

}  // namespace ops::copy
