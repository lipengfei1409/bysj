#pragma once

// Scalar dtype helpers shared by the dmk operators.
//
// DUMMA takes its inputs as float, __half or __hip_bfloat16 (DTK 26.04 DUMMA
// 手册, 表 3.1); every operator keeps its arithmetic in float and converts at
// the load/store boundary, so the kernels only need these two directions.

#include "ops/utils.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace ops::dtype {

template <typename T>
struct Traits;

template <>
struct Traits<float> {
    static DMK_HOST_DEVICE float to_float(float value) { return value; }
    static DMK_HOST_DEVICE float from_float(float value) { return value; }
};

template <>
struct Traits<__half> {
    static DMK_HOST_DEVICE float to_float(__half value) {
        return __half2float(value);
    }
    static DMK_HOST_DEVICE __half from_float(float value) {
        return __float2half_rn(value);
    }
};

template <>
struct Traits<__hip_bfloat16> {
    static DMK_HOST_DEVICE float to_float(__hip_bfloat16 value) {
        return __bfloat162float(value);
    }
    static DMK_HOST_DEVICE __hip_bfloat16 from_float(float value) {
        return __float2bfloat16(value);
    }
};

template <typename T>
DMK_HOST_DEVICE inline float to_float(T value) {
    return Traits<T>::to_float(value);
}

template <typename T>
DMK_HOST_DEVICE inline T from_float(float value) {
    return Traits<T>::from_float(value);
}

}  // namespace ops::dtype
