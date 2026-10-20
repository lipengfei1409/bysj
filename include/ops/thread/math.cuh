#pragma once
#include "../../common/config.cuh"
#include <cmath>

namespace dcu::thread {
DCU_HD float fma(float a, float b, float c) {
#if defined(__HIP_DEVICE_COMPILE__)
    return ::fmaf(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
DCU_HD double fma(double a, double b, double c) {
#if defined(__HIP_DEVICE_COMPILE__)
    return ::fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
} // namespace dcu::thread
