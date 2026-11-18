#pragma once
#include <hip/hip_runtime.h>
#include <du_mma.h>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <utility>

// Torch is used only by host-side Tensor adapters.
#ifndef __HIP_DEVICE_COMPILE__
#include <ATen/ATen.h>
#endif
using namespace std;

// HIP error checking
#define CHECK_HIP(call) {       \
    hipError_t err = call;      \
    if (err != hipSuccess) {    \
        std::cerr << "HIP Error: " << hipGetErrorString(err) <<" at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);     \
    } \
}\

#if defined(__HIPCC__)
#define DMK_HOST_DEVICE __host__ __device__
#else
#define DMK_HOST_DEVICE
#endif
