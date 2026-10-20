#pragma once
#include "../../common/config.cuh"
#if defined(__HIPCC__)
namespace dcu::group {
// Every thread in the block must arrive. NOT a role-specific named barrier.
DCU_DEVICE void block_sync() { __syncthreads(); }
// Ordering only, not rendezvous. No inter-block scheduler is provided here.
DCU_DEVICE void block_fence() { __threadfence_block(); }
DCU_DEVICE void device_fence() { __threadfence(); }
} // namespace dcu::group
#endif
