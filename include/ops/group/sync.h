#pragma once
#include "../../common/config.h"
#if defined(__HIPCC__)
namespace dmk::group {
// Every thread in the block must arrive. NOT a role-specific named barrier.
DCU_DEVICE void block_sync() { __syncthreads(); }
// Ordering only, not rendezvous. No inter-block scheduler is provided here.
DCU_DEVICE void block_fence() { __threadfence_block(); }
DCU_DEVICE void device_fence() { __threadfence(); }
} // namespace dmk::group
#endif
