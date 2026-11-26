#pragma once

// LDS synchronization and alignment primitives for the DCU megakernel.
//
// The layout follows the HazyResearch Megakernels design: the block has a
// static LDS region (instruction pipeline state, page semaphores, per-instruction
// semaphores) and a dynamic LDS region carved into fixed-size pages. Pages are
// recycled by the controller, which permutes logical page ids into physical page
// ids once per instruction; page semaphores tell the roles when a physical page
// is safe to reuse.
//
// DCU has no mbarrier, so `Semaphore` reimplements the mbarrier parity protocol
// with plain LDS words:
//   - `ticket` counts arrivals for the whole lifetime of the semaphore,
//   - `phase` is the parity of the current (incomplete) phase,
//   - `wait(parity)` returns once the current phase parity differs from
//     `parity`, i.e. the phase with that parity has completed.
//
// `init(expected); arrive(expected);` therefore completes phase 0, exactly like
// `mbarrier.init(N); mbarrier.arrive(N);` in the reference implementation.
//
// Hardware facts used here (DTK 26.04):
//   - 64 KiB LDS per CU, 32 banks x 4 bytes, 64-lane wavefronts on gfx926/928/
//     936/938.
//   - Dynamic LDS is `extern __shared__`, sized at launch; the kernel aligns the
//     base to 128 B so page 0 starts bank-aligned.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#ifndef DMK_HOST_DEVICE
#if defined(__HIPCC__)
#define DMK_HOST_DEVICE __host__ __device__
#else
#define DMK_HOST_DEVICE
#endif
#endif

namespace ops::lds {

constexpr size_t kDefaultAlignment = 128;

constexpr bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

constexpr size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Parity of the current phase after `ticket` cumulative arrivals. Kept
// host-callable so the phase protocol can be unit-tested without a device.
constexpr uint32_t phase_for_ticket(uint32_t ticket, uint32_t expected) {
    return expected == 0 ? 0u : ((ticket / expected) & 1u);
}

// One mbarrier-compatible phase semaphore. All members are 32-bit because LDS
// atomics are 32-bit on the supported DCU targets; wrapping after 2^32 arrivals
// is far beyond any realistic instruction stream.
struct alignas(16) Semaphore {
    // No default member initializers: the object must stay trivially
    // default-constructible so it can live in static __shared__ memory.
    // Call init() before the first use.
    uint32_t ticket;
    uint32_t phase;
    uint32_t expected;
    uint32_t reserved;

    __device__ void init(uint32_t expected_count) {
        ticket = 0;
        phase = 0;
        expected = expected_count;
        __threadfence_block();
    }

    __device__ void invalidate() {
        ticket = 0;
        phase = 0;
        expected = 0;
        __threadfence_block();
    }

    // Arrive with `amount`. The arrival that lands exactly on a phase boundary
    // publishes the new parity; all arrivals before it are ordered by the
    // release fence. This mirrors mbarrier's pending-count update without a
    // read-modify-write of a separate pending counter.
    __device__ void arrive(uint32_t amount = 1) {
        if (amount == 0) return;
        __threadfence_block();
        const uint32_t previous = atomicAdd(&ticket, amount);
        const uint32_t next = previous + amount;
        const uint32_t expected_count = expected;
        if (expected_count != 0 && next % expected_count == 0) {
            __threadfence_block();
            phase = phase_for_ticket(next, expected_count);
        }
    }

    __device__ bool try_wait(uint32_t parity) const {
        if (*reinterpret_cast<const volatile uint32_t*>(&phase) == parity)
            return false;
        __threadfence_block();
        return true;
    }

    __device__ void wait(uint32_t parity) const {
        while (*reinterpret_cast<const volatile uint32_t*>(&phase) == parity) {
        }
        __threadfence_block();
    }

    __device__ uint32_t current_phase() const {
        return *reinterpret_cast<const volatile uint32_t*>(&phase);
    }
};

// Monotonic counter for cross-instruction (and cross-op) dependencies.
//
// A Semaphore is phase-based and lives in an instruction slot, so it is only
// valid while that slot is alive; a dependency that spans instructions (op A's
// output feeding op B, or a tile feeding the next tile) needs a counter that
// never resets. The producer arrives, the consumer waits until the count
// reaches the target the host encoded for it, and any later instruction can
// wait the same event for a larger target.
struct alignas(4) Event {
    // No default member initializer: it lives in static __shared__ memory.
    uint32_t count;

    __device__ void init() {
        count = 0;
        __threadfence_block();
    }

    __device__ void arrive(uint32_t amount = 1) {
        if (amount == 0) return;
        __threadfence_block();
        atomicAdd(&count, amount);
    }

    __device__ uint32_t arrivals() const {
        return *reinterpret_cast<const volatile uint32_t*>(&count);
    }

    __device__ void wait(uint32_t target) const {
        while (arrivals() < target) {
        }
        __threadfence_block();
    }

    __device__ bool try_wait(uint32_t target) const {
        return arrivals() >= target;
    }
};

static_assert(sizeof(Event) == 4, "event must stay one 32-bit word");

static_assert(sizeof(Semaphore) == 16, "semaphore must stay 16-byte aligned");
static_assert(alignof(Semaphore) == 16, "semaphore must stay 16-byte aligned");

// Wave-level convergence point. gfx9 wavefronts execute in lockstep, but a
// divergent copy loop still needs an explicit join before one lane signals the
// other waves. Guarded for the host pass, where the AMDGPU builtin does not
// exist.
__device__ inline void wave_barrier() {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    __builtin_amdgcn_wave_barrier();
#endif
}

}  // namespace ops::lds
