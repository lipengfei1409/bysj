#pragma once

// Megakernel VM state and role dispatch.
//
// This is the DCU port of Megakernels' state<config> (util.cuh):
//   - static LDS: InstructionState[kStages], instruction_arrived/finished,
//     page_finished[NUM_PAGES][STAGE_BITS], semaphores_ready,
//   - dynamic LDS: Page[NUM_PAGES].
//
// Pages are named by logical id (lid) inside an op and translated to a physical
// page id (pid) by the controller's per-instruction pid_order permutation. All
// role code goes through pid(lid); nothing else indexes the page array.

#include "config.h"
#include "op.h"

#include <cstdint>

namespace dmk::mk {

// num_pages >= 1; each page is exactly kPageSize bytes and is aligned so vector
// accesses and bank patterns stay predictable.
template <typename Config>
struct alignas(128) Page {
    // Raw storage: ops reinterpret it as float/float4/etc. A char array avoids
    // giving the page a fixed element type that vector accesses would alias.
    char data[Config::kPageSize];

    __device__ void* ptr(size_t byte_offset = 0) { return data + byte_offset; }
    __device__ const void* ptr(size_t byte_offset = 0) const {
        return data + byte_offset;
    }
};

template <int N>
__device__ __host__ constexpr int ring_advance(int ring, int distance = 1) {
    return (ring + distance) % N;
}

template <int N>
__device__ __host__ constexpr int ring_retreat(int ring, int distance = 1) {
    return (ring + 16 * N - distance) % N;
}

// New physical page slot q receives the page that was logical page lid of the
// previous instruction. Kept host-callable for permutation tests.
DMK_HOST_DEVICE constexpr int remap_page_order(const int* previous_order,
                                               int lid) {
    return previous_order[lid];
}

template <typename Config>
struct State {
    InstructionState<Config>* instruction_states;
    lds::Semaphore* instruction_arrived;
    lds::Semaphore* instruction_finished;
    lds::Semaphore* page_finished;  // [page][stage bit], flattened
    lds::Semaphore* semaphores_ready;
    Page<Config>* pages;

    int instruction_index;
    int instruction_ring;
    uint64_t start_clock;

    static_assert(sizeof(Page<Config>) == Config::kPageSize,
                  "Page must be exactly one page of LDS");

    __device__ State(InstructionState<Config>* states,
                     lds::Semaphore* arrived, lds::Semaphore* finished,
                     lds::Semaphore* pages_finished,
                     lds::Semaphore* ready, Page<Config>* page_array,
                     uint64_t clock)
        : instruction_states(states),
          instruction_arrived(arrived),
          instruction_finished(finished),
          page_finished(pages_finished),
          semaphores_ready(ready),
          pages(page_array),
          instruction_index(0),
          instruction_ring(0),
          start_clock(clock) {}

    __device__ InstructionState<Config>& slot() const {
        return instruction_states[instruction_ring];
    }

    __device__ int (&instruction() const)[Config::kInstructionWidth] {
        return slot().instruction;
    }

    __device__ int (&timings() const)[Config::kTimingWidth] {
        return slot().timings;
    }

    __device__ int (&pid_order() const)[Config::kNumPages] {
        return slot().pid_order;
    }

    __device__ lds::Semaphore* semaphores() const {
        return slot().semaphores;
    }

    __device__ int num_semaphores() const { return slot().num_semaphores; }

    __device__ void* scratch() const { return slot().scratch; }

    __device__ int pid(int lid) const { return pid_order()[lid]; }

    __device__ void wait_page_ready(int pid) const {
#pragma unroll
        for (int bit = 0; bit < Config::kStageBits; ++bit) {
            const uint32_t parity = (instruction_index >> bit) & 1u;
            page_finished[pid * Config::kStageBits + bit].wait(parity);
        }
    }

    __device__ void finish_page(int pid, int count) const {
#pragma unroll
        for (int bit = 0; bit < Config::kStageBits; ++bit) {
            page_finished[pid * Config::kStageBits + bit].arrive(
                static_cast<uint32_t>(count));
        }
    }

    __device__ void warp_finish_page(int pid, int count) const {
        if (static_cast<int>(threadIdx.x % Config::kWaveSize) == 0)
            finish_page(pid, count);
    }

    __device__ void await_instruction() const {
        instruction_arrived[instruction_ring].wait(
            (instruction_index / Config::kStages) & 1u);
    }

    __device__ void next_instruction() {
        lds::wave_barrier();
        if (static_cast<int>(threadIdx.x % Config::kWaveSize) == 0)
            instruction_finished[instruction_ring].arrive();
        ++instruction_index;
        instruction_ring = ring_advance<Config::kStages>(instruction_ring);
    }

    __device__ void wait_semaphores_ready() const {
        semaphores_ready->wait(instruction_index & 1u);
    }

    __device__ void record(int event) const {
        if constexpr (Config::kTimingRecordEnabled) {
            timings()[event] = static_cast<int>(clock64() - start_clock);
        }
    }
};

// Role tags selected at compile time by the kernel's wave dispatch.
enum class Role { Loader, Storer, Consumer };

// Every worker wave runs this loop. The op object is constructed in a small
// local buffer for each instruction and called through BaseOp's virtual
// interface; the storage is reused after the call because ops are required to
// be trivially destructible.
template <Role R, typename Config, typename Globals, typename... Ops>
__device__ inline void worker_loop(const Globals& g, State<Config>& s) {
    alignas(16) unsigned char op_storage[Config::kOpStorageBytes];
    for (s.instruction_index = 0, s.instruction_ring = 0;
         s.instruction_index < g.num_instructions; s.next_instruction()) {
        s.await_instruction();

        BaseOp<Config, Globals>* op = make_op<Config, Globals, Ops...>(
            s.instruction()[0], op_storage, s.instruction());
        if (op == nullptr) __builtin_trap();

        if constexpr (R == Role::Loader)
            op->loader(s, g);
        else if constexpr (R == Role::Storer)
            op->storer(s, g);
        else
            op->consumer(s, g);
    }
}

}  // namespace dmk::mk
