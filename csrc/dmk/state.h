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

#include "ops/config.h"
#include "ops/op.h"


#include <cstdint>

namespace dmk::mk {

// The runtime drives the ops, so it names their types explicitly: the operator
// templates live in ops::, the per-wave state stays here in dmk::mk.  The
// aliases are scoped to this namespace so they cannot leak into a translation
// unit that includes the header.
namespace config = ops::config;
namespace lds = ops::lds;
namespace tensor = ops::tensor;
using ops::mk::BaseOp;
using ops::mk::InstructionState;

// num_pages >= 1; each page is exactly kPageSize bytes and is aligned so vector
// accesses and bank patterns stay predictable.
struct alignas(128) Page {
    // Raw storage: ops reinterpret it as float/float4/etc. A char array avoids
    // giving the page a fixed element type that vector accesses would alias.
    char data[config::kPageSize];

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

struct State {
    InstructionState* instruction_states;
    lds::Semaphore* instruction_arrived;
    lds::Semaphore* instruction_finished;
    lds::Semaphore* page_finished;  // [page][stage bit], flattened
    lds::Semaphore* semaphores_ready;
    lds::Event* events;  // [config::kEvents], persistent for the whole kernel
    Page* pages;

    int instruction_index;
    int instruction_ring;
    uint64_t start_clock;

    static_assert(sizeof(Page) == config::kPageSize,
                  "Page must be exactly one page of LDS");

    __device__ State(InstructionState* states,
                     lds::Semaphore* arrived, lds::Semaphore* finished,
                     lds::Semaphore* pages_finished,
                     lds::Semaphore* ready, lds::Event* event_array,
                     Page* page_array, uint64_t clock)
        : instruction_states(states),
          instruction_arrived(arrived),
          instruction_finished(finished),
          page_finished(pages_finished),
          semaphores_ready(ready),
          events(event_array),
          pages(page_array),
          instruction_index(0),
          instruction_ring(0),
          start_clock(clock) {}

    __device__ InstructionState& slot() const {
        return instruction_states[instruction_ring];
    }

    __device__ int (&instruction() const)[config::kInstructionWidth] {
        return slot().instruction;
    }

    __device__ int (&timings() const)[config::kTimingWidth] {
        return slot().timings;
    }

    __device__ int (&pid_order() const)[config::kNumPages] {
        return slot().pid_order;
    }

    __device__ lds::Semaphore* semaphores() const {
        return slot().semaphores;
    }

    __device__ int num_semaphores() const { return slot().num_semaphores; }

    __device__ void* scratch() const { return slot().scratch; }

    __device__ int pid(int lid) const { return pid_order()[lid]; }

    // Cross-instruction dependency counters: the producer arrives after its
    // data is visible, the consumer waits until the host-provided target has
    // been reached. Unlike s.semaphores(), they survive instruction slots.
    __device__ void arrive_event(int event, uint32_t amount = 1) const {
        events[event].arrive(amount);
    }

    __device__ void wait_event(int event, uint32_t target) const {
        events[event].wait(target);
    }

    __device__ uint32_t event_arrivals(int event) const {
        return events[event].arrivals();
    }

    __device__ void wait_page_ready(int pid) const {
#pragma unroll
        for (int bit = 0; bit < config::kStageBits; ++bit) {
            const uint32_t parity = (instruction_index >> bit) & 1u;
            page_finished[pid * config::kStageBits + bit].wait(parity);
        }
    }

    __device__ void finish_page(int pid, int count) const {
#pragma unroll
        for (int bit = 0; bit < config::kStageBits; ++bit) {
            page_finished[pid * config::kStageBits + bit].arrive(
                static_cast<uint32_t>(count));
        }
    }

    __device__ void warp_finish_page(int pid, int count) const {
        if (static_cast<int>(threadIdx.x % config::kWaveSize) == 0)
            finish_page(pid, count);
    }

    __device__ void await_instruction() const {
        instruction_arrived[instruction_ring].wait(
            (instruction_index / config::kStages) & 1u);
    }

    __device__ void next_instruction() {
        lds::wave_barrier();
        if (static_cast<int>(threadIdx.x % config::kWaveSize) == 0)
            instruction_finished[instruction_ring].arrive();
        ++instruction_index;
        instruction_ring = ring_advance<config::kStages>(instruction_ring);
    }

    __device__ void wait_semaphores_ready() const {
        semaphores_ready->wait(instruction_index & 1u);
    }

    __device__ void record(int event) const {
        if constexpr (config::kTimingRecordEnabled) {
            timings()[event] = static_cast<int>(clock64() - start_clock);
        }
    }
};

// Role tags selected at compile time by the kernel's wave dispatch. They are
// only a framework concept: a role picks which op hook runs, and the op itself
// only ever sees load/run/store.
enum class Role { Loader, Storer, Consumer };

// Every worker wave runs this loop. The op object is constructed in a small
// local buffer for each instruction and called through BaseOp's virtual
// interface; the storage is reused after the call because ops are required to
// be trivially destructible. One iteration is one tile: the hook it dispatches
// to handles exactly the tile its instruction words name.
template <Role R, typename Globals, typename... Ops>
__device__ inline void worker_loop(const Globals& g, State& s) {
    alignas(16) unsigned char op_storage[config::kOpStorageBytes];
    for (s.instruction_index = 0, s.instruction_ring = 0;
         s.instruction_index < g.num_instructions; s.next_instruction()) {
        s.await_instruction();

        BaseOp<Globals>* op = make_op<Globals, Ops...>(
            s.instruction()[0], op_storage, s.instruction());
        if (op == nullptr) __builtin_trap();

        if constexpr (R == Role::Loader)
            op->load(s, g);
        else if constexpr (R == Role::Storer)
            op->store(s, g);
        else
            op->run(s, g);
    }
}

}  // namespace dmk::mk
