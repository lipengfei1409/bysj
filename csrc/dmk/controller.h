#pragma once

// Controller loop: instruction fetch, page permutation, semaphore construction
// and instruction publication.
//
// This is the DCU port of Megakernels' controller/controller.cuh. The four
// steps per instruction are:
//   0. wait for the previous use of this pipeline slot, invalidate its
//      per-instruction semaphores,
//   1. fetch the 32 instruction words from HBM into the slot,
//   2. permute physical page ids using the previous instruction's pid_order and
//      its op's release_lid() mapping,
//   3. construct the current op object and its dynamic semaphores, then publish
//      instruction_arrived.
//
// Ops are polymorphic (BaseOp). The controller constructs a temporary op object
// in local storage for the previous and current instruction and calls the
// virtual hooks directly; no op-type switch appears here.

#include "dmk/state.h"
#include "ops/op.h"
#include "ops/tensor.h"

#include <cassert>

namespace dmk::mk {

// The controller drives op objects polymorphically; those types belong to ops::.
namespace config = ops::config;
namespace lds = ops::lds;
namespace tensor = ::ops::tensor;
using ops::BaseOp;
using ops::InstructionState;
using ops::ring_advance;
using ops::ring_retreat;
using ops::make_op;
namespace controller {

// Prepare one instruction. The caller sets instruction_index/instruction_ring.
template <typename Globals, typename... Ops>
__device__ void prepare_instruction(const Globals& g, State& s) {
    const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
    const int ring = s.instruction_ring;
    InstructionState& slot = s.instruction_states[ring];

    // Per-thread, per-call op storage. Ops are trivially destructible, so the
    // same buffer can be reused for the previous-op query and the current-op
    // initialization.
    alignas(16) unsigned char op_storage[config::kOpStorageBytes];

    // Step 0: the slot was used kStages instructions ago; wait for its
    // consumers and retire the semaphores they may still be polling.
    if (s.instruction_index >= config::kStages) {
        const int previous_index = s.instruction_index - config::kStages;
        const uint32_t parity = (previous_index / config::kStages) & 1u;
        s.instruction_finished[ring].wait(parity);

        const int previous_semaphores = slot.num_semaphores;
        if (lane < previous_semaphores) slot.semaphores[lane].invalidate();
        __threadfence_block();
    }

    // Step 1: instruction fetch (32 words, one per lane).
    if (lane < config::kInstructionWidth) {
        const tensor::Shape<3> at{
            {static_cast<size_t>(blockIdx.x),
             static_cast<size_t>(s.instruction_index),
             static_cast<size_t>(lane)}};
        slot.instruction[lane] = g.instructions.data[g.instructions.offset(at)];
    }
    // Lanes 1..31 wrote words other lanes are about to read.
    __threadfence_block();

    // Step 2: establish this instruction's physical page order. The new slot q
    // receives the page that was logical page lid(q) of the previous
    // instruction; the previous op decides that permutation from its words.
    if (s.instruction_index == 0) {
        if (lane < config::kNumPages) slot.pid_order[lane] = lane;
    } else {
        const int previous_ring = ring_retreat<config::kStages>(ring);
        InstructionState& previous = s.instruction_states[previous_ring];
        if (lane < config::kNumPages) {
            BaseOp<Globals>* previous_op = make_op<Globals, Ops...>(
                previous.instruction[0], op_storage, previous.instruction);
            if (previous_op == nullptr) __builtin_trap();

            const int lid = previous_op->release_lid(lane);
            assert(lid >= 0 && lid < config::kNumPages);
            slot.pid_order[lane] = remap_page_order(previous.pid_order, lid);
        }
    }

    // Step 3: construct this instruction's op and dynamic semaphores.
    if (lane == 0) {
        BaseOp<Globals>* op = make_op<Globals, Ops...>(
            slot.instruction[0], op_storage, slot.instruction);
        if (op == nullptr) __builtin_trap();

        const int count = op->init_semaphores(s, g);
        assert(count >= 0 && count <= config::kDynamicSemaphores);
        slot.num_semaphores = count;
    }

    // Step 4: publish. The convergence point and fence order instruction words,
    // page order and semaphore initialization before the arrival.
    lds::wave_barrier();
    __threadfence_block();
    if (lane == 0) s.instruction_arrived[ring].arrive();
}

template <typename Globals, typename... Ops>
__device__ void main_loop(const Globals& g, State& s) {
    const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
    const int total = g.num_instructions;

    for (s.instruction_index = 0, s.instruction_ring = 0;
         s.instruction_index < total;
         ++s.instruction_index,
         s.instruction_ring =
             ring_advance<config::kStages>(s.instruction_ring)) {
        prepare_instruction<Globals, Ops...>(g, s);
    }

    // Drain the final uses of both pipeline slots.
    for (int i = 0; i < config::kStages; ++i) {
        const int index = total - config::kStages + i;
        if (index < 0) continue;
        const int ring = index % config::kStages;
        const uint32_t parity = (index / config::kStages) & 1u;
        s.instruction_finished[ring].wait(parity);

        const int count = s.instruction_states[ring].num_semaphores;
        if (lane < count)
            s.instruction_states[ring].semaphores[lane].invalidate();
        __threadfence_block();
    }
}

}  // namespace controller
}  // namespace dmk::mk
