#pragma once

// Megakernel scheduling: the wave roles, the worker loop and the page-allocator
// permutation.
//
// The shared-memory contract itself (`Page`, `InstructionState` accessors and
// `State`) lives in ops/state.h, because operators are written against it and
// must not reach into dmk/.  What is left here is what only the runtime does:
// pick a role for a wave and drive the op hooks.

#include "ops/state.h"

#include <cstdint>

namespace dmk::mk {

namespace config = ops::config;
namespace lds = ops::lds;
using ops::BaseOp;
using ops::InstructionState;
using ops::Page;
using ops::State;

// New physical page slot q receives the page that was logical page lid of the
// previous instruction. Kept host-callable for permutation tests.
DMK_HOST_DEVICE constexpr int remap_page_order(const int* previous_order,
                                               int lid) {
    return previous_order[lid];
}

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
