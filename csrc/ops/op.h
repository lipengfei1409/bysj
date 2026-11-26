#pragma once

// Operation interface for the page-based megakernel.
//
// Every op derives from BaseOp<Globals> and describes one instruction
// of work with three data hooks plus the two controller hooks:
//
//   load(s, g)           loader wave: stage the operator's *first* tile, so its
//                        slow start overlaps with the previous operator
//   run(s, g)            consumer waves: execute the whole operator -- walk its
//                        own tiles, load what else it needs, keep its own
//                        intra-op pipeline, leave the result in its pages
//   store(s, g)          storer wave: write the final result out and release
//                        the operator's pages
//   release_lid(query)   controller: page-allocator permutation for the
//                        *previous* instruction (logical page -> new slot)
//   init_semaphores(s, g) controller: construct this instruction's dynamic
//                        semaphores, return their count
//
// load/run/store know nothing about roles. The kernel decides which wave runs
// which hook (Role::Loader calls load, Role::Consumer calls run, Role::Storer
// calls store), so an op can be tested as a plain sequence load -> run -> store
// and a new op never has to think about wave numbering.
//
// Granularity: one instruction is one whole operator invocation, not one tile.
// run() owns the operator's tile loop (K steps, key tiles, rows, ...), so the
// tiling and any intra-op pipelining are written inside the op; the runtime only
// drives one prefetch (load) and one epilogue (store) per instruction. A stream
// that needs N operators contains N instructions.
//
// The op object carries only a pointer to the current 32-word instruction; the
// hooks receive the per-wave State and the kernel Globals, so an op does not
// capture thread-local state or a particular wave.
//
// Construction rules on HIP/DCU:
//   - A polymorphic object must be constructed in device code; a host-created
//     object cannot be passed to the kernel.
//   - Each controller/worker call constructs the op into a small local buffer
//     with placement new, calls one virtual hook, then discards it. This costs
//     one vtable store and one indirect call per role per instruction (the
//     data loops inside the op are of course still inlined).
//   - Ops must be trivially destructible, because the VM reuses the local
//     buffer every instruction and never calls a destructor.
//   - Ops are header-only/template classes. If a future build splits them into
//     separate translation units, HIP's -fgpu-rdc is required for device
//     function pointers/vtables across TUs.
//
// Adding an op requires:
//   1. a class with a static constexpr int opcode,
//   2. a __device__ constructor taking const int* instruction words,
//   3. the five overrides below,
//   4. adding the type to the kernel's op pack (the list passed to make_op /
//      mk_internal). Controller code does not need to change.
//
// An op's load/store is also the right place for its vectorization: the tile
// shape, alignment and tail behaviour are op-specific, so the copy loops live
// there and only the 16-byte load/store primitives come from copy.h.

#include "ops/config.h"
#include "ops/utils.h"

#include <new>
#include <type_traits>

namespace ops {

// The per-wave state a hook receives is declared in ops/state.h; it is part of
// the contract between the runtime and the operators, not of the runtime.
struct State;

template <typename Globals>
class BaseOp {
public:
    // Page allocator: for the previous instruction, return the logical page id
    // that the new physical slot `query` should inherit.
    __device__ virtual int release_lid(int query) const = 0;

    // Construct this instruction's dynamic semaphores; return how many were
    // initialized (<= config::kDynamicSemaphores).
    __device__ virtual int init_semaphores(State& state,
                                           const Globals& globals) const = 0;

    __device__ virtual void load(State& state,
                                 const Globals& globals) const = 0;
    __device__ virtual void run(State& state,
                                const Globals& globals) const = 0;
    __device__ virtual void store(State& state,
                                  const Globals& globals) const = 0;

protected:
    __device__ explicit BaseOp(const int* instruction)
        : instruction_(instruction) {}

    // No user-declared destructor: derived ops must stay trivially
    // destructible because the VM reuses the op buffer without destroying it.

    DMK_HOST_DEVICE const int* instruction() const { return instruction_; }

private:
    const int* instruction_;
};

// Device-side factory. Each call placement-news the op selected by `opcode`
// into `storage` and returns it as BaseOp*. `storage` must be at least
// config::kOpStorageBytes and 16-byte aligned.
namespace detail {

template <typename Globals>
__device__ BaseOp<Globals>* make_op_impl(int, void*, const int*) {
    __builtin_trap();
    return nullptr;
}

template <typename Globals, typename Op, typename... Rest>
__device__ BaseOp<Globals>* make_op_impl(int opcode, void* storage,
                                         const int* instruction) {
    if (opcode == Op::opcode) {
        static_assert(std::is_trivially_destructible<Op>::value,
                      "megakernel ops must be trivially destructible");
        static_assert(sizeof(Op) <= config::kOpStorageBytes,
                      "op exceeds config::kOpStorageBytes");
        static_assert(alignof(Op) <= 16,
                      "op alignment exceeds the op storage alignment");
        return ::new (storage) Op(instruction);
    }
    return make_op_impl<Globals, Rest...>(opcode, storage, instruction);
}

}  // namespace detail

template <typename Globals, typename... Ops>
__device__ BaseOp<Globals>* make_op(int opcode, void* storage,
                                    const int* instruction) {
    return detail::make_op_impl<Globals, Ops...>(opcode, storage,
                                                 instruction);
}

}  // namespace ops
