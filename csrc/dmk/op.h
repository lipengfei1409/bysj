#pragma once

// Operation interface for the page-based megakernel.
//
// Every op derives from BaseOp<Config, Globals> and implements the five hooks
// the VM needs:
//
//   release_lid(query)        controller: page-allocator permutation for the
//                             *previous* instruction (logical page -> new slot)
//   init_semaphores(s, g)     controller: construct this instruction's dynamic
//                             semaphores, return their count
//   loader/consumer/storer    worker waves: run this role for the current
//                             instruction
//
// The op object carries only a pointer to the current 32-word instruction; the
// role methods receive the per-wave State and the kernel Globals, so an op does
// not capture thread-local state or a particular wave.
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

#include "utils.h"

#include <new>
#include <type_traits>

namespace dmk::mk {

template <typename Config>
struct State;

template <typename Config, typename Globals>
class BaseOp {
public:
    // Page allocator: for the previous instruction, return the logical page id
    // that the new physical slot `query` should inherit.
    __device__ virtual int release_lid(int query) const = 0;

    // Construct this instruction's dynamic semaphores; return how many were
    // initialized (<= Config::kDynamicSemaphores).
    __device__ virtual int init_semaphores(State<Config>& state,
                                           const Globals& globals) const = 0;

    __device__ virtual void loader(State<Config>& state,
                                   const Globals& globals) const = 0;
    __device__ virtual void consumer(State<Config>& state,
                                     const Globals& globals) const = 0;
    __device__ virtual void storer(State<Config>& state,
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
// Config::kOpStorageBytes and 16-byte aligned.
namespace detail {

template <typename Config, typename Globals>
__device__ BaseOp<Config, Globals>* make_op_impl(int, void*, const int*) {
    __builtin_trap();
    return nullptr;
}

template <typename Config, typename Globals, typename Op, typename... Rest>
__device__ BaseOp<Config, Globals>* make_op_impl(int opcode, void* storage,
                                                 const int* instruction) {
    if (opcode == Op::opcode) {
        static_assert(std::is_trivially_destructible<Op>::value,
                      "megakernel ops must be trivially destructible");
        static_assert(sizeof(Op) <= Config::kOpStorageBytes,
                      "op exceeds Config::kOpStorageBytes");
        static_assert(alignof(Op) <= 16,
                      "op alignment exceeds the op storage alignment");
        return ::new (storage) Op(instruction);
    }
    return make_op_impl<Config, Globals, Rest...>(opcode, storage,
                                                  instruction);
}

}  // namespace detail

template <typename Config, typename Globals, typename... Ops>
__device__ BaseOp<Config, Globals>* make_op(int opcode, void* storage,
                                            const int* instruction) {
    return detail::make_op_impl<Config, Globals, Ops...>(opcode, storage,
                                                         instruction);
}

}  // namespace dmk::mk
