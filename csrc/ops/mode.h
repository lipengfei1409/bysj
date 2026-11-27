#pragma once

// Which of the two implementations an operator header compiles into.
//
// Every operator in this library has two ways in:
//
//   OPS_MODE_PERSISTENT  the BaseOp subclass, driven by the megakernel VM in
//                        dmk/ (pages, semaphores, one instruction per call)
//   OPS_MODE_ORDINARY    a plain __global__ kernel plus an inline launcher,
//                        launched like any other operator in csrc/
//
// Both compile from the same header and share the operator's arithmetic, so the
// two paths cannot drift, and the ordinary one is the A/B baseline for the
// persistent one.
//
// The mode is a *compile-time* switch, because the two modes do not agree on
// anything that matters at runtime: the persistent form needs `State&` and the
// page contract, the ordinary form needs a grid/block geometry.  A
// translation unit selects its mode before including the operator header:
//
//     #define OPS_MODE OPS_MODE_ORDINARY
//     #include "ops/so2_conv_op.h"
//
// or, for a whole build, add `-DOPS_MODE=OPS_MODE_ORDINARY`.  The default is
// persistent, so a translation unit that includes an op header without thinking
// about it gets the megakernel form it is almost certainly part of.
//
// Only one definition site: the mode is a property of the *build*, and mixing
// both in one translation unit would give the same class two definitions.

#define OPS_MODE_PERSISTENT 0
#define OPS_MODE_ORDINARY 1

#ifndef OPS_MODE
#define OPS_MODE OPS_MODE_PERSISTENT
#endif

#if OPS_MODE != OPS_MODE_PERSISTENT && OPS_MODE != OPS_MODE_ORDINARY
#error "OPS_MODE must be OPS_MODE_PERSISTENT (0) or OPS_MODE_ORDINARY (1)"
#endif

namespace ops {

//: True when this translation unit is building the persistent (megakernel) form.
inline constexpr bool kPersistentMode = (OPS_MODE == OPS_MODE_PERSISTENT);

}  // namespace ops

// `OPS_MODE_PERSISTENT` is used in `#if`, so it has to stay a copy-pasteable
// literal; the constants above exist for the code paths that can be `if constexpr`.
#if OPS_MODE == OPS_MODE_PERSISTENT
#define OPS_IF_PERSISTENT(...) __VA_ARGS__
#define OPS_IF_ORDINARY(...)
#else
#define OPS_IF_PERSISTENT(...)
#define OPS_IF_ORDINARY(...) __VA_ARGS__
#endif
