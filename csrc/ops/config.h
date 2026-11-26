#pragma once

// DCU megakernel configuration.
//
// The split mirrors HazyResearch Megakernels' default_config:
//   - static LDS: instruction pipeline slots (instruction words, timings,
//     pid_order, per-instruction semaphores, scratch) plus the page and
//     instruction-state semaphores,
//   - dynamic LDS: kNumPages fixed-size pages.
//
// The reference configuration is tuned for H100 (228 KiB / 16 KiB pages / 13
// pages). DCU has 64 KiB per CU, so this configuration uses 8 KiB pages:
//   static  10,448 B
//   dynamic 49,152 B (6 pages)
//   launch  49,280 B (dynamic + one 128 B alignment unit)
//   total   59,728 B, leaving ~5.8 KiB for compiler static scratch.
//
// The tunables are plain constants in ops::config instead of a Config struct
// template parameter: the DCU has one fixed LDS budget, so every page, state
// and op sees the same numbers, and `config::kNumPages` reads shorter than
// threading a `Config` type through every template. The derived LDS budget
// stays in ops because it depends on sizeof(InstructionState).
//
// Changing kPageSize/kNumPages/kConsumerWaves requires rechecking the
// static_asserts below; the launch path also checks the runtime device limit
// before every launch.

#include "ops/lds.h"

#include <cstddef>

namespace ops::config {

// Instruction pipeline (same shape as the reference implementation).
inline constexpr int kStages = 2;
inline constexpr int kStageBits = 1;
inline constexpr int kInstructionWidth = 32;  // 128 B per instruction
inline constexpr int kTimingWidth = 128;

// Per-instruction semaphores (weights_arrived / outputs_finished / ...).
inline constexpr int kDynamicSemaphores = 16;

// Persistent counters for dependencies that span instructions (an op's output
// feeding the next op, or the next tile of the same op). They live in static
// LDS for the whole kernel and never reset, unlike the per-instruction
// semaphores.
inline constexpr int kEvents = 8;

// Roles: 4 consumer waves plus loader, storer and controller. The Blackwell
// reference also has a launcher wave for async MMA/TMA issue; DCU's synchronous
// DUMMA has no equivalent, so it is intentionally omitted here (this also
// removes one worker wave from the retirement count and 64 threads from the
// block).
inline constexpr int kConsumerWaves = 4;
inline constexpr int kServiceWaves = 3;  // loader, storer, controller
inline constexpr int kWaveSize = 64;
inline constexpr int kWorkerWaves = 2 + kConsumerWaves;
inline constexpr int kTotalWaves = kServiceWaves + kConsumerWaves;
inline constexpr int kThreads = kTotalWaves * kWaveSize;

// LDS layout.
inline constexpr size_t kLdsBytesPerCu = 64 * 1024;
inline constexpr size_t kDynamicAlign = 128;
inline constexpr int kScratchBytes = 4096;
inline constexpr int kPageSize = 8192;
inline constexpr int kNumPages = 6;

// Per-call local storage for a polymorphic op object (vptr + a few words).
// This is stack/local memory, not LDS.
inline constexpr int kOpStorageBytes = 64;

inline constexpr bool kTimingRecordEnabled = false;

static_assert((1 << kStageBits) == kStages,
              "stage bits must address the instruction pipeline");
static_assert(kNumPages <= 32,
              "page ids are assigned with one lane per page");
static_assert(kThreads <= 1024,
              "all role waves must fit one DCU workgroup");
static_assert(kStageBits == 1,
              "finish_page count convention currently assumes one stage bit");
static_assert(kServiceWaves == 3,
              "roles are loader, storer and controller");
static_assert(kWorkerWaves == kTotalWaves - 1,
              "every wave except the controller is a worker");
static_assert(kPageSize % kDynamicAlign == 0,
              "pages must start on the dynamic LDS alignment");

}  // namespace ops::config

namespace ops {

// One instruction pipeline slot, resident in static LDS.
struct alignas(128) InstructionState {
    int instruction[config::kInstructionWidth];
    int timings[config::kTimingWidth];
    int pid_order[config::kNumPages];
    int num_semaphores;
    lds::Semaphore semaphores[config::kDynamicSemaphores];
    int scratch[config::kScratchBytes / sizeof(int)];
};

static_assert(sizeof(InstructionState) % 128 == 0,
              "instruction state stride must stay 128 B aligned");

inline constexpr size_t kStaticBytes =
    sizeof(InstructionState) * config::kStages +
    sizeof(lds::Semaphore) *
        (2 * config::kStages + 1 + config::kNumPages * config::kStageBits) +
    sizeof(lds::Event) * config::kEvents;
inline constexpr size_t kDynamicBytes =
    static_cast<size_t>(config::kNumPages) * config::kPageSize;
inline constexpr size_t kLaunchBytes = kDynamicBytes + config::kDynamicAlign;

static_assert(kLaunchBytes <= config::kLdsBytesPerCu,
              "dynamic LDS request exceeds the 64 KiB LDS block");
static_assert(kStaticBytes + kLaunchBytes <= config::kLdsBytesPerCu,
              "static + dynamic LDS plan exceeds the 64 KiB LDS block");

}  // namespace ops
