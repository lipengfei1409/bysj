#pragma once

// DCU megakernel configuration.
//
// The split mirrors HazyResearch Megakernels' default_config:
//   - static LDS: instruction pipeline slots (instruction words, timings,
//     pid_order, per-instruction semaphores, scratch) plus the page and
//     instruction-state semaphores,
//   - dynamic LDS: NUM_PAGES fixed-size pages.
//
// The reference configuration is tuned for H100 (228 KiB / 16 KiB pages / 13
// pages). DCU has 64 KiB per CU, so this configuration uses 8 KiB pages:
//   static  10,416 B
//   dynamic 49,152 B (6 pages)
//   launch  49,280 B (dynamic + one 128 B alignment unit)
//   total   59,696 B, leaving ~5.8 KiB for compiler static scratch.
//
// Changing kPageSize/kNumPages/kConsumerWaves requires rechecking the
// static_asserts at the bottom of this header; the launch path also checks the
// runtime device limit before every launch.

#include "lds.h"

namespace dmk::mk {

struct Config {
    // Instruction pipeline (same shape as the reference implementation).
    static constexpr int kStages = 2;
    static constexpr int kStageBits = 1;
    static constexpr int kInstructionWidth = 32;  // 128 B per instruction
    static constexpr int kTimingWidth = 128;
    using instruction_t = int[kInstructionWidth];
    using timing_t = int[kTimingWidth];

    // Per-instruction semaphores (weights_arrived / outputs_finished / ...).
    static constexpr int kDynamicSemaphores = 16;

    // Roles: 4 consumer waves plus loader, storer and controller. The
    // Blackwell reference also has a launcher wave for async MMA/TMA issue;
    // DCU's synchronous DUMMA has no equivalent, so it is intentionally
    // omitted here (this also removes one worker wave from the retirement
    // count and 64 threads from the block).
    static constexpr int kConsumerWaves = 4;
    static constexpr int kServiceWaves = 3;  // loader, storer, controller
    static constexpr int kWaveSize = 64;
    static constexpr int kWorkerWaves = 2 + kConsumerWaves;
    static constexpr int kTotalWaves = kServiceWaves + kConsumerWaves;
    static constexpr int kThreads = kTotalWaves * kWaveSize;

    // LDS layout.
    static constexpr size_t kLdsBytesPerCu = 64 * 1024;
    static constexpr size_t kDynamicAlign = 128;
    static constexpr int kScratchBytes = 4096;
    static constexpr int kPageSize = 8192;
    static constexpr int kNumPages = 6;

    // Per-call local storage for a polymorphic op object (vptr + a few words).
    // This is stack/local memory, not LDS.
    static constexpr int kOpStorageBytes = 64;

    static constexpr bool kTimingRecordEnabled = false;

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
};

// One instruction pipeline slot, resident in static LDS.
template <typename Config>
struct alignas(128) InstructionState {
    int instruction[Config::kInstructionWidth];
    int timings[Config::kTimingWidth];
    int pid_order[Config::kNumPages];
    int num_semaphores;
    lds::Semaphore semaphores[Config::kDynamicSemaphores];
    int scratch[Config::kScratchBytes / sizeof(int)];
};

static_assert(sizeof(InstructionState<Config>) % 128 == 0,
              "instruction state stride must stay 128 B aligned");

inline constexpr size_t kStaticBytes =
    sizeof(InstructionState<Config>) * Config::kStages +
    sizeof(lds::Semaphore) *
        (2 * Config::kStages + 1 + Config::kNumPages * Config::kStageBits);
inline constexpr size_t kDynamicBytes =
    static_cast<size_t>(Config::kNumPages) * Config::kPageSize;
inline constexpr size_t kLaunchBytes = kDynamicBytes + Config::kDynamicAlign;

static_assert(kLaunchBytes <= Config::kLdsBytesPerCu,
              "dynamic LDS request exceeds the 64 KiB LDS block");
static_assert(kStaticBytes + kLaunchBytes <= Config::kLdsBytesPerCu,
              "static + dynamic LDS plan exceeds the 64 KiB LDS block");

}  // namespace dmk::mk
