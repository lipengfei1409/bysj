#pragma once

// DCU megakernel entry point and globals.
//
// The kernel structure follows HazyResearch Megakernels' mk_internal:
//   - static LDS holds the instruction pipeline slots, page semaphores and
//     instruction-state semaphores,
//   - dynamic LDS is a fixed-size page array,
//   - consumer waves run first, then loader / storer / controller.
//
// The demo globals below are intentionally small: an instruction tensor, one
// input and one output. A real model adds weight/activation pointers to Globals
// and adds ops to the kernel's op pack.

#include "config.h"
#include "controller.h"
#include "state.h"
#include "tensor.h"

namespace dmk::mk {

using tensor::MemorySpace;
using tensor::Shape;
using tensor::Tensor;

struct MkGlobals {
    // [blocks][num_instructions][kInstructionWidth]
    Tensor<const int, 3, MemorySpace::HBM> instructions;
    Tensor<const float, 1, MemorySpace::HBM> input;
    Tensor<float, 1, MemorySpace::HBM> output;
    int num_instructions = 0;

    // Optional tensors used by the concrete op examples. Ops that do not need
    // them ignore the empty views; the demo kernel's validity check only looks
    // at instructions/input/output.
    Tensor<const float, 2, MemorySpace::HBM> matrix_a{};
    Tensor<const float, 2, MemorySpace::HBM> matrix_b{};
    Tensor<float, 2, MemorySpace::HBM> matrix_c{};
    Tensor<const float, 2, MemorySpace::HBM> embedding_table{};
    Tensor<float, 2, MemorySpace::HBM> embedding_output{};

    DMK_HOST_DEVICE size_t elements() const { return input.shape.dims[0]; }

    DMK_HOST_DEVICE bool valid(uint32_t blocks) const {
        if (num_instructions <= 0) return false;
        if (instructions.data == nullptr) return false;
        if (instructions.shape.dims[0] < blocks ||
            instructions.shape.dims[1] < static_cast<size_t>(num_instructions) ||
            instructions.shape.dims[2] != Config::kInstructionWidth)
            return false;
        const size_t count = elements();
        if (output.shape.dims[0] != count) return false;
        if (count == 0) return true;
        return input.data != nullptr && output.data != nullptr &&
               input.layout.strides[0] == 1 && output.layout.strides[0] == 1;
    }
};

template <typename Config, typename Globals, typename... Ops>
__device__ inline void mk_internal(const Globals& g) {
    __shared__ alignas(128)
        InstructionState<Config> instruction_states[Config::kStages];
    __shared__ lds::Semaphore instruction_arrived[Config::kStages];
    __shared__ lds::Semaphore instruction_finished[Config::kStages];
    __shared__ lds::Semaphore
        page_finished[Config::kNumPages][Config::kStageBits];
    __shared__ lds::Semaphore semaphores_ready;

    extern __shared__ int dynamic_lds[];
    const std::uintptr_t raw =
        reinterpret_cast<std::uintptr_t>(dynamic_lds);
    const std::uintptr_t aligned =
        (raw + (Config::kDynamicAlign - 1)) &
        ~static_cast<std::uintptr_t>(Config::kDynamicAlign - 1);
    Page<Config>* pages = reinterpret_cast<Page<Config>*>(aligned);

    static_assert(
        sizeof(instruction_states) + sizeof(instruction_arrived) +
                sizeof(instruction_finished) + sizeof(page_finished) +
                sizeof(semaphores_ready) ==
            kStaticBytes,
        "static LDS accounting drifted from config.h");

    // Zero the timing slots and initialize the persistent semaphores.
    if (threadIdx.x < Config::kTimingWidth) {
#pragma unroll
        for (int stage = 0; stage < Config::kStages; ++stage)
            instruction_states[stage].timings[threadIdx.x] = 0;
    }
    if (threadIdx.x < Config::kStages) {
        instruction_arrived[threadIdx.x].init(1);
        instruction_finished[threadIdx.x].init(Config::kWorkerWaves);
    }
    if (threadIdx.x < Config::kNumPages) {
#pragma unroll
        for (int bit = 0; bit < Config::kStageBits; ++bit) {
            const uint32_t count =
                static_cast<uint32_t>(Config::kConsumerWaves << bit);
            page_finished[threadIdx.x][bit].init(count);
            // Complete phase 0 so pages are available for instruction 0.
            page_finished[threadIdx.x][bit].arrive(count);
        }
    }
    if (threadIdx.x == 0) {
        semaphores_ready.init(1);
        semaphores_ready.arrive(1);
    }
    __syncthreads();

    State<Config> state{instruction_states,
                        instruction_arrived,
                        instruction_finished,
                        &page_finished[0][0],
                        &semaphores_ready,
                        pages,
                        static_cast<uint64_t>(clock64())};

    const int wave = static_cast<int>(threadIdx.x / Config::kWaveSize);
    if (wave < Config::kConsumerWaves) {
        worker_loop<Role::Consumer, Config, Globals, Ops...>(g, state);
    } else if (wave == Config::kConsumerWaves) {
        worker_loop<Role::Loader, Config, Globals, Ops...>(g, state);
    } else if (wave == Config::kConsumerWaves + 1) {
        worker_loop<Role::Storer, Config, Globals, Ops...>(g, state);
    } else {
        controller::main_loop<Config, Globals, Ops...>(g, state);
    }
    __syncthreads();
}

template <typename Config, typename Globals, typename... Ops>
__global__ void mk(const Globals g) {
    mk_internal<Config, Globals, Ops...>(g);
}

// Launch the demo kernel (NoOp + ElementwiseOp). `blocks` CTAs each read their
// own instruction row from g.instructions[blockIdx.x]. Returns false and reports
// on stderr when the device or the LDS budget cannot support the launch.
bool launch_mk(const MkGlobals& g, uint32_t blocks,
               hipStream_t stream = nullptr);

}  // namespace dmk::mk
