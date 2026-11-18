#pragma once

// Reference page-based operator: y = scale * x + bias, one page per tile.
//
// Instruction words:
//   [0] opcode (1)
//   [1] tile index (tile * kTileElements is the HBM offset)
//   [2] scale, bit-cast from float
//   [3] bias, bit-cast from float
//   [4] valid element count in this tile (<= kTileElements)
//
// Page layout:
//   lid 0    the tile itself, loaded by the loader, scaled in place by the
//            consumer waves, written back by the storer
//   lid 1..N unused; the loader releases them immediately so their page phases
//            advance with the instruction stream
//
// Dynamic semaphores:
//   [0] data_ready   loader lane 0 arrives once; consumers wait
//   [1] data_done    each consumer wave arrives once; storer waits
//
// Page lifetime: the storer releases lid 0 after its copy, so the next
// instruction's loader can reuse the physical page.

#include "op.h"
#include "state.h"

namespace dmk::mk {

template <typename Config, typename Globals>
class ElementwiseOp final : public BaseOp<Config, Globals> {
public:
    static constexpr int opcode = 1;
    static constexpr int kTileElements = Config::kPageSize / sizeof(float);

    struct Instruction {
        int tile;
        float scale;
        float bias;
        int count;

        __device__ explicit Instruction(const int* words)
            : tile(words[1]),
              scale(__int_as_float(words[2])),
              bias(__int_as_float(words[3])),
              count(words[4]) {}
    };

    __device__ explicit ElementwiseOp(const int* instruction)
        : BaseOp<Config, Globals>(instruction) {}

    __device__ int release_lid(int query) const override { return query; }

    __device__ int init_semaphores(State<Config>& s,
                                   const Globals&) const override {
        s.semaphores()[0].init(1);
        s.semaphores()[1].init(Config::kConsumerWaves);
        return 2;
    }

    __device__ void loader(State<Config>& s,
                           const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        const int pid = s.pid(0);
        s.wait_page_ready(pid);

        float* page = static_cast<float*>(s.pages[pid].ptr());
        const size_t base = static_cast<size_t>(inst.tile) * kTileElements;
        for (int i = lane; i < kTileElements; i += Config::kWaveSize) {
            page[i] = i < inst.count ? g.input.data[base + i] : 0.0f;
        }
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[0].arrive();

        // Release pages this op does not use, mirroring NoOp's loader.
        if (lane >= 1 && lane < Config::kNumPages) {
            const int unused = s.pid(lane);
            s.wait_page_ready(unused);
            s.finish_page(unused, Config::kConsumerWaves);
        }
    }

    __device__ void consumer(State<Config>& s,
                             const Globals&) const override {
        const Instruction inst(this->instruction());
        const int wave = static_cast<int>(threadIdx.x / Config::kWaveSize);
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        const int pid = s.pid(0);

        s.semaphores()[0].wait(0);

        float* page = static_cast<float*>(s.pages[pid].ptr());
        for (int i = wave * Config::kWaveSize + lane; i < kTileElements;
             i += Config::kConsumerWaves * Config::kWaveSize) {
            page[i] = inst.scale * page[i] + inst.bias;
        }
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[1].arrive();
    }

    __device__ void storer(State<Config>& s,
                           const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        const int pid = s.pid(0);

        s.semaphores()[1].wait(0);

        const float* page = static_cast<const float*>(s.pages[pid].ptr());
        const size_t base = static_cast<size_t>(inst.tile) * kTileElements;
        for (int i = lane; i < inst.count; i += Config::kWaveSize) {
            g.output.data[base + i] = page[i];
        }
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.finish_page(pid, Config::kConsumerWaves);
    }
};

}  // namespace dmk::mk
