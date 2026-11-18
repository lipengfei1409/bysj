#pragma once

// Page-based embedding gather for the megakernel VM.
//
// One instruction copies one table row into one output row:
//     out[dst_row, 0:count] = table[src_row, 0:count]
// It is a concrete BaseOp example with a single data page and no consumer-side
// compute (the consumer hook is intentionally empty; a real embedding op can
// add scaling, positional encoding, etc.).
//
// Instruction words:
//   [0] opcode (3)
//   [1] src_row
//   [2] dst_row
//   [3] count (elements, <= kPageElements)
//
// Pages:
//   lid 0  the row, lid 1..N unused and released by the loader
//
// Dynamic semaphores:
//   [0] data_ready   loader arrives once; storer waits
//
// Globals must provide:
//   Tensor<const float, 2, HBM> embedding_table;
//   Tensor<float, 2, HBM>       embedding_output;

#include "op.h"
#include "state.h"

namespace dmk::mk {

template <typename Config, typename Globals>
class EmbeddingOp final : public BaseOp<Config, Globals> {
public:
    static constexpr int opcode = 3;
    static constexpr int kPageElements = Config::kPageSize / sizeof(float);

    struct Instruction {
        int src_row;
        int dst_row;
        int count;

        __device__ explicit Instruction(const int* words)
            : src_row(words[1]), dst_row(words[2]), count(words[3]) {}
    };

    __device__ explicit EmbeddingOp(const int* instruction)
        : BaseOp<Config, Globals>(instruction) {}

    __device__ int release_lid(int query) const override { return query; }

    __device__ int init_semaphores(State<Config>& s,
                                   const Globals&) const override {
        s.semaphores()[0].init(1);
        return 1;
    }

    // Valid element count for this row, clamped to the page and both tensors.
    __device__ int valid_count(const Globals& g) const {
        const Instruction inst(this->instruction());
        int count = inst.count;
        if (count < 0) count = 0;
        if (count > kPageElements) count = kPageElements;
        if (inst.src_row < 0 ||
            inst.src_row >= static_cast<int>(g.embedding_table.shape.dims[0]))
            return 0;
        if (inst.dst_row < 0 ||
            inst.dst_row >= static_cast<int>(g.embedding_output.shape.dims[0]))
            return 0;
        const int src_width =
            static_cast<int>(g.embedding_table.shape.dims[1]);
        const int dst_width =
            static_cast<int>(g.embedding_output.shape.dims[1]);
        if (count > src_width) count = src_width;
        if (count > dst_width) count = dst_width;
        return count;
    }

    __device__ void loader(State<Config>& s,
                           const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        const int pid = s.pid(0);
        s.wait_page_ready(pid);

        float* page = static_cast<float*>(s.pages[pid].ptr());
        const int count = valid_count(g);
        const auto table = g.embedding_table;
        for (int i = lane; i < kPageElements; i += Config::kWaveSize) {
            page[i] = i < count
                          ? table.data[static_cast<size_t>(inst.src_row) *
                                           table.layout.strides[0] +
                                       static_cast<size_t>(i) *
                                           table.layout.strides[1]]
                          : 0.0f;
        }
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[0].arrive();

        if (lane >= 1 && lane < Config::kNumPages) {
            const int unused = s.pid(lane);
            s.wait_page_ready(unused);
            s.finish_page(unused, Config::kConsumerWaves);
        }
    }

    __device__ void consumer(State<Config>&,
                             const Globals&) const override {}

    __device__ void storer(State<Config>& s,
                           const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        const int pid = s.pid(0);

        s.semaphores()[0].wait(0);

        const float* page = static_cast<const float*>(s.pages[pid].ptr());
        const int count = valid_count(g);
        auto out = g.embedding_output;
        for (int i = lane; i < count; i += Config::kWaveSize) {
            out.data[static_cast<size_t>(inst.dst_row) *
                         out.layout.strides[0] +
                     static_cast<size_t>(i) * out.layout.strides[1]] = page[i];
        }
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.finish_page(pid, Config::kConsumerWaves);
    }
};

}  // namespace dmk::mk
