#pragma once

// Page-based embedding gather for the megakernel VM.
//
// One instruction is one tile, and the tile is one table row:
//     out[dst_row, 0:count] = table[src_row, 0:count]
// It is a concrete BaseOp example with a single data page and no arithmetic, so
// run() is empty and the work is entirely load() -> store().
//
// Instruction words:
//   [0] opcode (3)
//   [1] src_row
//   [2] dst_row
//   [3] count (elements, <= kPageElements)
//
// Pages:
//   lid 0  the row, lid 1..N unused and released by load()
//
// Dynamic semaphores:
//   [0] data_ready   load arrives once; store waits
//
// A row is contiguous only when its column stride is 1; a strided table row is
// gathered element by element, so load()/store() each carry both paths.
//
// Globals must provide:
//   Tensor<const float, 2, HBM> embedding_table;
//   Tensor<float, 2, HBM>       embedding_output;

#include "copy.h"
#include "op.h"
#include "state.h"

namespace dmk::mk {

template <typename Globals>
class EmbeddingOp final : public BaseOp<Globals> {
public:
    static constexpr int opcode = 3;
    static constexpr int kPageElements = config::kPageSize / sizeof(float);

    static constexpr size_t kVector = sizeof(float4) / sizeof(float);
    static constexpr size_t kWaveStride = config::kWaveSize * kVector;

    struct Instruction {
        int src_row;
        int dst_row;
        int count;

        __device__ explicit Instruction(const int* words)
            : src_row(words[1]), dst_row(words[2]), count(words[3]) {}
    };

    __device__ explicit EmbeddingOp(const int* instruction)
        : BaseOp<Globals>(instruction) {}

    __device__ int release_lid(int query) const override { return query; }

    __device__ int init_semaphores(State& s,
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

    __device__ void load(State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int pid = s.pid(0);
        s.wait_page_ready(pid);

        float* page = static_cast<float*>(s.pages[pid].ptr());
        const auto table = g.embedding_table;
        const size_t row =
            static_cast<size_t>(inst.src_row) * table.layout.strides[0];
        const int count = valid_count(g);

        if (table.layout.strides[1] != 1) {
            // Row gather through the column stride; no vector access applies.
            for (int i = lane; i < count; i += config::kWaveSize) {
                page[i] = table.data[row + static_cast<size_t>(i) *
                                                table.layout.strides[1]];
            }
        } else {
            const float* src = table.data + row;
            size_t vector_end = 0;
            if (copy::aligned<float4>(src) && copy::aligned<float4>(page)) {
                vector_end = static_cast<size_t>(count) / kVector * kVector;
                for (size_t element = static_cast<size_t>(lane) * kVector;
                     element < vector_end; element += kWaveStride) {
                    copy::store_vec<float4>(
                        page + element,
                        copy::load_vec<float4>(src + element));
                }
            }
            for (size_t element = vector_end + lane;
                 element < static_cast<size_t>(count);
                 element += config::kWaveSize)
                page[element] = src[element];
        }
        for (int i = count + lane; i < kPageElements; i += config::kWaveSize)
            page[i] = 0.0f;

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[0].arrive();

        if (lane >= 1 && lane < config::kNumPages) {
            const int unused = s.pid(lane);
            s.wait_page_ready(unused);
            s.finish_page(unused, config::kConsumerWaves);
        }
    }

    __device__ void run(State&, const Globals&) const override {}

    __device__ void store(State& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        const int pid = s.pid(0);

        s.semaphores()[0].wait(0);

        const float* page = static_cast<const float*>(s.pages[pid].ptr());
        auto out = g.embedding_output;
        const size_t row =
            static_cast<size_t>(inst.dst_row) * out.layout.strides[0];
        const int count = valid_count(g);

        if (out.layout.strides[1] != 1) {
            for (int i = lane; i < count; i += config::kWaveSize) {
                out.data[row + static_cast<size_t>(i) * out.layout.strides[1]] =
                    page[i];
            }
        } else {
            float* dst = out.data + row;
            size_t vector_end = 0;
            if (copy::aligned<float4>(dst) && copy::aligned<float4>(page)) {
                vector_end = static_cast<size_t>(count) / kVector * kVector;
                for (size_t element = static_cast<size_t>(lane) * kVector;
                     element < vector_end; element += kWaveStride) {
                    copy::store_vec<float4>(
                        dst + element,
                        copy::load_vec<float4>(page + element));
                }
            }
            for (size_t element = vector_end + lane;
                 element < static_cast<size_t>(count);
                 element += config::kWaveSize)
                dst[element] = page[element];
        }

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.finish_page(pid, config::kConsumerWaves);
    }
};

}  // namespace dmk::mk
