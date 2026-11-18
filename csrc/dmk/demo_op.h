#pragma once

// Reference page-based operator: y = scale * x + bias, one page per tile.
//
// One instruction is one tile: load() fills page lid 0 from HBM, run() scales
// that page in place and store() writes it back. If the tile is partial, the
// same three hooks still cover exactly this one tile (the tail is zero-filled
// and only `count` elements are written back).
//
// Instruction words:
//   [0] opcode (1)
//   [1] tile index (tile * kTileElements is the HBM offset)
//   [2] scale, bit-cast from float
//   [3] bias, bit-cast from float
//   [4] valid element count in this tile (<= kTileElements)
//
// Page layout:
//   lid 0    the tile itself: load() brings it in from HBM, run() scales it in
//            place, store() writes it back
//   lid 1..N unused; load() releases them immediately so their page phases
//            advance with the instruction stream
//
// Dynamic semaphores:
//   [0] data_ready   load lane 0 arrives once; run waits
//   [1] data_done    each consumer wave arrives once; store waits
//
// load() and store() own their copy loops instead of calling a generic helper:
// the tile is a contiguous page, so lane i takes the i-th 16 B chunk, and the
// partial-tile tail is handled right here.

#include "copy.h"
#include "op.h"
#include "state.h"

namespace dmk::mk {

template <typename Config, typename Globals>
class ElementwiseOp final : public BaseOp<Config, Globals> {
public:
    static constexpr int opcode = 1;
    static constexpr int kTileElements = Config::kPageSize / sizeof(float);

    static constexpr size_t kVector = sizeof(float4) / sizeof(float);
    static constexpr size_t kWaveStride = Config::kWaveSize * kVector;

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

    // Elements of this tile that carry data: the instruction count clamped to
    // the page.
    __device__ static size_t valid_count(const Instruction& inst) {
        if (inst.count <= 0) return 0;
        return static_cast<size_t>(inst.count) <
                       static_cast<size_t>(kTileElements)
                   ? static_cast<size_t>(inst.count)
                   : static_cast<size_t>(kTileElements);
    }

    __device__ void load(State<Config>& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        const int pid = s.pid(0);
        s.wait_page_ready(pid);

        float* page = static_cast<float*>(s.pages[pid].ptr());
        const float* src =
            g.input.data + static_cast<size_t>(inst.tile) * kTileElements;
        const size_t count = valid_count(inst);

        // Coalesced HBM read: lane i takes the i-th 16 B chunk, so one wave
        // covers 1 KiB of consecutive addresses per iteration.
        size_t vector_end = 0;
        if (copy::aligned<float4>(src) && copy::aligned<float4>(page)) {
            vector_end = count / kVector * kVector;
            for (size_t element = static_cast<size_t>(lane) * kVector;
                 element < vector_end; element += kWaveStride) {
                copy::store_vec<float4>(
                    page + element,
                    copy::load_vec<float4>(src + element));
            }
        }
        // Unaligned base or partial tile: scalar tail, then zero the page.
        for (size_t element = vector_end + lane; element < count;
             element += Config::kWaveSize)
            page[element] = src[element];
        for (size_t element = count + lane; element < kTileElements;
             element += Config::kWaveSize)
            page[element] = 0.0f;

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[0].arrive();

        // Release pages this op does not use, mirroring NoOp's load().
        if (lane >= 1 && lane < Config::kNumPages) {
            const int unused = s.pid(lane);
            s.wait_page_ready(unused);
            s.finish_page(unused, Config::kConsumerWaves);
        }
    }

    __device__ void run(State<Config>& s, const Globals&) const override {
        const Instruction inst(this->instruction());
        const int wave = static_cast<int>(threadIdx.x / Config::kWaveSize);
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        const int pid = s.pid(0);

        s.semaphores()[0].wait(0);

        float* page = static_cast<float*>(s.pages[pid].ptr());
        static_assert(kTileElements % kVector == 0,
                      "elementwise vector path needs a tile of whole vectors");
        const size_t vectors = kTileElements / kVector;
        const size_t worker =
            static_cast<size_t>(wave) * Config::kWaveSize + lane;
        const size_t workers =
            static_cast<size_t>(Config::kConsumerWaves) * Config::kWaveSize;
        for (size_t vector = worker; vector < vectors; vector += workers) {
            float4 value = copy::load_vec<float4>(page + vector * kVector);
            value.x = inst.scale * value.x + inst.bias;
            value.y = inst.scale * value.y + inst.bias;
            value.z = inst.scale * value.z + inst.bias;
            value.w = inst.scale * value.w + inst.bias;
            copy::store_vec<float4>(page + vector * kVector, value);
        }
        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.semaphores()[1].arrive();
    }

    __device__ void store(State<Config>& s, const Globals& g) const override {
        const Instruction inst(this->instruction());
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        const int pid = s.pid(0);

        s.semaphores()[1].wait(0);

        const float* page = static_cast<const float*>(s.pages[pid].ptr());
        float* dst =
            g.output.data + static_cast<size_t>(inst.tile) * kTileElements;
        const size_t count = valid_count(inst);

        size_t vector_end = 0;
        if (copy::aligned<float4>(dst) && copy::aligned<float4>(page)) {
            vector_end = count / kVector * kVector;
            for (size_t element = static_cast<size_t>(lane) * kVector;
                 element < vector_end; element += kWaveStride) {
                copy::store_vec<float4>(
                    dst + element,
                    copy::load_vec<float4>(page + element));
            }
        }
        for (size_t element = vector_end + lane; element < count;
             element += Config::kWaveSize)
            dst[element] = page[element];

        lds::wave_barrier();
        __threadfence_block();
        if (lane == 0) s.finish_page(pid, Config::kConsumerWaves);
    }
};

}  // namespace dmk::mk
