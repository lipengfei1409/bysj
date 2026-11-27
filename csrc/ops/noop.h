#pragma once

// NoOp: instruction-stream filler and page-release operation.
//
// It is the DCU port of Megakernels' NoOp. Its load() releases every page
// immediately, which advances each page's phase exactly once for this
// instruction; this is what lets the next instruction reuse a page. It must be
// present in every kernel's op pack because opcode 0 is used as padding.

#include "ops/op.h"
#include "ops/state.h"

namespace ops {

template <typename Globals>
class NoOp final : public BaseOp<Globals> {
public:
    static constexpr int opcode = config::kOpNoop;

    __device__ explicit NoOp(const int* instruction)
        : BaseOp<Globals>(instruction) {}

    __device__ int release_lid(int query) const override { return query; }

    __device__ int init_semaphores(State&,
                                   const Globals&) const override {
        return 0;
    }

    __device__ void load(State& s, const Globals&) const override {
        const int lane = static_cast<int>(threadIdx.x % config::kWaveSize);
        if (lane < config::kNumPages) {
            const int pid = s.pid(lane);
            s.wait_page_ready(pid);
            s.finish_page(pid, config::kConsumerWaves);
        }
    }

    __device__ void run(State&, const Globals&) const override {}

    __device__ void store(State&, const Globals&) const override {}
};

}  // namespace ops
