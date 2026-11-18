#pragma once

// NoOp: instruction-stream filler and page-release operation.
//
// It is the DCU port of Megakernels' NoOp. Its load() releases every page
// immediately, which advances each page's phase exactly once for this
// instruction; this is what lets the next instruction reuse a page. It must be
// present in every kernel's op pack because opcode 0 is used as padding.

#include "op.h"
#include "state.h"

namespace dmk::mk {

template <typename Config, typename Globals>
class NoOp final : public BaseOp<Config, Globals> {
public:
    static constexpr int opcode = 0;

    __device__ explicit NoOp(const int* instruction)
        : BaseOp<Config, Globals>(instruction) {}

    __device__ int release_lid(int query) const override { return query; }

    __device__ int init_semaphores(State<Config>&,
                                   const Globals&) const override {
        return 0;
    }

    __device__ void load(State<Config>& s, const Globals&) const override {
        const int lane = static_cast<int>(threadIdx.x % Config::kWaveSize);
        if (lane < Config::kNumPages) {
            const int pid = s.pid(lane);
            s.wait_page_ready(pid);
            s.finish_page(pid, Config::kConsumerWaves);
        }
    }

    __device__ void run(State<Config>&, const Globals&) const override {}

    __device__ void store(State<Config>&, const Globals&) const override {}
};

}  // namespace dmk::mk
