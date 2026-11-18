// Host-side logic test for the DCU megakernel LDS layout and page protocol.
// It needs no DCU: it checks the static/dynamic LDS budget, page/semaphore
// layout, the mbarrier-compatible phase arithmetic, the page permutation and
// the globals contract. The device test runs the real instruction stream.
//
// Build: hipcc -O2 -std=c++17 -I csrc/dmk tests/csrc/test_dmk_lds.cpp -o /tmp/test_dmk_lds
#include "config.h"
#include "copy.h"
#include "lds.h"
#include "megakernel.h"
#include "state.h"

#include <cstddef>
#include <cstdio>
#include <type_traits>

namespace {

using namespace dmk;
namespace config = dmk::config;
using dmk::mk::InstructionState;
using dmk::mk::kDynamicBytes;
using dmk::mk::kLaunchBytes;
using dmk::mk::kStaticBytes;
using dmk::mk::MkGlobals;
using dmk::mk::Page;
using dmk::mk::remap_page_order;
using dmk::tensor::HbmLayout;
using dmk::tensor::MemorySpace;
using dmk::tensor::Shape;
using dmk::tensor::Tensor;

int failures = 0;

void check(bool condition, const char* expression, int line) {
    if (condition) return;
    std::fprintf(stderr, "FAIL line %d: %s\n", line, expression);
    ++failures;
}

#define CHECK(expression) check((expression), #expression, __LINE__)

void test_config_budget() {
    CHECK(config::kStages == 2);
    CHECK(config::kStageBits == 1);
    CHECK(config::kInstructionWidth == 32);
    CHECK(config::kConsumerWaves == 4);
    CHECK(config::kServiceWaves == 3);
    CHECK(config::kWorkerWaves == 6);
    CHECK(config::kTotalWaves == 7);
    CHECK(config::kThreads == 448);
    CHECK(config::kPageSize == 8192);
    CHECK(config::kNumPages == 6);

    CHECK(kDynamicBytes == 6 * 8192);
    CHECK(kLaunchBytes == kDynamicBytes + config::kDynamicAlign);
    CHECK(kStaticBytes ==
          sizeof(InstructionState) * config::kStages +
              sizeof(lds::Semaphore) *
                  (2 * config::kStages + 1 +
                   config::kNumPages * config::kStageBits));
    CHECK(kStaticBytes + kLaunchBytes <= config::kLdsBytesPerCu);

    CHECK(sizeof(lds::Semaphore) == 16);
    CHECK(alignof(lds::Semaphore) == 16);
    CHECK(sizeof(InstructionState) % 128 == 0);
    CHECK(sizeof(Page) == config::kPageSize);
    CHECK(alignof(Page) == 128);
    CHECK(std::is_trivially_destructible<InstructionState>::value);
}

void test_phase_arithmetic() {
    // mbarrier parity: after `init(N); arrive(N);` phase 0 is complete and the
    // current phase parity becomes 1.
    CHECK(dmk::lds::phase_for_ticket(0, 1) == 0);
    CHECK(dmk::lds::phase_for_ticket(1, 1) == 1);
    CHECK(dmk::lds::phase_for_ticket(2, 1) == 0);
    CHECK(dmk::lds::phase_for_ticket(3, 1) == 1);
    CHECK(dmk::lds::phase_for_ticket(0, 4) == 0);
    CHECK(dmk::lds::phase_for_ticket(3, 4) == 0);
    CHECK(dmk::lds::phase_for_ticket(4, 4) == 1);
    CHECK(dmk::lds::phase_for_ticket(7, 4) == 1);
    CHECK(dmk::lds::phase_for_ticket(8, 4) == 0);
    CHECK(dmk::lds::phase_for_ticket(12, 4) == 1);
    CHECK(dmk::lds::phase_for_ticket(42, 0) == 0);
}

void test_page_permutation() {
    constexpr int kPages = config::kNumPages;
    const int previous[kPages] = {2, 4, 0, 5, 1, 3};
    const int identity[kPages] = {0, 1, 2, 3, 4, 5};
    for (int query = 0; query < kPages; ++query) {
        CHECK(remap_page_order(previous, identity[query]) == previous[query]);
    }

    // A rotation of logical page ids rotates the physical assignment.
    const int rotate[kPages] = {1, 2, 3, 4, 5, 0};
    const int expected[kPages] = {4, 0, 5, 1, 3, 2};
    for (int query = 0; query < kPages; ++query) {
        CHECK(remap_page_order(previous, rotate[query]) == expected[query]);
    }

    // Identity page order is what the demo op uses.
    for (int query = 0; query < kPages; ++query) {
        CHECK(remap_page_order(identity, query) == query);
    }

    CHECK(dmk::mk::ring_advance<config::kStages>(0) == 1);
    CHECK(dmk::mk::ring_advance<config::kStages>(1) == 0);
    CHECK(dmk::mk::ring_retreat<config::kStages>(0) == 1);
    CHECK(dmk::mk::ring_retreat<config::kStages>(1) == 0);
}

void test_vector_access() {
    constexpr size_t kCount = 1000;
    constexpr size_t kVector = sizeof(float4) / sizeof(float);
    constexpr size_t kWaveStride = 64 * kVector;  // config::kWaveSize * kVector

    alignas(16) float src[kCount];
    alignas(16) float dst[kCount];
    for (size_t i = 0; i < kCount; ++i) {
        src[i] = static_cast<float>(i) * 0.5f;
        dst[i] = -1.0f;
    }

    CHECK(dmk::copy::aligned<float4>(src));
    CHECK(dmk::copy::aligned<float2>(src + 2));
    CHECK(!dmk::copy::aligned<float4>(src + 1));
    CHECK(!dmk::copy::aligned<float4>(src + 3));

    // Exactly the loop an op's load()/store() runs: lane i owns the 16 B chunk
    // starting at element i * kVector, and the tail below kVector * vectors is
    // left for the scalar path.
    const size_t vector_end = kCount / kVector * kVector;
    for (size_t lane = 0; lane < 64; ++lane) {
        for (size_t element = lane * kVector; element < vector_end;
             element += kWaveStride) {
            dmk::copy::store_vec<float4>(
                dst + element, dmk::copy::load_vec<float4>(src + element));
        }
    }
    for (size_t i = 0; i < vector_end; ++i) CHECK(dst[i] == src[i]);
    for (size_t i = vector_end; i < kCount; ++i) CHECK(dst[i] == -1.0f);

    // The narrow vector sees the same data with a different width.
    alignas(16) float narrow[kCount];
    for (size_t i = 0; i < kCount; ++i) narrow[i] = 0.0f;
    const size_t narrow_end = kCount / 2 * 2;
    for (size_t element = 0; element < narrow_end; element += 2) {
        dmk::copy::store_vec<float2>(narrow + element,
                                     dmk::copy::load_vec<float2>(src + element));
    }
    for (size_t i = 0; i < narrow_end; ++i) CHECK(narrow[i] == src[i]);

    // Zero fill with the same primitive.
    alignas(16) float page[256];
    for (float& value : page) value = -1.0f;
    const float4 zero{0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t element = 3; element + kVector <= 256; element += kVector)
        dmk::copy::store_vec<float4>(page + element, zero);
    for (size_t i = 0; i < 3; ++i) CHECK(page[i] == -1.0f);
    for (size_t i = 3; i < 255; ++i) CHECK(page[i] == 0.0f);
}

void test_globals_contract() {
    const Shape<3> instruction_shape{{2, 4, 32}};
    const Shape<1> data_shape{{2048}};
    const HbmLayout<1> data_layout = HbmLayout<1>::contiguous(data_shape);
    float input[2048]{};
    float output[2048]{};
    int instructions[2 * 4 * 32]{};

    const MkGlobals good{
        Tensor<const int, 3, MemorySpace::HBM>{
            instructions, instruction_shape,
            HbmLayout<3>::contiguous(instruction_shape)},
        Tensor<const float, 1, MemorySpace::HBM>{input, data_shape,
                                                 data_layout},
        Tensor<float, 1, MemorySpace::HBM>{output, data_shape, data_layout},
        4,
    };
    CHECK(good.valid(2));
    CHECK(!good.valid(3));  // fewer than `blocks` instruction rows

    MkGlobals wrong_width = good;
    wrong_width.instructions.shape.dims[2] = 16;
    CHECK(!wrong_width.valid(2));

    MkGlobals too_few_instructions = good;
    too_few_instructions.num_instructions = 5;
    CHECK(!too_few_instructions.valid(2));

    MkGlobals strided = good;
    strided.output.layout = HbmLayout<1>{{2}};
    CHECK(!strided.valid(2));

    MkGlobals empty = good;
    empty.num_instructions = 0;
    CHECK(!empty.valid(2));

    MkGlobals no_instructions = good;
    no_instructions.instructions.data = nullptr;
    CHECK(!no_instructions.valid(2));
}

}  // namespace

int main() {
    test_config_budget();
    test_phase_arithmetic();
    test_page_permutation();
    test_vector_access();
    test_globals_contract();
    if (failures != 0) {
        std::fprintf(stderr, "%d LDS host check(s) failed\n", failures);
        return 1;
    }
    std::printf("LDS host checks passed\n");
    return 0;
}
