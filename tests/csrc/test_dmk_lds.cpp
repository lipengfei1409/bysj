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
using dmk::mk::Config;
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
    CHECK(Config::kStages == 2);
    CHECK(Config::kStageBits == 1);
    CHECK(Config::kInstructionWidth == 32);
    CHECK(Config::kConsumerWaves == 4);
    CHECK(Config::kServiceWaves == 3);
    CHECK(Config::kWorkerWaves == 6);
    CHECK(Config::kTotalWaves == 7);
    CHECK(Config::kThreads == 448);
    CHECK(Config::kPageSize == 8192);
    CHECK(Config::kNumPages == 6);

    CHECK(kDynamicBytes == 6 * 8192);
    CHECK(kLaunchBytes == kDynamicBytes + Config::kDynamicAlign);
    CHECK(kStaticBytes ==
          sizeof(InstructionState<Config>) * Config::kStages +
              sizeof(lds::Semaphore) *
                  (2 * Config::kStages + 1 +
                   Config::kNumPages * Config::kStageBits));
    CHECK(kStaticBytes + kLaunchBytes <= Config::kLdsBytesPerCu);

    CHECK(sizeof(lds::Semaphore) == 16);
    CHECK(alignof(lds::Semaphore) == 16);
    CHECK(sizeof(InstructionState<Config>) % 128 == 0);
    CHECK(sizeof(Page<Config>) == Config::kPageSize);
    CHECK(alignof(Page<Config>) == 128);
    CHECK(std::is_trivially_destructible<InstructionState<Config>>::value);
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
    constexpr int kPages = Config::kNumPages;
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

    CHECK(dmk::mk::ring_advance<Config::kStages>(0) == 1);
    CHECK(dmk::mk::ring_advance<Config::kStages>(1) == 0);
    CHECK(dmk::mk::ring_retreat<Config::kStages>(0) == 1);
    CHECK(dmk::mk::ring_retreat<Config::kStages>(1) == 0);
}

void test_vector_io() {
    constexpr size_t kCount = 1000;
    alignas(16) float src[kCount];
    alignas(16) float dst[kCount];
    for (size_t i = 0; i < kCount; ++i) {
        src[i] = static_cast<float>(i) * 0.5f;
        dst[i] = -1.0f;
    }
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(dmk::copy::global_to_lds_vec<float4>(dst, src, kCount, worker, 4));
    for (size_t i = 0; i < kCount; ++i) CHECK(dst[i] == src[i]);

    // Misaligned pointers force the scalar fallback but must stay correct.
    alignas(16) float misaligned_src[kCount + 1];
    alignas(16) float misaligned_dst[kCount + 1];
    for (size_t i = 0; i < kCount + 1; ++i) {
        misaligned_src[i] = static_cast<float>(i + 1);
        misaligned_dst[i] = -1.0f;
    }
    for (size_t worker = 0; worker < 4; ++worker)
        CHECK(!dmk::copy::global_to_lds_vec<float4>(
            misaligned_dst + 1, misaligned_src + 1, kCount, worker, 4));
    for (size_t i = 0; i < kCount; ++i)
        CHECK(misaligned_dst[i + 1] == misaligned_src[i + 1]);

    // Unaligned fill head, vector middle.
    alignas(16) float fill[100];
    for (float& value : fill) value = -1.0f;
    CHECK(dmk::copy::zero_lds_vec<float4>(fill + 1, 95, 0, 1));
    CHECK(fill[0] == -1.0f);
    for (size_t i = 1; i <= 95; ++i) CHECK(fill[i] == 0.0f);
    for (size_t i = 96; i < 100; ++i) CHECK(fill[i] == -1.0f);

    // Row copy with padded destination and row strides that satisfy float4.
    constexpr size_t kRows = 3;
    constexpr size_t kCols = 8;
    alignas(16) float rows_src[kRows * kCols];
    alignas(16) float rows_dst[kRows * 12];
    for (size_t i = 0; i < kRows * kCols; ++i)
        rows_src[i] = static_cast<float>(i);
    for (float& value : rows_dst) value = -1.0f;
    for (size_t worker = 0; worker < 2; ++worker)
        CHECK(dmk::copy::global_rows_to_lds_vec<float4>(
            rows_dst, 12, rows_src, kCols, kRows, kCols, worker, 2));
    for (size_t row = 0; row < kRows; ++row) {
        for (size_t col = 0; col < kCols; ++col)
            CHECK(rows_dst[row * 12 + col] == rows_src[row * kCols + col]);
        for (size_t col = kCols; col < 12; ++col)
            CHECK(rows_dst[row * 12 + col] == -1.0f);
    }

    // Columns/row strides that are not multiples of the vector width force the
    // scalar row path.
    constexpr size_t kOddCols = 7;
    alignas(16) float odd_src[kRows * kOddCols];
    alignas(16) float odd_dst[kRows * 9];
    for (size_t i = 0; i < kRows * kOddCols; ++i)
        odd_src[i] = static_cast<float>(i + 3);
    for (float& value : odd_dst) value = -1.0f;
    CHECK(!dmk::copy::global_rows_to_lds_vec<float4>(
        odd_dst, 9, odd_src, kOddCols, kRows, kOddCols, 0, 1));
    for (size_t row = 0; row < kRows; ++row)
        for (size_t col = 0; col < kOddCols; ++col)
            CHECK(odd_dst[row * 9 + col] == odd_src[row * kOddCols + col]);
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
    test_vector_io();
    test_globals_contract();
    if (failures != 0) {
        std::fprintf(stderr, "%d LDS host check(s) failed\n", failures);
        return 1;
    }
    std::printf("LDS host checks passed\n");
    return 0;
}
