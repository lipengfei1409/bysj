#include "ops/ops.cuh"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    using namespace dcu;
    static_assert(dumma_supported<architecture::gfx926, float, 4>);
    static_assert(!dumma_supported<architecture::gfx926, float, 8>);
    static_assert(dumma_supported<architecture::gfx938, float, 8>);
    static_assert(!dumma_supported<architecture::unknown, float, 4>);
    static_assert(!dumma_supported<architecture::gfx938, float, 16>);
    static_assert(dumma_supported<architecture::gfx928, float, 4>);
    static_assert(dumma_supported<architecture::gfx936, float, 8>);
    float input[3 * 8];
    for (int i = 0; i < 24; ++i) input[i] = float(i - 7);
    const matrix_view<const float> source{input, 3, 5, 8};
    require(source.valid(), "strided view rejected");
    require(!matrix_view<float>{input, 3, 5, 4}.valid(), "short stride accepted");
    require(!matrix_view<float>{nullptr, 3, 5, 8}.valid(), "null view accepted");
    require(matrix_view<float>{nullptr, 0, 5, 8}.valid(), "empty view rejected");
    require(thread::load_masked(source, -1, 0) == 0, "negative mask");
    require(thread::load_masked(matrix_view<const float>{nullptr, 0, 5, 8}, 0, 0) == 0,
            "empty masked load");
    thread::store_masked(matrix_view<float>{nullptr, 0, 5, 8}, 0, 0, 1.f);
    register_tile<float, 2, 4> tail;
    thread::load(tail, source, 2, 3);
    require(tail(0, 0) == source(2, 3), "tail value");
    require(tail(0, 2) == 0 && tail(1, 0) == 0, "tail zero fill");
    float output[3 * 8];
    for (float& x : output) x = -99;
    matrix_view<float> target{output, 3, 5, 8};
    thread::store(target, tail, 2, 3);
    require(output[2 * 8 + 3] == input[2 * 8 + 3], "masked store value");
    require(output[2 * 8 + 5] == -99 && output[0] == -99, "masked store bounds");

    register_tile<float, 2, 3> a;
    register_tile<float, 3, 4> b;
    register_tile<float, 2, 4> c, d;
    for (int i = 0; i < 6; ++i) a.data[i] = float(i - 3) / 2;
    for (int i = 0; i < 12; ++i) b.data[i] = float(5 - i) / 4;
    thread::fill(c, .75f);
    thread::mma_reference(d, a, b, c);
    for (int r = 0; r < 2; ++r) for (int n = 0; n < 4; ++n) {
        float expected = .75f;
        for (int k = 0; k < 3; ++k) expected = std::fma(a(r, k), b(k, n), expected);
        require(d(r, n) == expected, "D=A*B+C");
    }
    thread::mma_reference(c, a, b, c);
    for (int i = 0; i < 8; ++i) require(c.data[i] == d.data[i], "D/C alias");
    require(thread::fma(1.25f, -2.f, .5f) == -2.f, "scalar fma");
    require(thread::fma(1.25, -2., .5) == -2., "double fma");
    std::cout << "host instruction contracts passed\n";
}
