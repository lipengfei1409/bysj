#include "common/tile.h"
#include "ops/thread/memory.h"
#include "ops/group/mma.h"
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    using namespace dmk;
    using namespace dmk::group;
    static_assert(tile_layout<16, 4>::rows == 16);
    static_assert(tile_layout<16, 4>::cols == 4);
    static_assert(tile_layout<16, 4>::offset(3, 2) == 14);
    static_assert(gemm_tile_shape<16, 16, 4>::a_layout::elements == 64);
    static_assert(gemm_tile_shape<16, 16, 4>::b_layout::elements == 64);
    static_assert(gemm_tile_shape<16, 16, 4>::c_layout::elements == 256);
    using fp32_k4 = select_mma_shape_t<mma_operand_type::fp32,
        mma_operand_type::fp32, mma_accumulator_type::fp32, 4>;
    using fp32_k8 = select_mma_shape_t<mma_operand_type::fp32,
        mma_operand_type::fp32, mma_accumulator_type::fp32, 8>;
    using tf32_k8 = select_mma_shape_t<mma_operand_type::tf32,
        mma_operand_type::tf32, mma_accumulator_type::fp32, 8>;
    using fp16_k16 = select_mma_shape_t<mma_operand_type::fp16,
        mma_operand_type::fp16, mma_accumulator_type::fp32, 16>;
    using bf16_k16 = select_mma_shape_t<mma_operand_type::bf16,
        mma_operand_type::bf16, mma_accumulator_type::fp32, 16>;
    using int8_k32 = select_mma_shape_t<mma_operand_type::int8,
        mma_operand_type::int8, mma_accumulator_type::int32, 32>;
    using uint8_k32 = select_mma_shape_t<mma_operand_type::uint8,
        mma_operand_type::uint8, mma_accumulator_type::int32, 32>;
    using int4_k64 = select_mma_shape_t<mma_operand_type::int4_packed,
        mma_operand_type::int4_packed, mma_accumulator_type::int32, 64>;
    using uint4_k64 = select_mma_shape_t<mma_operand_type::uint4_packed,
        mma_operand_type::uint4_packed, mma_accumulator_type::int32, 64>;
    using fp64_k4 = select_mma_shape_t<mma_operand_type::fp64,
        mma_operand_type::fp64, mma_accumulator_type::fp64, 4>;
    using fp8_e4m3_k32 = select_mma_shape_t<mma_operand_type::fp8_e4m3,
        mma_operand_type::fp8_e4m3, mma_accumulator_type::fp32, 32>;
    using fp8_e5m2_k32 = select_mma_shape_t<mma_operand_type::fp8_e5m2,
        mma_operand_type::fp8_e5m2, mma_accumulator_type::fp32, 32>;
    static_assert(fp32_k4::m == 16 && fp32_k4::n == 16 && fp32_k4::k == 4);
    static_assert(fp32_k4::supports_arch<architecture::gfx926> &&
                  fp32_k4::supports_arch<architecture::gfx928> &&
                  fp32_k4::supports_arch<architecture::gfx936> &&
                  fp32_k4::supports_arch<architecture::gfx938>);
    static_assert(fp32_k8::k == 8 && !fp32_k8::supports_arch<architecture::gfx926> &&
                  fp32_k8::supports_arch<architecture::gfx928>);
    static_assert(tf32_k8::k == 8 && tf32_k8::supports_arch<architecture::gfx938>);
    static_assert(fp16_k16::k == 16 && fp16_k16::supports_arch<architecture::gfx928>);
    static_assert(bf16_k16::k == 16 && bf16_k16::supports_arch<architecture::gfx936>);
    static_assert(int8_k32::k == 32 && int8_k32::supports_arch<architecture::gfx928>);
    static_assert(uint8_k32::k == 32 && uint8_k32::supports_arch<architecture::gfx938>);
    static_assert(int4_k64::k == 64 && !int4_k64::supports_arch<architecture::gfx928> &&
                  int4_k64::supports_arch<architecture::gfx936>);
    static_assert(uint4_k64::k == 64 && uint4_k64::supports_arch<architecture::gfx938>);
    static_assert(fp64_k4::k == 4 && fp64_k4::supports_arch<architecture::gfx926> &&
                  !fp64_k4::supports_arch<architecture::gfx928>);
    static_assert(fp8_e4m3_k32::supports_arch<architecture::gfx938> &&
                  !fp8_e4m3_k32::supports_arch<architecture::gfx936>);
    static_assert(fp8_e5m2_k32::supports_arch<architecture::gfx938> &&
                  !fp8_e5m2_k32::supports_arch<architecture::gfx926>);
    static_assert(dumma_supported<architecture::gfx926, float, 4>);
    static_assert(dumma_supported<architecture::gfx928, float, 4>);
    static_assert(dumma_supported<architecture::gfx936, float, 4>);
    static_assert(dumma_supported<architecture::gfx938, float, 4>);
    static_assert(!dumma_supported<architecture::gfx926, float, 8>);
    static_assert(!dumma_supported<architecture::unknown, float, 4>);
    static_assert(!dumma_supported<architecture::gfx938, float, 16>);

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
    register_tile<float, 2, 4> tile;
    thread::load(tile, source, 2, 3);
    require(tile(0, 0) == source(2, 3), "tail value");
    require(tile(0, 2) == 0 && tile(1, 0) == 0, "tail zero fill");
    std::cout << "host DUMMA GEMM contracts passed\n";
}
