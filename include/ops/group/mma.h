#pragma once
#include "../../common/config.h"
#include "dumma.h"
#include <type_traits>

namespace dmk::group {
// Type tags and compile-time shape metadata from DTK 26.04 DUMMA manual §3.
// These descriptors select a documented instruction shape; they do not imply
// that every dtype has a project-level GEMM kernel implementation.
enum class mma_operand_type {
    fp32,
    tf32,
    fp16,
    bf16,
    int8,
    uint8,
    int4_packed,
    uint4_packed,
    fp64,
    fp8_e4m3,
    fp8_e5m2,
};

enum class mma_accumulator_type { fp32, int32, fp64 };

template<mma_operand_type A, mma_operand_type B,
         mma_accumulator_type Accumulator, int K,
         architecture... SupportedArchitectures>
struct mma_shape_config {
    static constexpr int m = 16;
    static constexpr int n = 16;
    static constexpr int k = K;
    static constexpr int wave_threads = 64;
    static constexpr mma_operand_type a_type = A;
    static constexpr mma_operand_type b_type = B;
    static constexpr mma_accumulator_type accumulator_type = Accumulator;

    template<architecture Arch>
    static constexpr bool supports_arch =
        ((Arch == SupportedArchitectures) || ...);

    static constexpr bool supports(architecture arch) {
        return ((arch == SupportedArchitectures) || ...);
    }
};

using mma_f32_16x16x4_shape = mma_shape_config<
    mma_operand_type::fp32, mma_operand_type::fp32,
    mma_accumulator_type::fp32, 4,
    architecture::gfx926, architecture::gfx928,
    architecture::gfx936, architecture::gfx938>;
using mma_f32_16x16x8_shape = mma_shape_config<
    mma_operand_type::fp32, mma_operand_type::fp32,
    mma_accumulator_type::fp32, 8,
    architecture::gfx928, architecture::gfx936, architecture::gfx938>;
using mma_tf32_16x16x8_shape = mma_shape_config<
    mma_operand_type::tf32, mma_operand_type::tf32,
    mma_accumulator_type::fp32, 8,
    architecture::gfx928, architecture::gfx936, architecture::gfx938>;
using mma_f16_16x16x16_shape = mma_shape_config<
    mma_operand_type::fp16, mma_operand_type::fp16,
    mma_accumulator_type::fp32, 16,
    architecture::gfx928, architecture::gfx936, architecture::gfx938>;
using mma_bf16_16x16x16_shape = mma_shape_config<
    mma_operand_type::bf16, mma_operand_type::bf16,
    mma_accumulator_type::fp32, 16,
    architecture::gfx928, architecture::gfx936, architecture::gfx938>;
using mma_i8_16x16x32_shape = mma_shape_config<
    mma_operand_type::int8, mma_operand_type::int8,
    mma_accumulator_type::int32, 32,
    architecture::gfx928, architecture::gfx936, architecture::gfx938>;
using mma_u8_16x16x32_shape = mma_shape_config<
    mma_operand_type::uint8, mma_operand_type::uint8,
    mma_accumulator_type::int32, 32,
    architecture::gfx928, architecture::gfx936, architecture::gfx938>;
using mma_s4_16x16x64_shape = mma_shape_config<
    mma_operand_type::int4_packed, mma_operand_type::int4_packed,
    mma_accumulator_type::int32, 64,
    architecture::gfx936, architecture::gfx938>;
using mma_u4_16x16x64_shape = mma_shape_config<
    mma_operand_type::uint4_packed, mma_operand_type::uint4_packed,
    mma_accumulator_type::int32, 64,
    architecture::gfx936, architecture::gfx938>;
using mma_f64_16x16x4_shape = mma_shape_config<
    mma_operand_type::fp64, mma_operand_type::fp64,
    mma_accumulator_type::fp64, 4,
    architecture::gfx926, architecture::gfx936, architecture::gfx938>;
using mma_fp8_e4m3_16x16x32_shape = mma_shape_config<
    mma_operand_type::fp8_e4m3, mma_operand_type::fp8_e4m3,
    mma_accumulator_type::fp32, 32, architecture::gfx938>;
using mma_fp8_e5m2_16x16x32_shape = mma_shape_config<
    mma_operand_type::fp8_e5m2, mma_operand_type::fp8_e5m2,
    mma_accumulator_type::fp32, 32, architecture::gfx938>;

template<mma_operand_type A, mma_operand_type B,
         mma_accumulator_type Accumulator, int K>
struct select_mma_shape;

#define DMK_MMA_SHAPE_SELECT(A, B, ACC, K, SHAPE) \
    template<> struct select_mma_shape<A, B, ACC, K> { using type = SHAPE; }

DMK_MMA_SHAPE_SELECT(mma_operand_type::fp32, mma_operand_type::fp32,
                     mma_accumulator_type::fp32, 4, mma_f32_16x16x4_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::fp32, mma_operand_type::fp32,
                     mma_accumulator_type::fp32, 8, mma_f32_16x16x8_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::tf32, mma_operand_type::tf32,
                     mma_accumulator_type::fp32, 8, mma_tf32_16x16x8_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::fp16, mma_operand_type::fp16,
                     mma_accumulator_type::fp32, 16, mma_f16_16x16x16_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::bf16, mma_operand_type::bf16,
                     mma_accumulator_type::fp32, 16, mma_bf16_16x16x16_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::int8, mma_operand_type::int8,
                     mma_accumulator_type::int32, 32, mma_i8_16x16x32_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::uint8, mma_operand_type::uint8,
                     mma_accumulator_type::int32, 32, mma_u8_16x16x32_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::int4_packed, mma_operand_type::int4_packed,
                     mma_accumulator_type::int32, 64, mma_s4_16x16x64_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::uint4_packed, mma_operand_type::uint4_packed,
                     mma_accumulator_type::int32, 64, mma_u4_16x16x64_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::fp64, mma_operand_type::fp64,
                     mma_accumulator_type::fp64, 4, mma_f64_16x16x4_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::fp8_e4m3, mma_operand_type::fp8_e4m3,
                     mma_accumulator_type::fp32, 32, mma_fp8_e4m3_16x16x32_shape);
DMK_MMA_SHAPE_SELECT(mma_operand_type::fp8_e5m2, mma_operand_type::fp8_e5m2,
                     mma_accumulator_type::fp32, 32, mma_fp8_e5m2_16x16x32_shape);

#undef DMK_MMA_SHAPE_SELECT

template<mma_operand_type A, mma_operand_type B,
         mma_accumulator_type Accumulator, int K>
using select_mma_shape_t =
    typename select_mma_shape<A, B, Accumulator, K>::type;

#if defined(DCU_DUMMA_AVAILABLE)
// DUMMA's fragment lane distribution is opaque and remains owned by DTK.
// This operation wrapper instantiates the first project GEMM shape only.
struct mma_f32_16x16x4 : mma_f32_16x16x4_shape {
    using a_fragment = dumma::DUFragment<dumma::matrix_a, m, n, k, float,
                                         dumma::row_major>;
    using b_fragment = dumma::DUFragment<dumma::matrix_b, m, n, k, float,
                                         dumma::row_major>;
    using accumulator = dumma::DUFragment<dumma::accumulator, m, n, k, float>;

    DCU_DEVICE static void check_arch() {
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(mma_f32_16x16x4_shape::template supports_arch<compiled_arch>,
                      "DUMMA FP32 K=4 is unsupported on this gfx target");
#endif
    }
    DCU_DEVICE static void fill(accumulator& dst, float value) {
        check_arch();
        dumma::du_fill_fragment(dst, value);
    }
    DCU_DEVICE static void load_a(a_fragment& dst, const float* src, unsigned ld) {
        check_arch();
        dumma::du_load_matrix_sync(dst, src, ld);
    }
    DCU_DEVICE static void load_b(b_fragment& dst, const float* src, unsigned ld) {
        check_arch();
        dumma::du_load_matrix_sync(dst, src, ld);
    }
    DCU_DEVICE static void mma(accumulator& d, const a_fragment& a,
                               const b_fragment& b, const accumulator& c) {
        check_arch();
        dumma::du_mma_sync(d, a, b, c);
    }
    DCU_DEVICE static void store(float* dst, const accumulator& src, unsigned ld) {
        check_arch();
        dumma::du_store_matrix_sync(dst, src, ld, dumma::row_major_memory);
    }
};
#endif // DCU_DUMMA_AVAILABLE
} // namespace dmk::group
