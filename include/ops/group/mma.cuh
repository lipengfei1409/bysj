#pragma once
#include "../../common/config.cuh"
#include "dumma.cuh"
#include <type_traits>

#if defined(DCU_DUMMA_AVAILABLE)

namespace dcu {
template<architecture Arch>
inline constexpr bool dumma_supported<Arch, __half, 16> =
    Arch == architecture::gfx928 || Arch == architecture::gfx936 ||
    Arch == architecture::gfx938;
}

namespace dcu::group {
// Opaque register distribution is owned by DTK du::dumma. One full 64-lane
// wave executes all calls convergently, with uniform pointers/strides.
// A[16,K], B[K,16], C/D[16,16], row-major. No masking or conversion inside MMA.
template<typename InputT, int K>
struct mma_16x16 {
    static_assert((std::is_same_v<InputT, float> && (K == 4 || K == 8)) ||
                  (std::is_same_v<InputT, __half> && K == 16),
                  "supported MMA: FP32 K=4/8 or FP16 K=16, FP32 accumulator");
    static constexpr int m = 16, n = 16, k = K, wave_threads = 64;
    using input_type = InputT;
    using accumulator_type = float;
    using a_fragment = dumma::DUFragment<dumma::matrix_a, m, n, K, InputT, dumma::row_major>;
    using b_fragment = dumma::DUFragment<dumma::matrix_b, m, n, K, InputT, dumma::row_major>;
    using accumulator = dumma::DUFragment<dumma::accumulator, m, n, K, float>;

    DCU_DEVICE static void check_arch() {
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(dumma_supported<compiled_arch, InputT, K>,
                      "DUMMA shape/dtype unsupported on this gfx target; see DTK table 3.1");
#endif
    }
    DCU_DEVICE static void fill(accumulator& dst, float value) {
        check_arch();
        dumma::fill_fragment(dst, value);
    }
    DCU_DEVICE static void load_a(a_fragment& dst, const InputT* src, unsigned ld) {
        check_arch();
        dumma::load_matrix_sync(dst, src, ld);
    }
    DCU_DEVICE static void load_b(b_fragment& dst, const InputT* src, unsigned ld) {
        check_arch();
        dumma::load_matrix_sync(dst, src, ld);
    }
    DCU_DEVICE static void load_c(accumulator& dst, const float* src, unsigned ld) {
        check_arch();
        dumma::load_matrix_sync(dst, src, ld, dumma::row_major_memory);
    }
    DCU_DEVICE static void mma(accumulator& d, const a_fragment& a,
                               const b_fragment& b, const accumulator& c) {
        check_arch();
        dumma::mma_sync(d, a, b, c);
    }
    DCU_DEVICE static void store(float* dst, const accumulator& src, unsigned ld) {
        check_arch();
        dumma::store_matrix_sync(dst, src, ld, dumma::row_major_memory);
    }
};
using mma_f32_16x16x4 = mma_16x16<float, 4>;
using mma_f32_16x16x8 = mma_16x16<float, 8>;
using mma_f16_16x16x16 = mma_16x16<__half, 16>;
} // namespace dcu::group
#endif // DCU_DUMMA_AVAILABLE
