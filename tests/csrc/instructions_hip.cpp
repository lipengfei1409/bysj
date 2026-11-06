#include <hip/hip_runtime.h>
#include "ops/group/dumma.h"
#include "ops/group/memory.h"
#include "ops/group/mma.h"
#include "ops/group/sync.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static void hip_check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

static void close(float actual, float expected) {
    const float tolerance = 3e-3f * fmaxf(1.f, fabsf(expected));
    if (fabsf(actual - expected) > tolerance)
        throw std::runtime_error("DUMMA tile result mismatch");
}

struct device_buffer {
    float* data = nullptr;
    explicit device_buffer(std::size_t count) { hip_check(hipMalloc(reinterpret_cast<void**>(&data), count * sizeof(float))); }
    ~device_buffer() { if (data) hipFree(data); }
    void upload(const std::vector<float>& values) {
        hip_check(hipMemcpy(data, values.data(), values.size() * sizeof(float), hipMemcpyHostToDevice));
    }
    std::vector<float> download(std::size_t count) const {
        std::vector<float> values(count);
        hip_check(hipMemcpy(values.data(), data, count * sizeof(float), hipMemcpyDeviceToHost));
        return values;
    }
};

#if DPA4_ENABLE_DUMMA
template<typename Op>
__global__ void mma_kernel(const float* a, const float* b, const float* c, float* d,
                           bool tail, bool zero_c) {
    __shared__ dmk::shared_tile<float, 16, Op::k> lhs;
    __shared__ dmk::shared_tile<float, Op::k, 16> rhs;
    dmk::group::load(lhs, dmk::matrix_view<const float>{a, tail ? 13 : 16,
                                                       tail ? Op::k - 1 : Op::k, 32});
    dmk::group::load(rhs, dmk::matrix_view<const float>{b, tail ? Op::k - 1 : Op::k,
                                                       tail ? 11 : 16, 24});
    dmk::group::block_sync();
    typename Op::a_fragment af;
    typename Op::b_fragment bf;
    typename Op::accumulator cf, df;
    Op::load_a(af, lhs.data, Op::k);
    Op::load_b(bf, rhs.data, 16);
    if (zero_c) Op::fill(cf, 0.f);
    else dmk::dumma::du_load_matrix_sync(cf, c, 24, dmk::dumma::row_major_memory);
    Op::mma(df, af, bf, cf);
    Op::store(d, df, 24);
}

static void mma_test() {
    using Op = dmk::group::mma_f32_16x16x4;
    std::vector<float> a(16 * 32), b(Op::k * 24), c(16 * 24), initial(16 * 24, -99.f);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(int(i % 23) - 11) / 7.3f;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = float(int(i % 17) - 8) / 9.7f;
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = float(int(i % 11) - 5) / 3.1f;
    device_buffer da(a.size()), db(b.size()), dc(c.size()), dd(initial.size());
    da.upload(a); db.upload(b); dc.upload(c);
    for (bool tail : {false, true}) for (bool zero_c : {false, true}) {
        dd.upload(initial);
        hipLaunchKernelGGL((mma_kernel<Op>), dim3(1), dim3(Op::wave_threads), 0, 0,
                           da.data, db.data, dc.data, dd.data, tail, zero_c);
        hip_check(hipGetLastError());
        hip_check(hipDeviceSynchronize());
        const auto actual = dd.download(initial.size());
        for (int r = 0; r < 16; ++r) for (int n = 0; n < 24; ++n) {
            float expected = n < 16 ? (zero_c ? 0.f : c[r * 24 + n]) : -99.f;
            if (n < (tail ? 11 : 16) && r < (tail ? 13 : 16))
                for (int k = 0; k < (tail ? Op::k - 1 : Op::k); ++k)
                    expected = fmaf(a[r * 32 + k], b[k * 24 + n], expected);
            close(actual[r * 24 + n], expected);
        }
    }
    std::cout << "DUMMA FP32 16x16x4, tail padding and zero accumulator passed\n";
}
#endif

int main() {
    try {
#if DPA4_ENABLE_DUMMA
        hipDeviceProp_t properties{};
        int device = 0;
        hip_check(hipGetDevice(&device));
        hip_check(hipGetDeviceProperties(&properties, device));
        if (properties.warpSize != 64)
            throw std::runtime_error("DUMMA requires a full wave64");
        mma_test();
        std::cout << "HIP DUMMA GEMM contracts passed\n";
        return 0;
#else
        throw std::runtime_error("compile this test with DPA4_ENABLE_DUMMA=1");
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
