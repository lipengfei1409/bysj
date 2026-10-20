// Standalone real-device contracts; no Torch dependency and no mock intrinsics.
#include "ops/ops.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void hip_check(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
template<typename T>
struct device_buffer {
    T* data = nullptr;
    explicit device_buffer(std::size_t count) {
        hip_check(hipMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~device_buffer() { if (data) hipFree(data); }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
    void upload(const std::vector<T>& values) {
        hip_check(hipMemcpy(data, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    std::vector<T> download(std::size_t count) const {
        std::vector<T> result(count);
        hip_check(hipMemcpy(result.data(), data, count * sizeof(T), hipMemcpyDeviceToHost));
        return result;
    }
};

static void close(float actual, float expected, const char* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-4f + 2e-4f * std::abs(expected))
        throw std::runtime_error(std::string(label) + " actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
}

__global__ void memory_wave_kernel(const float* input, float* output, float* wave_results) {
    __shared__ dcu::shared_tile<float, 16, 16> tile;
    // Logical tail 13x11 inside a physical 16x24 allocation, offset (1,2).
    dcu::group::load(tile, dcu::matrix_view<const float>{input, 13, 11, 24}, 1, 2);
    dcu::group::block_sync();
    for (unsigned i = dcu::group::thread_rank(); i < 256; i += dcu::group::block_threads())
        tile.data[i] = dcu::thread::fma(tile.data[i], 2.f, .5f);
    dcu::group::block_sync();
    dcu::group::store(dcu::matrix_view<float>{output, 13, 11, 24}, tile, 1, 2);
    unsigned tid = dcu::group::thread_rank();
    float value = float(dcu::group::lane_id() + 1);
    wave_results[3 * tid] = dcu::group::wave_sum(value);
    wave_results[3 * tid + 1] = dcu::group::wave_max(value);
    wave_results[3 * tid + 2] = dcu::group::shuffle(value, warpSize - 1);
}

static void memory_wave_test(int wave_size) {
    std::vector<float> input(16 * 24), output(16 * 24, -99.f);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 31) - 15) / 8.f;
    device_buffer<float> x(input.size()), y(output.size()), sums(3 * wave_size * 2);
    x.upload(input); y.upload(output);
    // Two complete waves; 2-D block tests flattened thread rank.
    hipLaunchKernelGGL(memory_wave_kernel, dim3(1), dim3(8, wave_size / 4), 0, 0,
                       x.data, y.data, sums.data);
    hip_check(hipGetLastError()); hip_check(hipDeviceSynchronize());
    auto actual = y.download(output.size());
    for (int r = 0; r < 16; ++r) for (int c = 0; c < 24; ++c)
        close(actual[r * 24 + c],
              r >= 1 && r < 13 && c >= 2 && c < 11 ? 2 * input[r * 24 + c] + .5f : -99.f,
              "cooperative masked load/store");
    auto results = sums.download(3 * wave_size * 2);
    for (int tid = 0; tid < wave_size * 2; ++tid) {
        close(results[3 * tid], float(wave_size * (wave_size + 1) / 2), "wave sum");
        close(results[3 * tid + 1], float(wave_size), "wave max");
        close(results[3 * tid + 2], float(wave_size), "shuffle");
    }
    std::cout << "cooperative LDS, FMA, two-wave sum/max/shuffle passed\n";
}

#if DPA4_ENABLE_DUMMA
template<typename Op, bool Direct = false>
__global__ void mma_kernel(const typename Op::input_type* a,
                           const typename Op::input_type* b,
                           const float* c, float* d, bool tail, bool zero_c) {
    using T = typename Op::input_type;
    __shared__ dcu::shared_tile<T, 16, Op::k> lhs;
    __shared__ dcu::shared_tile<T, Op::k, 16> rhs;
    // Loaders pad tails before the collective instruction. MMA itself is unmasked.
    dcu::group::load(lhs, dcu::matrix_view<const T>{a, tail ? 13 : 16,
                                                  tail ? Op::k - 1 : Op::k, 32});
    dcu::group::load(rhs, dcu::matrix_view<const T>{b, tail ? Op::k - 1 : Op::k,
                                                  tail ? 11 : 16, 24});
    dcu::group::block_sync();
    typename Op::a_fragment af;
    typename Op::b_fragment bf;
    typename Op::accumulator cf, df;
    if constexpr (Direct) {
        // Exercise the four independent instructions, not just the shape facade.
        Op::check_arch();
        dcu::dumma::load_matrix_sync(af, lhs.data, Op::k);
        dcu::dumma::load_matrix_sync(bf, rhs.data, 16);
        if (zero_c) dcu::dumma::fill_fragment(cf, 0.f);
        else dcu::dumma::load_matrix_sync(cf, c, 24, dcu::dumma::row_major_memory);
        dcu::dumma::mma_sync(df, af, bf, cf);
        dcu::dumma::store_matrix_sync(d, df, 24);
    } else {
        Op::load_a(af, lhs.data, Op::k);
        Op::load_b(bf, rhs.data, 16);
        if (zero_c) Op::fill(cf, 0.f);
        else Op::load_c(cf, c, 24);
        Op::mma(df, af, bf, cf);
        Op::store(d, df, 24);
    }
}

template<typename T> static T convert(float value) { return T(value); }
template<> __half convert<__half>(float value) { return __float2half(value); }
template<typename T> static float as_float(T value) { return float(value); }
template<> float as_float<__half>(__half value) { return __half2float(value); }

template<typename Op>
static void mma_test() {
    using T = typename Op::input_type;
    std::vector<T> a(16 * 32), b(Op::k * 24);
    std::vector<float> c(16 * 24), initial(16 * 24, -99.f);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = convert<T>(float(int(i % 23) - 11) / 7.3f);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = convert<T>(float(int(i % 17) - 8) / 9.7f);
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = float(int(i % 11) - 5) / 3.1f;
    device_buffer<T> da(a.size()), db(b.size());
    device_buffer<float> dc(c.size()), dd(initial.size());
    da.upload(a); db.upload(b); dc.upload(c);
    for (bool direct : {false, true})
    for (bool tail : {false, true}) for (bool zero_c : {false, true}) {
        dd.upload(initial);
        if (direct)
            hipLaunchKernelGGL((mma_kernel<Op, true>), dim3(1), dim3(Op::wave_threads), 0, 0,
                               da.data, db.data, dc.data, dd.data, tail, zero_c);
        else
            hipLaunchKernelGGL((mma_kernel<Op>), dim3(1), dim3(Op::wave_threads), 0, 0,
                               da.data, db.data, dc.data, dd.data, tail, zero_c);
        hip_check(hipGetLastError()); hip_check(hipDeviceSynchronize());
        auto actual = dd.download(initial.size());
        for (int r = 0; r < 16; ++r) for (int n = 0; n < 24; ++n) {
            float expected = n < 16 ? (zero_c ? 0.f : c[r * 24 + n]) : -99.f;
            if (n < (tail ? 11 : 16) && r < (tail ? 13 : 16))
                for (int k = 0; k < (tail ? Op::k - 1 : Op::k); ++k)
                    expected = std::fma(as_float(a[r * 32 + k]), as_float(b[k * 24 + n]), expected);
            close(actual[r * 24 + n], expected, "DUMMA D=A*B+C/padding");
        }
    }
    std::cout << "DUMMA 16x16x" << Op::k << ", input bytes=" << sizeof(T)
              << ": direct/shape facade, nonzero/zero C, strided memory, LDS tail padding passed\n";
    if (std::getenv("DPA4_INSTRUCTION_BENCHMARK")) {
        for (int i = 0; i < 10; ++i)
            hipLaunchKernelGGL((mma_kernel<Op>), dim3(1), dim3(Op::wave_threads), 0, 0,
                               da.data, db.data, dc.data, dd.data, false, false);
        hip_check(hipGetLastError()); hip_check(hipDeviceSynchronize());
        hipEvent_t start, stop;
        hip_check(hipEventCreate(&start)); hip_check(hipEventCreate(&stop));
        hip_check(hipEventRecord(start, 0));
        for (int i = 0; i < 100; ++i)
            hipLaunchKernelGGL((mma_kernel<Op>), dim3(1), dim3(Op::wave_threads), 0, 0,
                               da.data, db.data, dc.data, dd.data, false, false);
        hip_check(hipGetLastError()); hip_check(hipEventRecord(stop, 0));
        hip_check(hipEventSynchronize(stop));
        float elapsed = 0;
        hip_check(hipEventElapsedTime(&elapsed, start, stop));
        hip_check(hipEventDestroy(start)); hip_check(hipEventDestroy(stop));
        std::cout << "single-tile global/LDS/MMA/store kernel mean us=" << elapsed * 10
                  << " (includes launch overhead, NOT isolated instruction latency)\n";
    }
}
#endif

int main() {
    try {
        hipDeviceProp_t properties{};
        int device = 0;
        hip_check(hipGetDevice(&device)); hip_check(hipGetDeviceProperties(&properties, device));
        std::cout << "device=" << properties.name << " gfx=" << properties.gcnArchName
                  << " wave=" << properties.warpSize << '\n';
        if (properties.warpSize != 32 && properties.warpSize != 64)
            throw std::runtime_error("wave contract only tested for wave32/wave64");
        memory_wave_test(properties.warpSize);
#if DPA4_ENABLE_DUMMA
        if (properties.warpSize != 64) throw std::runtime_error("DUMMA requires a full wave64");
        mma_test<dcu::group::mma_f32_16x16x4>();
#if defined(DPA4_TEST_EXTENDED_DUMMA)
        mma_test<dcu::group::mma_f32_16x16x8>();
        mma_test<dcu::group::mma_f16_16x16x16>();
#endif
#else
        std::cout << "DUMMA disabled: no hardware MMA was tested\n";
#endif
        std::cout << "HIP instruction contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
