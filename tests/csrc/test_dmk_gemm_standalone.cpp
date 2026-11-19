// Standalone GEMM operator test: one kernel launch per case, a CPU reference in
// double precision, and no megakernel anywhere in the path.
//
// The operator is dmk::gemm::run_gemm_wave, templated on the input dtype and the
// DUMMA K (float: k4/k8, __half/__hip_bfloat16: k16). Which pairs exist depends
// on the target, so the runner defines DMK_MMA_F32K8 / DMK_MMA_F16 for
// gfx928/936/938; without them only the universal float k4 path is built.
//
// Build/run through: python tests/csrc/run_dmk_ops_test.py --op gemm
#include "gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#ifndef DMK_MMA_F32K8
#define DMK_MMA_F32K8 0
#endif
#ifndef DMK_MMA_F16
#define DMK_MMA_F16 0
#endif

namespace {

using dmk::tensor::HbmLayout;
using dmk::tensor::MemorySpace;
using dmk::tensor::Shape;
using dmk::tensor::Tensor;

void check_hip(hipError_t status, const char* what) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(status));
    std::exit(1);
}

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
}

struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        check_hip(hipMalloc(&ptr, bytes != 0 ? bytes : 1), "hipMalloc");
    }
    ~DeviceBuffer() {
        if (ptr) hipFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    T* data() const {
        return static_cast<T*>(ptr);
    }
};

float sample(size_t index) {
    const float x = static_cast<float>((index * 2654435761u) % 1000) / 500.0f;
    return x - 1.0f;
}

struct Event {
    hipEvent_t event{};
    Event() { check_hip(hipEventCreate(&event), "hipEventCreate"); }
    ~Event() {
        if (event) hipEventDestroy(event);
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

template <typename T>
const char* dtype_name() {
    if (std::is_same<T, float>::value) return "f32";
    if (std::is_same<T, __half>::value) return "f16";
    return "bf16";
}

// Fills [rows, cols], stored row-major or column-major.
template <typename T>
std::vector<T> make_matrix(size_t rows, size_t cols, bool column_major,
                           size_t salt) {
    std::vector<T> data(rows * cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            const size_t index =
                column_major ? col * rows + row : row * cols + col;
            data[index] = dmk::dtype::from_float<T>(
                sample(row * 131 + col * 17 + salt));
        }
    }
    return data;
}

HbmLayout<2> matrix_layout(size_t rows, size_t cols, bool column_major) {
    return column_major ? HbmLayout<2>{{1, rows}} : HbmLayout<2>{{cols, 1}};
}

__global__ void gemm_kernel_f32k4(dmk::gemm::GemmParamsT<float> params) {
    __shared__ dmk::gemm::GemmSharedT<float, 4> shared;
    dmk::gemm::run_gemm_wave(params, shared);
}

#if DMK_MMA_F32K8
__global__ void gemm_kernel_f32k8(dmk::gemm::GemmParamsT<float> params) {
    __shared__ dmk::gemm::GemmSharedT<float, 8> shared;
    dmk::gemm::run_gemm_wave(params, shared);
}
#endif

#if DMK_MMA_F16
__global__ void gemm_kernel_f16(dmk::gemm::GemmParamsT<__half> params) {
    __shared__ dmk::gemm::GemmSharedT<__half, 16> shared;
    dmk::gemm::run_gemm_wave(params, shared);
}

__global__ void gemm_kernel_bf16(
    dmk::gemm::GemmParamsT<__hip_bfloat16> params) {
    __shared__ dmk::gemm::GemmSharedT<__hip_bfloat16, 16> shared;
    dmk::gemm::run_gemm_wave(params, shared);
}
#endif

// One specialization per (dtype, K) pair the target supports.
template <typename T, int K>
struct Launch;

template <>
struct Launch<float, 4> {
    static void run(const dmk::gemm::GemmParamsT<float>& params, dim3 grid) {
        hipLaunchKernelGGL(gemm_kernel_f32k4, grid, dim3(64), 0, 0, params);
    }
};

#if DMK_MMA_F32K8
template <>
struct Launch<float, 8> {
    static void run(const dmk::gemm::GemmParamsT<float>& params, dim3 grid) {
        hipLaunchKernelGGL(gemm_kernel_f32k8, grid, dim3(64), 0, 0, params);
    }
};
#endif

#if DMK_MMA_F16
template <>
struct Launch<__half, 16> {
    static void run(const dmk::gemm::GemmParamsT<__half>& params, dim3 grid) {
        hipLaunchKernelGGL(gemm_kernel_f16, grid, dim3(64), 0, 0, params);
    }
};

template <>
struct Launch<__hip_bfloat16, 16> {
    static void run(const dmk::gemm::GemmParamsT<__hip_bfloat16>& params,
                    dim3 grid) {
        hipLaunchKernelGGL(gemm_kernel_bf16, grid, dim3(64), 0, 0, params);
    }
};
#endif

template <typename T, int K>
void run_case(const char* name, size_t m, size_t n, size_t k, bool column_b,
              bool column_c, double tolerance) {
    std::vector<T> host_a = make_matrix<T>(m, k, false, 1);
    std::vector<T> host_b = make_matrix<T>(k, n, column_b, 2);
    std::vector<float> host_c(m * n, -12345.0f);

    // Reference: C = A * B in double, on the values the kernel sees.
    std::vector<double> expected(m * n, 0.0);
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            double sum = 0.0;
            for (size_t inner = 0; inner < k; ++inner) {
                const size_t bi =
                    column_b ? col * k + inner : inner * n + col;
                sum += static_cast<double>(dmk::dtype::to_float(
                           host_a[row * k + inner])) *
                       static_cast<double>(dmk::dtype::to_float(host_b[bi]));
            }
            expected[row * n + col] = sum;
        }
    }

    DeviceBuffer device_a(host_a.size() * sizeof(T));
    DeviceBuffer device_b(host_b.size() * sizeof(T));
    DeviceBuffer device_c(host_c.size() * sizeof(float));
    check_hip(hipMemcpy(device_a.data<void>(), host_a.data(),
                        host_a.size() * sizeof(T), hipMemcpyHostToDevice),
              "copy a");
    check_hip(hipMemcpy(device_b.data<void>(), host_b.data(),
                        host_b.size() * sizeof(T), hipMemcpyHostToDevice),
              "copy b");
    check_hip(hipMemcpy(device_c.data<void>(), host_c.data(),
                        host_c.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy c");

    const dmk::gemm::GemmParamsT<T> params{
        Tensor<const T, 2, MemorySpace::HBM>{device_a.data<const T>(),
                                             Shape<2>{{m, k}},
                                             HbmLayout<2>{{k, 1}}},
        Tensor<const T, 2, MemorySpace::HBM>{device_b.data<const T>(),
                                             Shape<2>{{k, n}},
                                             matrix_layout(k, n, column_b)},
        Tensor<float, 2, MemorySpace::HBM>{device_c.data<float>(),
                                           Shape<2>{{m, n}},
                                           matrix_layout(m, n, column_c)},
    };
    if (!params.valid()) fail("params rejected");

    const dim3 grid(static_cast<unsigned>((n + 15) / 16),
                    static_cast<unsigned>((m + 15) / 16));
    Launch<T, K>::run(params, grid);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");
    check_hip(hipMemcpy(host_c.data(), device_c.data<void>(),
                        host_c.size() * sizeof(float), hipMemcpyDeviceToHost),
              "copy result");

    double max_reference = 0.0;
    double max_difference = 0.0;
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            const size_t index = column_c ? col * m + row : row * n + col;
            const double reference = expected[row * n + col];
            const double difference =
                std::fabs(static_cast<double>(host_c[index]) - reference);
            max_reference = std::max(max_reference, std::fabs(reference));
            max_difference = std::max(max_difference, difference);
        }
    }
    const double budget = tolerance * std::max(1.0, max_reference);
    const bool passed = max_difference <= budget;
    std::printf("%-26s %-4s m=%3zu n=%3zu k=%3zu b=%s c=%s  max|ref|=%.4f "
                "max|diff|=%.2e\n",
                name, dtype_name<T>(), m, n, k, column_b ? "col" : "row",
                column_c ? "col" : "row", max_reference, max_difference);
    if (!passed) {
        std::fprintf(stderr, "  gemm mismatch: budget %.2e\n", budget);
        fail(name);
    }
}

// Same launch path as the correctness cases, without the reference: reports
// the kernel time and the arithmetic throughput.
void time_case(const char* name, size_t m, size_t n, size_t k, int repeats) {
    std::vector<float> host_a = make_matrix<float>(m, k, false, 1);
    std::vector<float> host_b = make_matrix<float>(k, n, false, 2);
    std::vector<float> host_c(m * n, 0.0f);

    DeviceBuffer device_a(host_a.size() * sizeof(float));
    DeviceBuffer device_b(host_b.size() * sizeof(float));
    DeviceBuffer device_c(host_c.size() * sizeof(float));
    check_hip(hipMemcpy(device_a.data<void>(), host_a.data(),
                        host_a.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy a");
    check_hip(hipMemcpy(device_b.data<void>(), host_b.data(),
                        host_b.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy b");
    check_hip(hipMemcpy(device_c.data<void>(), host_c.data(),
                        host_c.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy c");

    const dmk::gemm::GemmParamsT<float> params{
        Tensor<const float, 2, MemorySpace::HBM>{device_a.data<const float>(),
                                                 Shape<2>{{m, k}},
                                                 HbmLayout<2>{{k, 1}}},
        Tensor<const float, 2, MemorySpace::HBM>{device_b.data<const float>(),
                                                 Shape<2>{{k, n}},
                                                 HbmLayout<2>{{n, 1}}},
        Tensor<float, 2, MemorySpace::HBM>{device_c.data<float>(),
                                           Shape<2>{{m, n}},
                                           HbmLayout<2>{{n, 1}}},
    };
    const dim3 grid(static_cast<unsigned>((n + 15) / 16),
                    static_cast<unsigned>((m + 15) / 16));

    Launch<float, 4>::run(params, grid);
    check_hip(hipDeviceSynchronize(), "warm-up");

    Event start;
    Event stop;
    check_hip(hipEventRecord(start.event), "start");
    for (int repeat = 0; repeat < repeats; ++repeat)
        Launch<float, 4>::run(params, grid);
    check_hip(hipEventRecord(stop.event), "stop");
    check_hip(hipEventSynchronize(stop.event), "synchronize");

    float total_ms = 0.0f;
    check_hip(hipEventElapsedTime(&total_ms, start.event, stop.event),
              "elapsed");
    const double flops = 2.0 * static_cast<double>(m) * n * k * repeats;
    const double seconds = static_cast<double>(total_ms) * 1e-3;
    std::printf("%-26s %7.2f ms  %8.1f GFLOP/s\n", name,
                total_ms / static_cast<float>(repeats),
                seconds > 0.0 ? flops / seconds / 1e9 : 0.0);
}

}  // namespace

int main() {
    std::printf("gemm operator (standalone)\n");

    // float m16n16k4 is the only shape every supported DCU exposes.
    run_case<float, 4>("16x16x16", 16, 16, 16, false, false, 2e-3);
    run_case<float, 4>("33x17x29", 33, 17, 29, false, false, 2e-3);
    run_case<float, 4>("64x48x96 column B", 64, 48, 96, true, false, 2e-3);
    run_case<float, 4>("32x32x8 column C", 32, 32, 8, false, true, 2e-3);
    run_case<float, 4>("7x5x3", 7, 5, 3, false, false, 2e-3);
    run_case<float, 4>("128x80x64 column B/C", 128, 80, 64, true, true, 2e-3);
    run_case<float, 4>("256x256x256", 256, 256, 256, false, false, 2e-3);
    run_case<float, 4>("512x384x256 column B", 512, 384, 256, true, false,
                       2e-3);

#if DMK_MMA_F32K8
    // float m16n16k8 (gfx928/936/938).
    run_case<float, 8>("16x16x16 k8", 16, 16, 16, false, false, 2e-3);
    run_case<float, 8>("64x48x96 k8 column B", 64, 48, 96, true, false, 2e-3);
    run_case<float, 8>("33x17x29 k8", 33, 17, 29, false, false, 2e-3);
    run_case<float, 8>("256x256x256 k8", 256, 256, 256, false, false, 2e-3);
#endif

#if DMK_MMA_F16
    // __half m16n16k16 (gfx928/936/938).
    run_case<__half, 16>("16x16x16 f16", 16, 16, 16, false, false, 8e-3);
    run_case<__half, 16>("64x48x96 f16 column B", 64, 48, 96, true, false,
                         8e-3);
    run_case<__half, 16>("128x96x64 f16", 128, 96, 64, false, false, 8e-3);
    // __hip_bfloat16 m16n16k16 (gfx928/936/938).
    run_case<__hip_bfloat16, 16>("16x16x16 bf16", 16, 16, 16, false, false,
                                 3e-2);
    run_case<__hip_bfloat16, 16>("64x48x96 bf16 column B", 64, 48, 96, true,
                                 false, 3e-2);
    run_case<__hip_bfloat16, 16>("128x96x64 bf16", 128, 96, 64, false, false,
                                 3e-2);
#endif

    time_case("1024x1024x1024 timing", 1024, 1024, 1024, 3);

    std::printf("gemm checks passed\n");
    return 0;
}
