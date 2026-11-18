// Direct DUMMA GemmOp accuracy and kernel-only performance test.
// Build/run with tests/csrc/run_dmk_gemm_test.py on a DTK 26.04 DCU host.
#include "gemm.h"
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

namespace {

void check(hipError_t status, const char* name) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", name, hipGetErrorString(status));
        std::exit(1);
    }
}

void check_rocblas(rocblas_status status, const char* name) {
    if (status != rocblas_status_success) {
        std::fprintf(stderr, "%s: rocBLAS status %d\n", name,
                     static_cast<int>(status));
        std::exit(1);
    }
}

struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        check(hipMalloc(&ptr, bytes), "hipMalloc");
    }
    ~DeviceBuffer() { if (ptr) hipFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T> T* data() const { return static_cast<T*>(ptr); }
};

struct Event {
    hipEvent_t event{};
    Event() { check(hipEventCreate(&event), "hipEventCreate"); }
    ~Event() { hipEventDestroy(event); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

__global__ void gemm_kernel(dmk::gemm::GemmParams params) {
    __shared__ dmk::gemm::GemmShared shared;
    dmk::gemm::run_gemm_wave(params, shared);
}

size_t checked_product(size_t a, size_t b) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        std::fprintf(stderr, "matrix size overflow\n");
        std::exit(1);
    }
    return a * b;
}

size_t parse_positive(const char* value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (!value[0] || *end || parsed == 0 || value[0] == '-') {
        std::fprintf(stderr, "expected a positive integer: %s\n", value);
        std::exit(1);
    }
    return static_cast<size_t>(parsed);
}

void run_case(size_t m, size_t n, size_t k, bool column_b,
              bool column_c, int repeats) {
    const size_t a_count = checked_product(m, k);
    const size_t b_count = checked_product(k, n);
    const size_t c_count = checked_product(m, n);
    std::vector<float> host_a(a_count), host_b(b_count), host_c(c_count);
    for (size_t row = 0; row < m; ++row)
        for (size_t col = 0; col < k; ++col)
            host_a[row * k + col] =
                static_cast<float>(static_cast<int>((row * 17 + col * 13) % 23) - 11) / 16.0f;
    for (size_t row = 0; row < k; ++row)
        for (size_t col = 0; col < n; ++col)
            host_b[column_b ? col * k + row : row * n + col] =
                static_cast<float>(static_cast<int>((row * 7 + col * 19) % 29) - 14) / 16.0f;

    DeviceBuffer device_a(checked_product(a_count, sizeof(float)));
    DeviceBuffer device_b(checked_product(b_count, sizeof(float)));
    DeviceBuffer device_c(checked_product(c_count, sizeof(float)));
    check(hipMemcpy(device_a.ptr, host_a.data(), a_count * sizeof(float),
                    hipMemcpyHostToDevice), "copy A");
    check(hipMemcpy(device_b.ptr, host_b.data(), b_count * sizeof(float),
                    hipMemcpyHostToDevice), "copy B");
    const dmk::tensor::Shape<2> a_shape{{m, k}}, b_shape{{k, n}}, c_shape{{m, n}};
    const dmk::tensor::StridedLayout<2> a_layout{{k, 1}};
    const dmk::tensor::StridedLayout<2> b_layout =
        column_b ? dmk::tensor::StridedLayout<2>{{1, k}}
                 : dmk::tensor::StridedLayout<2>{{n, 1}};
    const dmk::tensor::StridedLayout<2> c_layout =
        column_c ? dmk::tensor::StridedLayout<2>{{1, m}}
                 : dmk::tensor::StridedLayout<2>{{n, 1}};
    dmk::gemm::GemmParams params{
        {device_a.data<float>(), a_shape, a_layout},
        {device_b.data<float>(), b_shape, b_layout},
        {device_c.data<float>(), c_shape, c_layout}};
    const dim3 block(64);
    const dim3 grid((n + 15) / 16, (m + 15) / 16);
    hipLaunchKernelGGL(gemm_kernel, grid, block, 0, 0, params);
    check(hipGetLastError(), "GEMM launch");
    check(hipDeviceSynchronize(), "GEMM execution");
    check(hipMemcpy(host_c.data(), device_c.ptr, c_count * sizeof(float),
                    hipMemcpyDeviceToHost), "copy C");

    const bool full_check = c_count <= 4096;
    const size_t samples = full_check ? c_count : 256;
    double max_abs = 0.0, max_rel = 0.0;
    size_t failures = 0;
    for (size_t index = 0; index < samples; ++index) {
        const size_t row = full_check ? index / n :
                           index == 1 ? m - 1 : (index * 131 + index / 13) % m;
        const size_t col = full_check ? index % n :
                           index == 1 ? n - 1 : (index * 197 + index / 17) % n;
        double expected = 0.0;
        for (size_t inner = 0; inner < k; ++inner)
            expected += static_cast<double>(host_a[row * k + inner]) *
                        host_b[column_b ? col * k + inner : inner * n + col];
        const float actual = host_c[column_c ? col * m + row : row * n + col];
        const double abs_error = std::fabs(static_cast<double>(actual) - expected);
        const double rel_error = abs_error / std::fmax(std::fabs(expected), 1e-6);
        if (abs_error > max_abs) max_abs = abs_error;
        if (rel_error > max_rel) max_rel = rel_error;
        const double tolerance = 1e-3 + 1e-3 * std::fabs(expected);
        if (!std::isfinite(actual) || abs_error > tolerance) {
            if (failures++ < 5)
                std::fprintf(stderr, "C[%zu,%zu] got %.8g expected %.8g (abs %.3g)\n",
                             row, col, actual, expected, abs_error);
        }
    }
    std::printf("M=%zu N=%zu K=%zu B=%s C=%s: %s, checked=%zu, max_abs=%.3g, max_rel=%.3g\n",
                m, n, k, column_b ? "column" : "row",
                column_c ? "column" : "row", failures ? "FAIL" : "PASS",
                samples, max_abs, max_rel);
    if (failures) std::exit(1);

    if (repeats > 0) {
        for (int warmup = 0; warmup < 5; ++warmup)
            hipLaunchKernelGGL(gemm_kernel, grid, block, 0, 0, params);
        check(hipGetLastError(), "warmup launch");
        check(hipDeviceSynchronize(), "warmup execution");
        Event start, stop;
        check(hipEventRecord(start.event), "start event");
        for (int i = 0; i < repeats; ++i)
            hipLaunchKernelGGL(gemm_kernel, grid, block, 0, 0, params);
        check(hipGetLastError(), "timed launch");
        check(hipEventRecord(stop.event), "stop event");
        check(hipEventSynchronize(stop.event), "timed execution");
        float total_ms = 0.0f;
        check(hipEventElapsedTime(&total_ms, start.event, stop.event), "elapsed time");
        const double average_ms = total_ms / repeats;
        const double gflops = (2.0 * m * n * k) / (average_ms * 1.0e6);
        std::printf("kernel-only: %.4f ms/launch, %.2f GFLOP/s (%d runs)\n",
                    average_ms, gflops, repeats);

        // The measured case is row-major. rocBLAS is column-major, so
        // C^T = B^T * A^T uses the same buffers with swapped operands.
        if (!column_b && !column_c) {
            const size_t max_rocblas_int =
                static_cast<size_t>(std::numeric_limits<rocblas_int>::max());
            if (m > max_rocblas_int || n > max_rocblas_int ||
                k > max_rocblas_int) {
                std::fprintf(stderr, "matrix dimension exceeds rocblas_int\n");
                std::exit(1);
            }
            rocblas_handle handle = nullptr;
            check_rocblas(rocblas_create_handle(&handle), "rocblas_create_handle");
            check_rocblas(rocblas_set_stream(handle, 0), "rocblas_set_stream");
            check_rocblas(rocblas_set_pointer_mode(handle,
                            rocblas_pointer_mode_host), "rocblas_set_pointer_mode");
            const float alpha = 1.0f, beta = 0.0f;
            const rocblas_int rm = static_cast<rocblas_int>(m);
            const rocblas_int rn = static_cast<rocblas_int>(n);
            const rocblas_int rk = static_cast<rocblas_int>(k);
            auto run_rocblas = [&]() {
                check_rocblas(rocblas_sgemm(
                    handle, rocblas_operation_none, rocblas_operation_none,
                    rn, rm, rk, &alpha,
                    device_b.data<float>(), rn,
                    device_a.data<float>(), rk,
                    &beta, device_c.data<float>(), rn),
                    "rocblas_sgemm");
            };
            run_rocblas();
            check(hipDeviceSynchronize(), "rocBLAS correctness execution");
            std::vector<float> rocblas_c(c_count);
            check(hipMemcpy(rocblas_c.data(), device_c.ptr,
                            c_count * sizeof(float), hipMemcpyDeviceToHost),
                  "copy rocBLAS C");
            size_t mismatches = 0;
            double max_difference = 0.0;
            for (size_t index = 0; index < c_count; ++index) {
                const double expected = host_c[index];
                const double actual = rocblas_c[index];
                const double difference = std::fabs(actual - expected);
                if (difference > max_difference) max_difference = difference;
                if (!std::isfinite(actual) ||
                    difference > 2e-3 + 1e-3 * std::fabs(expected)) {
                    if (mismatches++ < 5)
                        std::fprintf(stderr,
                                     "rocBLAS C[%zu,%zu] got %.8g, DMK %.8g\n",
                                     index / n, index % n, actual, expected);
                }
            }
            std::printf("rocBLAS vs DMK: %s, max_abs_diff=%.3g\n",
                        mismatches ? "FAIL" : "PASS", max_difference);
            if (mismatches) std::exit(1);

            for (int warmup = 0; warmup < 5; ++warmup) run_rocblas();
            check(hipDeviceSynchronize(), "rocBLAS warmup");
            Event rocblas_start, rocblas_stop;
            check(hipEventRecord(rocblas_start.event), "rocBLAS start event");
            for (int i = 0; i < repeats; ++i) run_rocblas();
            check(hipEventRecord(rocblas_stop.event), "rocBLAS stop event");
            check(hipEventSynchronize(rocblas_stop.event), "rocBLAS timed execution");
            float rocblas_total_ms = 0.0f;
            check(hipEventElapsedTime(&rocblas_total_ms,
                                      rocblas_start.event, rocblas_stop.event),
                  "rocBLAS elapsed time");
            const double rocblas_average_ms = rocblas_total_ms / repeats;
            const double rocblas_gflops =
                (2.0 * m * n * k) / (rocblas_average_ms * 1.0e6);
            std::printf("rocBLAS SGEMM: %.4f ms/launch, %.2f GFLOP/s (%d runs)\n",
                        rocblas_average_ms, rocblas_gflops, repeats);
            std::printf("rocBLAS / DMK throughput: %.2fx\n",
                        average_ms / rocblas_average_ms);
            check_rocblas(rocblas_destroy_handle(handle), "rocblas_destroy_handle");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 5) {
        std::fprintf(stderr, "usage: %s [M N K REPEATS]\n", argv[0]);
        return 2;
    }
    int device_count = 0;
    check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count < 1) {
        std::fprintf(stderr, "no DCU available\n");
        return 2;
    }
    run_case(16, 16, 4, false, false, 0);
    run_case(19, 18, 7, false, false, 0);
    run_case(31, 33, 13, true, true, 0);
    run_case(64, 64, 17, true, false, 0);
    const size_t m = argc == 5 ? parse_positive(argv[1]) : 1024;
    const size_t n = argc == 5 ? parse_positive(argv[2]) : 1024;
    const size_t k = argc == 5 ? parse_positive(argv[3]) : 1024;
    const size_t repeat_count = argc == 5 ? parse_positive(argv[4]) : 30;
    if (repeat_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "REPEATS exceeds int range\n");
        return 2;
    }
    const int repeats = static_cast<int>(repeat_count);
    run_case(m, n, k, false, false, repeats);
}
