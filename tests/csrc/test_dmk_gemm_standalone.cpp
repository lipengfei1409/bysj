// Standalone GEMM operator test: one kernel launch per case, a CPU reference in
// double precision, and no megakernel anywhere in the path.
//
// The operator itself is dmk::gemm::run_gemm_wave (16x16 output tile per wave,
// DUMMA m16n16k4); this file only launches it on its own.
//
// Build/run through: python tests/csrc/run_dmk_ops_test.py --op gemm
#include "gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using dmk::gemm::GemmParams;
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

__global__ void gemm_kernel(GemmParams params) {
    __shared__ dmk::gemm::GemmShared shared;
    dmk::gemm::run_gemm_wave(params, shared);
}

// Fills [rows, cols], stored row-major or column-major.
std::vector<float> make_matrix(size_t rows, size_t cols, bool column_major,
                               size_t salt) {
    std::vector<float> data(rows * cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            const size_t index =
                column_major ? col * rows + row : row * cols + col;
            data[index] = sample(row * 131 + col * 17 + salt);
        }
    }
    return data;
}

HbmLayout<2> matrix_layout(size_t rows, size_t cols, bool column_major) {
    return column_major ? HbmLayout<2>{{1, rows}} : HbmLayout<2>{{cols, 1}};
}

void run_case(const char* name, size_t m, size_t n, size_t k, bool column_b,
              bool column_c) {
    std::vector<float> host_a = make_matrix(m, k, false, 1);
    std::vector<float> host_b = make_matrix(k, n, column_b, 2);
    std::vector<float> host_c(m * n, -12345.0f);

    // Reference: C = A * B in double.
    std::vector<double> expected(m * n, 0.0);
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            double sum = 0.0;
            for (size_t inner = 0; inner < k; ++inner) {
                const size_t bi =
                    column_b ? col * k + inner : inner * n + col;
                sum += static_cast<double>(host_a[row * k + inner]) *
                       host_b[bi];
            }
            expected[row * n + col] = sum;
        }
    }

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

    const GemmParams params{
        Tensor<const float, 2, MemorySpace::HBM>{device_a.data<const float>(),
                                                Shape<2>{{m, k}},
                                                HbmLayout<2>{{k, 1}}},
        Tensor<const float, 2, MemorySpace::HBM>{
            device_b.data<const float>(), Shape<2>{{k, n}},
            matrix_layout(k, n, column_b)},
        Tensor<float, 2, MemorySpace::HBM>{device_c.data<float>(),
                                           Shape<2>{{m, n}},
                                           matrix_layout(m, n, column_c)},
    };
    if (!params.valid()) fail("params rejected");

    const dim3 grid(static_cast<unsigned>((n + 15) / 16),
                    static_cast<unsigned>((m + 15) / 16));
    hipLaunchKernelGGL(gemm_kernel, grid, dim3(64), 0, 0, params);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");
    check_hip(hipMemcpy(host_c.data(), device_c.data<void>(),
                        host_c.size() * sizeof(float), hipMemcpyDeviceToHost),
              "copy result");

    double max_reference = 0.0;
    double max_difference = 0.0;
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            const size_t index =
                column_c ? col * m + row : row * n + col;
            const double reference = expected[row * n + col];
            const double difference =
                std::fabs(static_cast<double>(host_c[index]) - reference);
            max_reference = std::max(max_reference, std::fabs(reference));
            max_difference = std::max(max_difference, difference);
        }
    }
    const double budget = 2e-3 * std::max(1.0, max_reference);
    const bool passed = max_difference <= budget;
    std::printf("%-28s m=%3zu n=%3zu k=%3zu b=%s c=%s  max|ref|=%.4f "
                "max|diff|=%.2e\n",
                name, m, n, k, column_b ? "col" : "row",
                column_c ? "col" : "row", max_reference, max_difference);
    if (!passed) {
        std::fprintf(stderr, "  gemm mismatch: budget %.2e\n", budget);
        fail(name);
    }
}

}  // namespace

int main() {
    std::printf("gemm operator (standalone)\n");
    run_case("16x16x16", 16, 16, 16, false, false);
    run_case("33x17x29", 33, 17, 29, false, false);
    run_case("64x48x96 column B", 64, 48, 96, true, false);
    run_case("32x32x8 column C", 32, 32, 8, false, true);
    run_case("7x5x3", 7, 5, 3, false, false);
    run_case("128x80x64 column B and C", 128, 80, 64, true, true);
    std::printf("gemm checks passed\n");
    return 0;
}
