// Standalone Tensor/HIP smoke test. It does not use the DUMMA GemmOp.
// Build: hipcc -O2 -std=c++17 -I csrc/dmk tests/csrc/test_dmk_tensor_matmul.cpp -o /tmp/test_dmk_tensor_matmul
#include "tensor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

using Matrix = dmk::Tensor<const float, 2, dmk::MemorySpace::HBM>;
using MutableMatrix = dmk::Tensor<float, 2, dmk::MemorySpace::HBM>;

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
        exit(1);
    }
}

__global__ void matmul_kernel(Matrix a, Matrix b, MutableMatrix c) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= c.shape.dims[0] || col >= c.shape.dims[1]) return;

    float sum = 0.0f;
    for (size_t k = 0; k < a.shape.dims[1]; ++k) {
        sum += a.data[a.offset(dmk::Shape<2>{{row, k}})] *
               b.data[b.offset(dmk::Shape<2>{{k, col}})];
    }
    c.data[c.offset(dmk::Shape<2>{{row, col}})] = sum;
}

}  // namespace

int main() {
    constexpr size_t m = 1024;
    constexpr size_t k = 1024;
    constexpr size_t n = 1024;
    vector<float> host_a(m * k);
    vector<float> host_b(k * n);
    vector<float> actual(m * n, 0.0f);
    vector<float> expected(m * n, 0.0f);

    for (size_t i = 0; i < host_a.size(); ++i)
        host_a[i] = static_cast<float>(static_cast<int>(i % 7) - 2) * 0.5f;
    for (size_t i = 0; i < host_b.size(); ++i)
        host_b[i] = static_cast<float>(static_cast<int>(i % 5) - 1) * 0.25f;

    // A repeats every 7 elements and B every 5. Their product repeats every
    // 35 reduction steps, so a full reference needs at most 35 products/cell.
    constexpr size_t period = 35;
    const size_t full_periods = k / period;
    const size_t tail = k % period;
    const size_t first_period = k < period ? k : period;
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            float period_sum = 0.0f;
            float tail_sum = 0.0f;
            for (size_t inner = 0; inner < first_period; ++inner) {
                const float product = host_a[row * k + inner] *
                                      host_b[inner * n + col];
                period_sum += product;
                if (inner < tail) tail_sum += product;
            }
            expected[row * n + col] = period_sum * full_periods + tail_sum;
        }
    }

    // Check the shortened reference against full dot products at edge/middle cells.
    const size_t sample_rows[] = {0, m / 2, m - 1};
    const size_t sample_cols[] = {0, n / 2, n - 1};
    for (size_t row : sample_rows) {
        for (size_t col : sample_cols) {
            float direct = 0.0f;
            for (size_t inner = 0; inner < k; ++inner)
                direct += host_a[row * k + inner] * host_b[inner * n + col];
            assert(fabsf(direct - expected[row * n + col]) < 1e-5f);
        }
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_a), host_a.size() * sizeof(float)),
              "hipMalloc(A)");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_b), host_b.size() * sizeof(float)),
              "hipMalloc(B)");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device_c), actual.size() * sizeof(float)),
              "hipMalloc(C)");
    check_hip(hipMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float),
                        hipMemcpyHostToDevice), "hipMemcpy(A)");
    check_hip(hipMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float),
                        hipMemcpyHostToDevice), "hipMemcpy(B)");

    const dmk::Shape<2> a_shape{{m, k}};
    const dmk::Shape<2> b_shape{{k, n}};
    const dmk::Shape<2> c_shape{{m, n}};
    Matrix a{device_a, a_shape, dmk::HbmLayout<2>::contiguous(a_shape)};
    Matrix b{device_b, b_shape, dmk::HbmLayout<2>::contiguous(b_shape)};
    MutableMatrix c{device_c, c_shape, dmk::HbmLayout<2>::contiguous(c_shape)};

    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(a, b, c);
    check_hip(hipGetLastError(), "matmul_kernel launch");
    check_hip(hipDeviceSynchronize(), "matmul_kernel execution");
    check_hip(hipMemcpy(actual.data(), device_c, actual.size() * sizeof(float),
                        hipMemcpyDeviceToHost), "hipMemcpy(C)");

    bool correct = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1e-5f * (1.0f + fabsf(expected[i]));
        if (fabsf(actual[i] - expected[i]) > tolerance) {
            fprintf(stderr, "C[%zu]: got %.7g, expected %.7g\n", i,
                    actual[i], expected[i]);
            correct = false;
        }
    }

    check_hip(hipFree(device_a), "hipFree(A)");
    check_hip(hipFree(device_b), "hipFree(B)");
    check_hip(hipFree(device_c), "hipFree(C)");
    if (!correct) return 1;
    printf("Tensor matrix multiplication (%zux%zu * %zux%zu): PASS\n", m, k, k, n);
}
