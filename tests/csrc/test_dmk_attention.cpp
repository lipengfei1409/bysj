// Standalone attention operator test: one kernel launch per case, a CPU
// reference in double precision, and no megakernel anywhere in the path.
//
// Build/run through: python tests/csrc/run_dmk_ops_test.py --op attention
#include "attention.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using dmk::attention::AttentionParams;
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

// Deterministic values in [-1, 1] without pulling in a generator.
float sample(size_t index) {
    const float x = static_cast<float>((index * 2654435761u) % 1000) / 500.0f;
    return x - 1.0f;
}

__global__ void attention_kernel(AttentionParams params) {
    dmk::attention::run_attention(params);
}

// Fills a [rows, dim] or a transposed [rows, dim] view of `buffer`.
std::vector<float> make_matrix(size_t rows, size_t dim, bool column_major,
                               size_t salt) {
    std::vector<float> data(rows * dim);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < dim; ++col) {
            const size_t index = column_major ? col * rows + row : row * dim + col;
            data[index] = sample(row * 131 + col * 17 + salt);
        }
    }
    return data;
}

HbmLayout<2> matrix_layout(size_t rows, size_t dim, bool column_major) {
    return column_major ? HbmLayout<2>{{1, rows}} : HbmLayout<2>{{dim, 1}};
}

std::vector<double> reference_attention(const std::vector<float>& q,
                                        const std::vector<float>& k,
                                        const std::vector<float>& v,
                                        size_t rows, size_t keys, size_t dim,
                                        bool column_major) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    std::vector<double> out(rows * dim, 0.0);
    std::vector<double> scores(keys, 0.0);
    for (size_t row = 0; row < rows; ++row) {
        double max_score = -INFINITY;
        for (size_t key = 0; key < keys; ++key) {
            double dot = 0.0;
            for (size_t col = 0; col < dim; ++col) {
                const size_t qi = row * dim + col;
                const size_t ki =
                    column_major ? col * keys + key : key * dim + col;
                dot += static_cast<double>(q[qi]) * k[ki];
            }
            scores[key] = dot * scale;
            max_score = scores[key] > max_score ? scores[key] : max_score;
        }
        double total = 0.0;
        for (size_t key = 0; key < keys; ++key) {
            scores[key] = std::exp(scores[key] - max_score);
            total += scores[key];
        }
        const double inverse = total != 0.0 ? 1.0 / total : 0.0;
        for (size_t key = 0; key < keys; ++key) {
            const double weight = scores[key] * inverse;
            for (size_t col = 0; col < dim; ++col) {
                const size_t vi =
                    column_major ? col * keys + key : key * dim + col;
                out[row * dim + col] += weight * v[vi];
            }
        }
    }
    return out;
}

void run_case(const char* name, size_t rows, size_t keys, size_t dim,
              bool column_major) {
    if (rows == 0 || dim > dmk::attention::kMaxHeadDim) fail("bad case");

    std::vector<float> host_q = make_matrix(rows, dim, false, 1);
    std::vector<float> host_k = make_matrix(keys, dim, column_major, 2);
    std::vector<float> host_v = make_matrix(keys, dim, column_major, 3);
    std::vector<float> host_out(rows * dim, -12345.0f);

    const std::vector<double> expected =
        reference_attention(host_q, host_k, host_v, rows, keys, dim,
                            column_major);

    DeviceBuffer device_q(host_q.size() * sizeof(float));
    DeviceBuffer device_k(host_k.size() * sizeof(float));
    DeviceBuffer device_v(host_v.size() * sizeof(float));
    DeviceBuffer device_out(host_out.size() * sizeof(float));
    check_hip(hipMemcpy(device_q.data<void>(), host_q.data(),
                        host_q.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy q");
    check_hip(hipMemcpy(device_k.data<void>(), host_k.data(),
                        host_k.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy k");
    check_hip(hipMemcpy(device_v.data<void>(), host_v.data(),
                        host_v.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy v");
    check_hip(hipMemcpy(device_out.data<void>(), host_out.data(),
                        host_out.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy out");

    const Shape<2> shape{{rows, dim}};
    AttentionParams params{
        Tensor<const float, 2, MemorySpace::HBM>{device_q.data<const float>(),
                                                shape,
                                                HbmLayout<2>{{dim, 1}}},
        Tensor<const float, 2, MemorySpace::HBM>{
            device_k.data<const float>(), Shape<2>{{keys, dim}},
            matrix_layout(keys, dim, column_major)},
        Tensor<const float, 2, MemorySpace::HBM>{
            device_v.data<const float>(), Shape<2>{{keys, dim}},
            matrix_layout(keys, dim, column_major)},
        Tensor<float, 2, MemorySpace::HBM>{device_out.data<float>(), shape,
                                           HbmLayout<2>{{dim, 1}}},
    };
    if (!params.valid()) fail("params rejected");

    hipLaunchKernelGGL(attention_kernel, dim3(dmk::attention::attention_blocks(rows)),
                       dim3(dmk::attention::kThreadsPerBlock), 0, 0, params);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");
    check_hip(hipMemcpy(host_out.data(), device_out.data<void>(),
                        host_out.size() * sizeof(float), hipMemcpyDeviceToHost),
              "copy result");

    double max_reference = 0.0;
    double max_difference = 0.0;
    for (size_t i = 0; i < host_out.size(); ++i) {
        const double difference =
            std::fabs(static_cast<double>(host_out[i]) - expected[i]);
        max_reference = std::max(max_reference, std::fabs(expected[i]));
        max_difference = std::max(max_difference, difference);
    }
    const double budget = 3e-4 * std::max(1.0, max_reference);
    const bool passed = max_difference <= budget;
    std::printf("%-28s rows=%3zu keys=%4zu dim=%3zu %s  max|ref|=%.4f "
                "max|diff|=%.2e\n",
                name, rows, keys, dim, column_major ? "col" : "row",
                max_reference, max_difference);
    if (!passed) {
        std::fprintf(stderr, "  attention mismatch: budget %.2e\n", budget);
        fail(name);
    }
}

}  // namespace

int main() {
    std::printf("attention operator (standalone)\n");
    run_case("dense 8x64x64", 8, 64, 64, false);
    run_case("dense 3x37x32", 3, 37, 32, false);
    run_case("single row and key", 1, 1, 128, false);
    run_case("odd shapes 5x129x17", 5, 129, 17, false);
    run_case("head dim 1", 2, 9, 1, false);
    run_case("transposed K/V", 4, 50, 96, true);
    std::printf("attention checks passed\n");
    return 0;
}
