// Standalone attention operator test: one kernel launch per case, a CPU
// reference in double precision, and no megakernel anywhere in the path.
//
// The operator is dmk::attention::run_attention, templated on the element type
// (float, __half, __hip_bfloat16); the softmax arithmetic stays in float and
// the output follows the input dtype.
//
// Build/run through: python tests/csrc/run_dmk_ops_test.py --op attention
#include "attention.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

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

// Deterministic values in [-1, 1] without pulling in a generator.
float sample(size_t index) {
    const float x = static_cast<float>((index * 2654435761u) % 1000) / 500.0f;
    return x - 1.0f;
}

template <typename T>
const char* dtype_name() {
    if (std::is_same<T, float>::value) return "f32";
    if (std::is_same<T, __half>::value) return "f16";
    return "bf16";
}

// Fills a [rows, dim] matrix, stored row-major or column-major.
template <typename T>
std::vector<T> make_matrix(size_t rows, size_t dim, bool column_major,
                           size_t salt) {
    std::vector<T> data(rows * dim);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < dim; ++col) {
            const size_t index = column_major ? col * rows + row : row * dim + col;
            data[index] =
                dmk::dtype::from_float<T>(sample(row * 131 + col * 17 + salt));
        }
    }
    return data;
}

HbmLayout<2> matrix_layout(size_t rows, size_t dim, bool column_major) {
    return column_major ? HbmLayout<2>{{1, rows}} : HbmLayout<2>{{dim, 1}};
}

template <typename T>
std::vector<double> reference_attention(const std::vector<T>& q,
                                        const std::vector<T>& k,
                                        const std::vector<T>& v, size_t rows,
                                        size_t keys, size_t dim,
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
                const size_t ki = column_major ? col * keys + key : key * dim + col;
                dot += static_cast<double>(dmk::dtype::to_float(q[qi])) *
                       static_cast<double>(dmk::dtype::to_float(k[ki]));
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
                const size_t vi = column_major ? col * keys + key : key * dim + col;
                out[row * dim + col] +=
                    weight * static_cast<double>(dmk::dtype::to_float(v[vi]));
            }
        }
    }
    return out;
}

__global__ void attention_kernel_f32(
    dmk::attention::AttentionParamsT<float> params) {
    dmk::attention::run_attention(params);
}

__global__ void attention_kernel_f16(
    dmk::attention::AttentionParamsT<__half> params) {
    dmk::attention::run_attention(params);
}

__global__ void attention_kernel_bf16(
    dmk::attention::AttentionParamsT<__hip_bfloat16> params) {
    dmk::attention::run_attention(params);
}

template <typename T>
void launch_attention(const dmk::attention::AttentionParamsT<T>& params,
                      unsigned blocks) {
    const dim3 threads(dmk::attention::kThreadsPerBlock);
    if constexpr (std::is_same<T, float>::value) {
        hipLaunchKernelGGL(attention_kernel_f32, dim3(blocks), threads, 0, 0,
                           params);
    } else if constexpr (std::is_same<T, __half>::value) {
        hipLaunchKernelGGL(attention_kernel_f16, dim3(blocks), threads, 0, 0,
                           params);
    } else {
        hipLaunchKernelGGL(attention_kernel_bf16, dim3(blocks), threads, 0, 0,
                           params);
    }
}

template <typename T>
void run_case(const char* name, size_t rows, size_t keys, size_t dim,
              bool column_major, double tolerance) {
    if (rows == 0 || dim > dmk::attention::kMaxHeadDim) fail("bad case");

    std::vector<T> host_q = make_matrix<T>(rows, dim, false, 1);
    std::vector<T> host_k = make_matrix<T>(keys, dim, column_major, 2);
    std::vector<T> host_v = make_matrix<T>(keys, dim, column_major, 3);
    std::vector<T> host_out(rows * dim,
                            dmk::dtype::from_float<T>(-12345.0f));

    const std::vector<double> expected =
        reference_attention(host_q, host_k, host_v, rows, keys, dim,
                            column_major);

    DeviceBuffer device_q(host_q.size() * sizeof(T));
    DeviceBuffer device_k(host_k.size() * sizeof(T));
    DeviceBuffer device_v(host_v.size() * sizeof(T));
    DeviceBuffer device_out(host_out.size() * sizeof(T));
    check_hip(hipMemcpy(device_q.data<void>(), host_q.data(),
                        host_q.size() * sizeof(T), hipMemcpyHostToDevice),
              "copy q");
    check_hip(hipMemcpy(device_k.data<void>(), host_k.data(),
                        host_k.size() * sizeof(T), hipMemcpyHostToDevice),
              "copy k");
    check_hip(hipMemcpy(device_v.data<void>(), host_v.data(),
                        host_v.size() * sizeof(T), hipMemcpyHostToDevice),
              "copy v");
    check_hip(hipMemcpy(device_out.data<void>(), host_out.data(),
                        host_out.size() * sizeof(T), hipMemcpyHostToDevice),
              "copy out");

    const Shape<2> shape{{rows, dim}};
    const dmk::attention::AttentionParamsT<T> params{
        Tensor<const T, 2, MemorySpace::HBM>{device_q.data<const T>(), shape,
                                             HbmLayout<2>{{dim, 1}}},
        Tensor<const T, 2, MemorySpace::HBM>{
            device_k.data<const T>(), Shape<2>{{keys, dim}},
            matrix_layout(keys, dim, column_major)},
        Tensor<const T, 2, MemorySpace::HBM>{
            device_v.data<const T>(), Shape<2>{{keys, dim}},
            matrix_layout(keys, dim, column_major)},
        Tensor<T, 2, MemorySpace::HBM>{device_out.data<T>(), shape,
                                       HbmLayout<2>{{dim, 1}}},
    };
    if (!params.valid()) fail("params rejected");

    launch_attention(params, dmk::attention::attention_blocks(rows));
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");
    check_hip(hipMemcpy(host_out.data(), device_out.data<void>(),
                        host_out.size() * sizeof(T), hipMemcpyDeviceToHost),
              "copy result");

    double max_reference = 0.0;
    double max_difference = 0.0;
    for (size_t i = 0; i < host_out.size(); ++i) {
        const double difference =
            std::fabs(static_cast<double>(dmk::dtype::to_float(host_out[i])) -
                      expected[i]);
        max_reference = std::max(max_reference, std::fabs(expected[i]));
        max_difference = std::max(max_difference, difference);
    }
    const double budget = tolerance * std::max(1.0, max_reference);
    const bool passed = max_difference <= budget;
    std::printf("%-26s %-4s rows=%3zu keys=%4zu dim=%3zu %s  max|ref|=%.4f "
                "max|diff|=%.2e\n",
                name, dtype_name<T>(), rows, keys, dim,
                column_major ? "col" : "row", max_reference, max_difference);
    if (!passed) {
        std::fprintf(stderr, "  attention mismatch: budget %.2e\n", budget);
        fail(name);
    }
}

}  // namespace

int main() {
    std::printf("attention operator (standalone)\n");
    run_case<float>("dense 8x64x64", 8, 64, 64, false, 3e-4);
    run_case<float>("dense 3x37x32", 3, 37, 32, false, 3e-4);
    run_case<float>("single row and key", 1, 1, 128, false, 3e-4);
    run_case<float>("odd shapes 5x129x17", 5, 129, 17, false, 3e-4);
    run_case<float>("head dim 1", 2, 9, 1, false, 3e-4);
    run_case<float>("transposed K/V", 4, 50, 96, true, 3e-4);

    run_case<__half>("dense 8x64x64 f16", 8, 64, 64, false, 8e-3);
    run_case<__half>("transposed K/V f16", 4, 50, 96, true, 8e-3);
    run_case<__hip_bfloat16>("dense 8x64x64 bf16", 8, 64, 64, false, 3e-2);
    run_case<__hip_bfloat16>("transposed K/V bf16", 4, 50, 96, true, 3e-2);

    std::printf("attention checks passed\n");
    return 0;
}
