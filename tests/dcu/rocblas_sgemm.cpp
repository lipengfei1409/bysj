#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <torch/extension.h>

#include <limits>

namespace {

void check_rocblas(rocblas_status status, const char *operation) {
  TORCH_CHECK(status == rocblas_status_success, operation,
              " failed with rocBLAS status ", static_cast<int>(status));
}

} // namespace

class RocblasSgemm {
public:
  RocblasSgemm() = default;

  RocblasSgemm(const RocblasSgemm &) = delete;
  RocblasSgemm &operator=(const RocblasSgemm &) = delete;

  ~RocblasSgemm() {
    if (handle_ != nullptr) {
      rocblas_destroy_handle(handle_);
    }
  }

  torch::Tensor operator()(const torch::Tensor &a, const torch::Tensor &b) {
    TORCH_CHECK(a.defined() && b.defined(), "rocBLAS inputs must be defined");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "expected A[M,K] and B[K,N] tensors");
    TORCH_CHECK(a.size(1) == b.size(0),
                "A[M,K] and B[K,N] must have matching inner dimensions");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                    b.scalar_type() == torch::kFloat32,
                "direct rocblas_sgemm requires float32 tensors");
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device() == b.device(),
                "A and B must be on the same HIP/DCU device");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "direct rocBLAS requires contiguous inputs");

    const int64_t m64 = a.size(0);
    const int64_t k64 = a.size(1);
    const int64_t n64 = b.size(1);
    const int64_t max_dim = std::numeric_limits<int>::max();
    TORCH_CHECK(m64 <= max_dim && n64 <= max_dim && k64 <= max_dim,
                "rocblas_sgemm uses 32-bit dimensions");

    auto output = torch::empty({m64, n64}, a.options());
    if (m64 == 0 || n64 == 0) {
      return output;
    }
    if (k64 == 0) {
      return output.zero_();
    }

    const int device_index = a.get_device();
    ensure_handle(device_index);
    const hipStream_t stream =
        at::hip::getCurrentHIPStream(device_index).stream();
    if (!stream_initialized_ || stream_ != stream) {
      check_rocblas(rocblas_set_stream(handle_, stream), "rocblas_set_stream");
      stream_ = stream;
      stream_initialized_ = true;
    }

    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Torch tensors are row-major. Reinterpret the same storage as column-major
    // A^T[K,M], B^T[N,K], and C^T[N,M], then compute C^T = B^T * A^T.
    check_rocblas(
        rocblas_sgemm(handle_, rocblas_operation_none, rocblas_operation_none,
                      n, m, k, &alpha, b.data_ptr<float>(), n,
                      a.data_ptr<float>(), k, &beta, output.data_ptr<float>(), n),
        "rocblas_sgemm");
    return output;
  }

  void close() {
    if (handle_ != nullptr) {
      const rocblas_status status = rocblas_destroy_handle(handle_);
      handle_ = nullptr;
      stream_ = nullptr;
      stream_initialized_ = false;
      check_rocblas(status, "rocblas_destroy_handle");
    }
  }

private:
  void ensure_handle(int device_index) {
    if (handle_ == nullptr) {
      check_rocblas(rocblas_create_handle(&handle_), "rocblas_create_handle");
      device_index_ = device_index;
      check_rocblas(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host),
                    "rocblas_set_pointer_mode");
      return;
    }
    TORCH_CHECK(device_index_ == device_index,
                "one RocblasSgemm instance cannot switch DCU devices");
  }

  rocblas_handle handle_ = nullptr;
  int device_index_ = -1;
  hipStream_t stream_ = nullptr;
  bool stream_initialized_ = false;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<RocblasSgemm>(module, "RocblasSgemm")
      .def(pybind11::init<>())
      .def("__call__", &RocblasSgemm::operator())
      .def("close", &RocblasSgemm::close);
}
