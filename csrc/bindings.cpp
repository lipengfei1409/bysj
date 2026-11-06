#include "include/ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm_dumma_hip);
}
