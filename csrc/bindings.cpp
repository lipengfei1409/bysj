#include "include/ops.h"
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef DPA4_ENABLE_DUMMA
  m.def("gemm", &gemm_dumma_hip);
#endif
#ifdef DPA4_ENABLE_RMSNORM
  m.def("rmsnorm_forward", &rmsnorm_forward_hip);
  m.def("rmsnorm_backward", &rmsnorm_backward_hip);
#endif
#ifdef DPA4_ENABLE_GIE
  m.def("gie_zonal_scatter_forward", &gie_zonal_scatter_forward_hip);
  m.def("gie_zonal_scatter_backward", &gie_zonal_scatter_backward_hip);
#endif
}
