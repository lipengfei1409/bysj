#include "ops.h"
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward_hip);
  m.def("rmsnorm_backward", &rmsnorm_backward_hip);
  m.def("gie_zonal_scatter_forward", &gie_zonal_scatter_forward_hip);
  m.def("gie_zonal_scatter_backward", &gie_zonal_scatter_backward_hip);
  m.def("so2_envelope_reduce_forward", &so2_envelope_reduce_forward_hip);
  m.def("so2_envelope_reduce_backward", &so2_envelope_reduce_backward_hip);
}
