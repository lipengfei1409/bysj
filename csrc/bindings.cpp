#include "include/ops.h"
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("edge_geometry", &edge_geometry_hip);
  m.def("edge_geometry_backward", &edge_geometry_backward_hip);
  m.def("radial", &radial_hip);
  m.def("radial_backward", &radial_backward_hip);
  m.def("edge_features", &edge_features_hip);
  m.def("edge_features_backward", &edge_features_backward_hip);
  m.def("edge_features_persistent", &edge_features_persistent_hip);
  m.def("segment_sum", &segment_sum_hip);
  m.def("force_virial", &force_virial_hip);
  m.def("rmsnorm", &rmsnorm_hip);
  m.def("rmsnorm_backward", &rmsnorm_backward_hip);
  m.def("linear", &linear_hip);
#if defined(DPA4_ENABLE_ROCBLAS) && DPA4_ENABLE_ROCBLAS
  m.def("blas_available", []() { return true; });
  m.def("blas_gemm", &blas_gemm_hip);
#else
  m.def("blas_available", []() { return false; });
#endif
}
