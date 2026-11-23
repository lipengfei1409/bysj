#include "ops.h"
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward_hip);
  m.def("rmsnorm_backward", &rmsnorm_backward_hip);
  m.def("gie_zonal_scatter_forward", &gie_zonal_scatter_forward_hip);
  m.def("gie_zonal_scatter_backward", &gie_zonal_scatter_backward_hip);
  m.def("segment_sum_forward", &segment_sum_forward_hip,
        py::arg("src"), py::arg("offsets"), py::arg("edge_scale") = py::none(),
        py::arg("node_scale") = py::none(), py::arg("order") = py::none());
  m.def("segment_sum_backward", &segment_sum_backward_hip,
        py::arg("grad_out"), py::arg("src"), py::arg("dst"),
        py::arg("offsets"), py::arg("edge_scale") = py::none(),
        py::arg("node_scale") = py::none(),
        py::arg("want_node_scale_grad") = false, py::arg("order") = py::none());
  m.def("segment_softmax_forward", &segment_softmax_forward_hip,
        py::arg("logit"), py::arg("env"), py::arg("offsets"),
        py::arg("null_logit") = py::none(), py::arg("order") = py::none());
  m.def("segment_softmax_backward", &segment_softmax_backward_hip,
        py::arg("alpha"), py::arg("grad_alpha"), py::arg("env"),
        py::arg("offsets"), py::arg("null_alpha") = py::none(),
        py::arg("grad_null_alpha") = py::none(), py::arg("order") = py::none());
  m.def("force_assembly", &force_assembly_hip, py::arg("grad_edge"),
        py::arg("edge_vec"), py::arg("dst_offsets"), py::arg("src_offsets"),
        py::arg("src_order"), py::arg("dst_order") = py::none());
  m.def("slot_pack_scale", &slot_pack_scale_hip, py::arg("src"),
        py::arg("slot"), py::arg("scale") = py::none());
  m.def("slot_reduce", &slot_reduce_hip, py::arg("src"),
        py::arg("weight") = py::none());
  m.def("edge_combine", &edge_combine_hip, py::arg("terms"),
        py::arg("coefficients"), py::arg("modes"), py::arg("edges"),
        py::arg("rows"), py::arg("channels"));
}
