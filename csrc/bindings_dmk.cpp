// Python entry points for the persistent megakernel, and nothing else.
//
// `bindings.cpp` is the full set for this tree; this file exists for builds that
// compile only the framework (`csrc/dmk/`) and none of the standalone kernels, so
// that the pybind module still links.  A registration may only name a symbol whose
// translation unit is in the same build, which is why the narrow build needs its
// own module rather than a flag on the full one.
//
// When the standalone kernels come back into the build, switch `setup.py` back to
// `bindings.cpp` -- the two files register the same `dmk_gemm`, so nothing else
// changes.

#include "ops.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "DPA4 native operators: persistent megakernel only";
  m.def("dmk_gemm", &dmk_gemm_hip, py::arg("instructions"), py::arg("a"),
        py::arg("b"), py::arg("c"));
}
