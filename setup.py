"""Build the DPA4 native HIP operators by default."""
import os
import pathlib

from setuptools import find_packages, setup

# Native kernels use HIP syntax in .cpp translation units. Follow the DTK
# manual's hipcc test.cpp convention rather than dispatching by suffix.
hip_cxx = os.environ.get("HIPCXX") or "hipcc"
os.environ["CXX"] = hip_cxx

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Absolute, because a relative `-Icsrc` is resolved against whatever working
# directory the build tool runs the compiler in -- under torch's ninja backend
# that is the build temp dir, not the project root, so the flag silently points
# at nothing and every `#include "ops/..."` from a subdirectory fails with
# "'ops.h' file not found".
CSRC = pathlib.Path(__file__).resolve().parent / "csrc"

# `DPA4_GEMM_SCALAR=1` builds the persistent GEMM against the FP32 scalar loop
# instead of DUMMA.  It exists because the matrix-unit API in `du_mma.h` differs
# between DTK releases: when the installed header does not expose the fragment
# API, the kernel can still be built, run and measured, and the scheduling side
# (pages, waves, split) can be validated on its own.
CSCALAR_ARGS = ["-DDPA4_GEMM_SCALAR"] if os.environ.get("DPA4_GEMM_SCALAR") == "1" else []

# DTK's du_mma.h hard-errors on anything outside gfx926/gfx928/gfx936/gfx938, and
# torch compiles for every target in PYTORCH_ROCM_ARCH (defaulting to all the
# arches the toolkit knows, gfx906 included).  Fail here, with the fix in the
# message, instead of as "undefined template DUFragment" deep in a device pass.
_DUMMA_ARCHES = ("gfx926", "gfx928", "gfx936", "gfx938")
_ARCH_ENV = os.environ.get("PYTORCH_ROCM_ARCH", "")
if not _ARCH_ENV and not CSCALAR_ARGS:
    # No explicit arch: ask the device.  torch otherwise compiles for every arch
    # the toolkit knows, gfx906 included, and DTK's du_mma.h hard-errors on it.
    try:
        _detected = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:  # noqa: BLE001 - no device visible; leave it to the build
        _detected = ""
    if _detected:
        os.environ["PYTORCH_ROCM_ARCH"] = _detected
        _ARCH_ENV = _detected
if not CSCALAR_ARGS and _ARCH_ENV:
    _targets = [a.strip() for a in _ARCH_ENV.replace(";", " ").split(",") if a.strip()]
    _bad = [a for a in _targets if a not in _DUMMA_ARCHES]
    if _bad:
        raise SystemExit(
            "PYTORCH_ROCM_ARCH=" + _ARCH_ENV + " contains "
            + ", ".join(_bad)
            + ", which DTK's du_mma.h does not support ("
            + ", ".join(_DUMMA_ARCHES)
            + ").\nSet PYTORCH_ROCM_ARCH to this card's arch (rocminfo | grep gfx), "
            "or build the scalar GEMM with DPA4_GEMM_SCALAR=1."
        )

if not torch.version.hip:
    raise RuntimeError("native build requires a ROCm/DTK PyTorch installation")

sources = [
    # Only the framework for now: `csrc/ops/` is header-only, so the two units
    # below are the persistent megakernel and its Python entry points.  The
    # standalone kernels (rmsnorm, gie, segment_*, force_assembly, slot_ops,
    # edge_combine, monomial, blockdiag, so2_conv) are deliberately out of the
    # build; `csrc/bindings.cpp` is the full registration set for when they
    # return -- swap it in below together with their sources.
    "csrc/bindings_dmk.cpp",
    "csrc/dmk/megakernel.cpp",
]
ext_modules = [CUDAExtension(
    "kernels._C",
    sources=sources,
    # ops/ and dmk/ are addressed as "ops/..." and "dmk/..." from every
    # translation unit, including the headers they include from each other.
    include_dirs=[str(CSRC)],
    define_macros=[],
    extra_compile_args={"cxx": ["-O3", "-std=c++17"] + CSCALAR_ARGS},
)]
cmdclass = {"build_ext": BuildExtension}

setup(name="dcu-dpa4", version="0.1.0", python_requires=">=3.9",
      package_dir={"": "python"}, packages=find_packages("python"),
      ext_modules=ext_modules, cmdclass=cmdclass)
