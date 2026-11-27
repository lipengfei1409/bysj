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

if not torch.version.hip:
    raise RuntimeError("native build requires a ROCm/DTK PyTorch installation")

sources = [
    "csrc/bindings.cpp",
    "csrc/rmsnorm.cpp",
    "csrc/gie.cpp",
    "csrc/segment_sum.cpp",
    "csrc/segment_softmax.cpp",
    "csrc/force_assembly.cpp",
    "csrc/slot_ops.cpp",
    "csrc/edge_combine.cpp",
    "csrc/monomial.cpp",
    "csrc/blockdiag.cpp",
    "csrc/so2_conv.cpp",
    "csrc/dmk/gemm_launch.cpp",
]
ext_modules = [CUDAExtension(
    "kernels._C",
    sources=sources,
    # ops/ and dmk/ are addressed as "ops/..." and "dmk/..." from every
    # translation unit, including the headers they include from each other.
    include_dirs=[str(CSRC)],
    define_macros=[],
    extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
)]
cmdclass = {"build_ext": BuildExtension}

setup(name="dcu-dpa4", version="0.1.0", python_requires=">=3.9",
      package_dir={"": "python"}, packages=find_packages("python"),
      ext_modules=ext_modules, cmdclass=cmdclass)
