"""Build the DPA4 native HIP operators by default."""
import os

from setuptools import find_packages, setup

# Native kernels use HIP syntax in .cpp translation units. Follow the DTK
# manual's hipcc test.cpp convention rather than dispatching by suffix.
hip_cxx = os.environ.get("HIPCXX") or "hipcc"
os.environ["CXX"] = hip_cxx

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if not torch.version.hip:
    raise RuntimeError("native build requires a ROCm/DTK PyTorch installation")

sources = [
    "csrc/bindings.cpp",
    "csrc/rmsnorm/rmsnorm.cpp",
    "csrc/gie/gie.cpp",
]
ext_modules = [CUDAExtension(
    "kernels._C",
    sources=sources,
    define_macros=[],
    extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
)]
cmdclass = {"build_ext": BuildExtension}

setup(name="dcu-dpa4", version="0.1.0", python_requires=">=3.9",
      package_dir={"": "python"}, packages=find_packages("python"),
      ext_modules=ext_modules, cmdclass=cmdclass)
