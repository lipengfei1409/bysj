"""Pure Python install by default; DPA4_BUILD_HIP=1 builds DUMMA GEMM."""
import os
from pathlib import Path

from setuptools import find_packages, setup

ext_modules = []
cmdclass = {}
build_hip = os.environ.get("DPA4_BUILD_HIP") == "1"
build_dumma = os.environ.get("DPA4_ENABLE_DUMMA") == "1"
if build_dumma and not build_hip:
    raise RuntimeError("DPA4_ENABLE_DUMMA=1 requires DPA4_BUILD_HIP=1")
if build_hip and not build_dumma:
    raise RuntimeError("the native GEMM requires DPA4_ENABLE_DUMMA=1")
if build_hip:
    # The native kernel uses HIP syntax in a .cpp translation unit. Follow the
    # DTK manual's hipcc test.cpp convention rather than dispatching by suffix.
    hip_cxx = (os.environ.get("DPA4_HIP_CXX") or
               os.environ.get("HIPCXX") or "hipcc")
    os.environ["CXX"] = hip_cxx
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if not torch.version.hip:
        raise RuntimeError("native build requires a ROCm/DTK PyTorch installation")
    ext_modules = [CUDAExtension(
        "kernels._C",
        sources=["csrc/bindings.cpp", "csrc/gemm/gemm.cpp"],
        define_macros=[("DPA4_ENABLE_DUMMA", "1")],
        include_dirs=[str(Path(__file__).resolve().parent / "include")],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
    )]
    cmdclass = {"build_ext": BuildExtension}

setup(name="dcu-dpa4", version="0.1.0", python_requires=">=3.9",
      package_dir={"": "python"}, packages=find_packages("python"),
      ext_modules=ext_modules, cmdclass=cmdclass)
