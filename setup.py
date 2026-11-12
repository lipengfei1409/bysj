"""Pure Python install by default; opt into individual HIP native operators."""
import os
from pathlib import Path

from setuptools import find_packages, setup

ext_modules = []
cmdclass = {}
build_hip = os.environ.get("DPA4_BUILD_HIP") == "1"
build_dumma = os.environ.get("DPA4_ENABLE_DUMMA") == "1"
build_rmsnorm = os.environ.get("DPA4_ENABLE_RMSNORM") == "1"
if build_dumma and not build_hip:
    raise RuntimeError("DPA4_ENABLE_DUMMA=1 requires DPA4_BUILD_HIP=1")
if build_rmsnorm and not build_hip:
    raise RuntimeError("DPA4_ENABLE_RMSNORM=1 requires DPA4_BUILD_HIP=1")
if build_hip and not (build_dumma or build_rmsnorm):
    raise RuntimeError(
        "the native GEMM requires DPA4_ENABLE_DUMMA=1 unless "
        "DPA4_ENABLE_RMSNORM=1 is selected"
    )
if build_hip:
    # Native kernels use HIP syntax in .cpp translation units. Follow the DTK
    # manual's hipcc test.cpp convention rather than dispatching by suffix.
    hip_cxx = (os.environ.get("DPA4_HIP_CXX") or
               os.environ.get("HIPCXX") or "hipcc")
    os.environ["CXX"] = hip_cxx
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if not torch.version.hip:
        raise RuntimeError("native build requires a ROCm/DTK PyTorch installation")

    sources = ["csrc/bindings.cpp"]
    define_macros = []
    if build_dumma:
        sources.append("csrc/gemm/gemm.cpp")
        define_macros.append(("DPA4_ENABLE_DUMMA", "1"))
    if build_rmsnorm:
        sources.append("csrc/rmsnorm/rmsnorm.cpp")
        define_macros.append(("DPA4_ENABLE_RMSNORM", "1"))

    ext_modules = [CUDAExtension(
        "kernels._C",
        sources=sources,
        define_macros=define_macros,
        include_dirs=[str(Path(__file__).resolve().parent / "include")],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
    )]
    cmdclass = {"build_ext": BuildExtension}

setup(name="dcu-dpa4", version="0.1.0", python_requires=">=3.9",
      package_dir={"": "python"}, packages=find_packages("python"),
      ext_modules=ext_modules, cmdclass=cmdclass)
