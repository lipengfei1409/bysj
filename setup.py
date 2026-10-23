"""Pure Python install by default; DPA4_BUILD_HIP=1 builds native operators."""
import os
from pathlib import Path

from setuptools import find_packages, setup

ext_modules = []
cmdclass = {}
build_blas = os.environ.get("DPA4_BUILD_ROCBLAS") == "1"
if build_blas and os.environ.get("DPA4_BUILD_HIP") != "1":
    raise RuntimeError("DPA4_BUILD_ROCBLAS=1 requires DPA4_BUILD_HIP=1")
if os.environ.get("DPA4_BUILD_HIP") == "1":
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    if not torch.version.hip:
        raise RuntimeError("native build requires a ROCm/DTK PyTorch installation")
    sources = [
        "csrc/bindings.cpp", "csrc/parallel/geometry.cu", "csrc/parallel/radial.cu",
        "csrc/parallel/edge_features.cu", "csrc/megakernel/runtime.cu",
        "csrc/parallel/reduce.cu", "csrc/parallel/norm.cu", "csrc/gemm/linear.cu",
    ]
    if build_blas:
        sources.append("csrc/gemm/blas.cu")
    ext_modules = [CUDAExtension("dcu_dpa4._C", sources=sources,
       libraries=["rocblas"] if build_blas else [],
       define_macros=[("DPA4_ENABLE_ROCBLAS", "1")] if build_blas else [],
       include_dirs=[str(Path(__file__).resolve().parent / "include")],
       extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": ["-O3", "-std=c++17"]})]
    cmdclass = {"build_ext": BuildExtension}

setup(name="dcu-dpa4", version="0.1.0", python_requires=">=3.9",
      package_dir={"": "python"}, packages=find_packages("python"),
      extras_require={"qwen": ["transformers>=4.51.0"]},
      ext_modules=ext_modules, cmdclass=cmdclass)
