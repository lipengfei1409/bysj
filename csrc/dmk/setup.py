"""Build the dmk subtree's own extension (persistent runtime + ordinary entries).

Deliberately separate from the top-level ``setup.py``: the megakernel carries a
fixed LDS budget, a page layout and a scheduling contract that the main operator
build has no business inheriting, and the ordinary-launch entry points that
mirror the persistent ops are only meaningful next to that runtime.  This module
is therefore shipped as ``kernels.dmk._C``.

    python csrc/dmk/setup.py build_ext --inplace     # from the repo root
"""

import os

from setuptools import setup

# Native kernels use HIP syntax in .cpp translation units, as the top-level
# build does; follow the DTK manual's hipcc test.cpp convention.
os.environ["CXX"] = os.environ.get("HIPCXX") or "hipcc"

import torch  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

if not torch.version.hip:
    raise RuntimeError("the dmk extension needs a ROCm/DTK PyTorch installation")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root

sources = [
    os.path.join("csrc", "dmk", "so2_conv_dmk.cpp"),
]

setup(
    name="dcu-dpa4-dmk",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "kernels.dmk._C",
            sources=sources,
            # `ops/...` and `dmk/...` are addressed from every translation unit.
            include_dirs=[os.path.join(ROOT, "csrc")],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
