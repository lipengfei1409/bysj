"""Hand a schedule to the runtime (the reference's ``dispatch.py``).

`Megakernels/megakernels/dispatch.py` is the boundary between the Python schedule
and the C++ VM.  Ours is one call: `kernels._C.dmk_gemm`, the GEMM-only megakernel
in `csrc/dmk/gemm_launch.cpp`.

Keeping it in its own module means the pure-Python side (`instructions`,
`scheduler`, `python_vm`) never imports an extension, so it works on a host.
"""

from __future__ import annotations

import os

import torch

from kernels.native import _extension


#: Which engine `launch_gemm` uses unless told otherwise.  `DPA4_GEMM_FLAT=1` selects
#: the role-free kernel, which is the same schedule executed without the loader /
#: storer / controller waves -- the A/B switch for "does the role split pay".
FLAT_DEFAULT = os.environ.get("DPA4_GEMM_FLAT") == "1"


def available(flat: bool = FLAT_DEFAULT) -> bool:
    """Whether this process can reach the requested persistent GEMM kernel."""
    try:
        _extension("dmk_gemm_flat" if flat else "dmk_gemm")
    except (ImportError, OSError, RuntimeError):
        return False
    return True


def launch_gemm(instructions: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
    """Run one persistent-GEMM launch; ``c`` is written in place."""
    if not available():
        raise RuntimeError(
            "the persistent GEMM kernel is not built: csrc/dmk/gemm_launch.cpp is "
            "part of kernels._C, so `python setup.py build_ext --inplace` on a DCU "
            "is what provides dmk_gemm"
        )
    return _extension("dmk_gemm").dmk_gemm(instructions, a, b, c)


def dense_available() -> bool:
    """Whether the ordinary dense GEMM (`csrc/gemm.cpp`) is in this build."""
    try:
        _extension("gemm_dense")
    except (ImportError, OSError, RuntimeError):
        return False
    return True


def launch_dense(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run the ordinary (non-persistent) dense GEMM; returns a new tensor."""
    if not dense_available():
        raise RuntimeError(
            "the dense GEMM is not built: gemm_dense is part of kernels._C, so "
            "`python setup.py build_ext --inplace` on a DCU provides it"
        )
    return _extension("gemm_dense").gemm_dense(a, b)


def flat_available() -> bool:
    """Whether the role-free kernel is in this build."""
    return available(flat=True)


def emulate_available() -> bool:
    """Always true: the emulator has no build step."""
    return True


__all__ = ["available", "dense_available", "emulate_available",
           "flat_available", "launch_dense", "launch_gemm"]
