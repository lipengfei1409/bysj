"""Hand a schedule to the runtime (the reference's ``dispatch.py``).

`Megakernels/megakernels/dispatch.py` is the boundary between the Python schedule
and the C++ VM.  Ours is one call: `kernels._C.dmk_gemm`, the GEMM-only megakernel
in `csrc/dmk/gemm_launch.cpp`.

Keeping it in its own module means the pure-Python side (`instructions`,
`scheduler`, `python_vm`) never imports an extension, so it works on a host.
"""

from __future__ import annotations

import torch

from kernels.native import _extension


def available() -> bool:
    """Whether this process can reach the persistent GEMM kernel."""
    try:
        _extension("dmk_gemm")
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


def emulate_available() -> bool:
    """Always true: the emulator has no build step."""
    return True


__all__ = ["available", "emulate_available", "launch_gemm"]
