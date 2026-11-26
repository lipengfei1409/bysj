"""SO(2) convolution, ordinary kernel entry point (C8).

The same per-edge span exists as a megakernel operator
(``csrc/ops/so2_conv_op.h``, run under the persistent VM in ``csrc/dmk/``); this
wrapper reaches the launch-a-kernel version in ``csrc/so2_conv_dmk.cpp``.  Both
call ``csrc/ops/so2_conv_math.h``, so the two paths share the arithmetic and the
accuracy test covers the semantics once for both.

    msg[e, d, c] = alpha[e] * (R^T[e] @ stack(R[e] @ x[src[e]][:, c]))

The destination reduction is *not* part of this operator: ``msg`` is per edge,
and summing it onto atoms is :func:`kernels.segment.segment_sum`'s job.
"""

from __future__ import annotations

import torch

from .native import _extension, require_dcu


def so2_conv_dmk_available() -> bool:
    """Return whether this process can load the ordinary SO(2) conv entry point."""
    try:
        require_dcu()
        extension = _extension("so2_conv_dmk")
    except (ImportError, OSError, RuntimeError):
        return False
    return hasattr(extension, "so2_conv_dmk")


def _validate(x, src, rot, rot_t, kc, alpha) -> None:
    if x.dim() != 3:
        raise ValueError("x must be (n_node, dim, channels)")
    if x.dtype != torch.float32:
        raise TypeError("x must be float32")
    if src.dtype != torch.int64 or src.dim() != 1:
        raise TypeError("src must be an int64 (E,) vector")
    edges = int(src.shape[0])
    for name, tensor in (("rot", rot), ("rot_t", rot_t), ("kc", kc), ("alpha", alpha)):
        if tensor.shape[0] != edges:
            raise ValueError(f"{name} must have one row per edge")
    if rot.dim() != 3 or rot_t.dim() != 3 or kc.dim() != 5:
        raise ValueError("expected rot (E, red, dim), rot_t (E, dim, red), kc (E, layers, 2, red, red)")
    if alpha.dim() != 1:
        raise ValueError("alpha must be (E,)")
    if int(rot.shape[2]) != int(rot_t.shape[1]) or int(rot.shape[1]) != int(rot_t.shape[2]):
        raise ValueError("rot_t must be the transpose of rot")
    if int(x.shape[2]) > 64:
        raise ValueError("one lane per channel requires channels <= 64")
    for name, tensor in (("x", x), ("rot", rot), ("rot_t", rot_t), ("kc", kc), ("alpha", alpha)):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be float32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def so2_conv_dmk(x: torch.Tensor, src: torch.Tensor, rot: torch.Tensor,
                 rot_t: torch.Tensor, kc: torch.Tensor,
                 alpha: torch.Tensor) -> torch.Tensor:
    """Per-edge SO(2) message, ``(E, dim, channels)``."""
    _validate(x, src, rot, rot_t, kc, alpha)
    require_dcu(x)
    return _extension("so2_conv_dmk").so2_conv_dmk(x, src, rot, rot_t, kc, alpha)
