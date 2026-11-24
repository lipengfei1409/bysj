"""Block-diagonal mixing GEMM (C7-b).

One launch for the whole diagonal block set: the weight is consumed as stored
(strided slices are not copied), the block width is the real one (no power-of-two
padding), and the output is written once instead of being concatenated.

The reverse direction supplies the activation gradient only -- inference freezes
the weights, which is the same boundary the other fused operators draw.
"""

from __future__ import annotations

import torch

from .native import _extension, require_dcu


def blockdiag_available() -> bool:
    """Return whether this process can load both block-diagonal entry points."""
    try:
        require_dcu()
        extension = _extension("blockdiag_gemm")
    except (ImportError, OSError, RuntimeError):
        return False
    return hasattr(extension, "blockdiag_gemm_backward")


def _validate(x: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor) -> None:
    if x.dim() != 2:
        raise ValueError("x must be (E, ROW)")
    if x.dtype != torch.float32:
        raise TypeError("x must be float32")
    if weights.dim() != 3 or weights.shape[1] != weights.shape[2]:
        raise ValueError("weights must be (blocks, size, size)")
    if offsets.dtype != torch.int64 or offsets.dim() != 1:
        raise TypeError("offsets must be an int64 (blocks + 1) vector")
    if offsets.numel() != weights.shape[0] + 1:
        raise ValueError("offsets must hold one more entry than there are blocks")
    if int(offsets[0]) != 0 or int(offsets[-1]) != x.shape[1]:
        raise ValueError("offsets must span the activation width")
    if not (x.is_contiguous() and weights.is_contiguous() and offsets.is_contiguous()):
        raise ValueError("inputs must be contiguous")


def blockdiag_gemm(x: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``out[e, off+o] = sum_k W_b[o,k] * x[e, off+k]``."""
    _validate(x, weights, offsets)
    require_dcu(x)
    return _extension("blockdiag_gemm").blockdiag_gemm(x, weights, offsets)


def blockdiag_gemm_backward(grad_out: torch.Tensor, weights: torch.Tensor,
                            offsets: torch.Tensor) -> torch.Tensor:
    """``gx[e, off+k] = sum_o W_b[o,k] * gout[e, off+o]``."""
    _validate(grad_out, weights, offsets)
    require_dcu(grad_out)
    return _extension("blockdiag_gemm_backward").blockdiag_gemm_backward(
        grad_out, weights, offsets
    )
