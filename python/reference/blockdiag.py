"""Explicit oracles for the C7-b block-diagonal mixing GEMM.

The oracle is the dense reference form: slice each diagonal block out of the
strided weight, ``torch.bmm`` it, and concatenate -- the three costs the fused
operator removes.
"""

from __future__ import annotations

import torch


def blockdiag_gemm_reference(x: torch.Tensor, weights: torch.Tensor,
                             offsets: torch.Tensor) -> torch.Tensor:
    """``out[e, off+o] = sum_k W_b[o, k] * x[e, off+k]``, block by block."""
    out = x.to(torch.float64).new_zeros(x.shape)
    for b in range(weights.shape[0]):
        start = int(offsets[b])
        size = weights.shape[1]
        block = weights[b].to(torch.float64)
        out[:, start:start + size] = x[:, start:start + size].to(torch.float64) @ block.t()
    return out.to(x.dtype)


def blockdiag_gemm_backward_reference(grad_out: torch.Tensor, weights: torch.Tensor,
                                      offsets: torch.Tensor) -> torch.Tensor:
    """``gx[e, off+k] = sum_o W_b[o, k] * gout[e, off+o]``."""
    grad_x = grad_out.to(torch.float64).new_zeros(grad_out.shape)
    for b in range(weights.shape[0]):
        start = int(offsets[b])
        size = weights.shape[1]
        block = weights[b].to(torch.float64)
        grad_x[:, start:start + size] = grad_out[:, start:start + size].to(torch.float64) @ block
    return grad_x.to(grad_out.dtype)
