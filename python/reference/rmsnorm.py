"""PyTorch oracle for the standalone last-dimension RMSNorm contract."""
from __future__ import annotations

import math

import torch


def _validate(x: torch.Tensor, scale: torch.Tensor | None, eps: float) -> float:
    if not isinstance(x, torch.Tensor):
        raise TypeError("RMSNorm input must be a tensor")
    if x.dtype != torch.float32:
        raise TypeError("RMSNorm requires float32 input")
    if x.ndim < 1:
        raise ValueError("RMSNorm input must have at least one dimension")
    if x.shape[-1] <= 0:
        raise ValueError("RMSNorm feature dimension must be positive")
    if not x.is_contiguous():
        raise ValueError("RMSNorm input must be contiguous")
    if scale is not None:
        if not isinstance(scale, torch.Tensor):
            raise TypeError("RMSNorm scale must be a tensor or None")
        if scale.dtype != torch.float32:
            raise TypeError("RMSNorm scale must be float32")
        if scale.shape != (x.shape[-1],):
            raise ValueError("RMSNorm scale must have shape (feature_dim,)")
        if scale.device != x.device:
            raise ValueError("RMSNorm input and scale must share a device")
        if not scale.is_contiguous():
            raise ValueError("RMSNorm scale must be contiguous")
    try:
        eps = float(eps)
    except (TypeError, ValueError) as error:
        raise ValueError("RMSNorm eps must be a finite positive number") from error
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("RMSNorm eps must be a finite positive number")
    return eps


def rmsnorm(x: torch.Tensor, scale: torch.Tensor | None = None, eps: float = 1.0e-6):
    """Normalize the final feature axis using RMS, optionally applying ``scale``.

    The operation is ``x * rsqrt(mean(x**2) + eps) * scale`` with no bias.
    Leading dimensions are treated as independent rows. This reference is
    differentiable and is used for CPU contracts and explicit parity tests.
    """
    eps = _validate(x, scale, eps)
    inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    output = x * inverse_rms
    if scale is not None:
        output = output * scale
    return output


__all__ = ["rmsnorm"]
