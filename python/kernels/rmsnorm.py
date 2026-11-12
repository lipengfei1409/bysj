"""Inference-oriented HIP RMSNorm with gradients for energy/force propagation.

This is a standalone primitive; the current DPA4Model adapter delegates its
model graph to external DeepMD and does not dispatch this operator.
"""
from __future__ import annotations

import math

import torch

from .native import _extension, require_dcu


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


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, eps: float):
        extension = _extension("rmsnorm_forward")
        output, inverse_rms = extension.rmsnorm_forward(x, scale, float(eps))
        ctx.save_for_backward(x, scale, inverse_rms)
        ctx.has_scale = scale.numel() != 0
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, scale, inverse_rms = ctx.saved_tensors
        need_scale_grad = ctx.has_scale and ctx.needs_input_grad[1]
        extension = _extension("rmsnorm_backward")
        grad_x, grad_scale = extension.rmsnorm_backward(
            grad_output.contiguous(), x, scale, inverse_rms, need_scale_grad
        )
        return (
            grad_x if ctx.needs_input_grad[0] else None,
            grad_scale if need_scale_grad else None,
            None,
        )


def rmsnorm(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Apply last-axis RMSNorm on a supported DCU, with a PyTorch input-gradient path.

    The contract is contiguous FP32 input, arbitrary leading dimensions, optional
    FP32 scale of shape ``(x.shape[-1],)``, and finite positive ``eps``. There is
    no bias. Input and scale gradients are supported for force/inference
    differentiation and scale tuning.
    """
    eps = _validate(x, scale, eps)
    scale_tensor = scale if scale is not None else x.new_empty((0,))
    require_dcu(x, scale_tensor)
    return _RMSNormFunction.apply(x, scale_tensor, eps)


def rmsnorm_available() -> bool:
    """Return whether this process can load the RMSNorm native entry points."""
    try:
        require_dcu()
        extension = _extension("rmsnorm_forward")
        return hasattr(extension, "rmsnorm_backward")
    except (ImportError, OSError, RuntimeError):
        return False


__all__ = ["rmsnorm", "rmsnorm_available"]
