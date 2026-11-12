"""Standalone native DPA4 GIE zonal-scatter operator.

The full DPA4 graph is owned by external DeepMD; this API does not itself wire
an operator into DeepPot. It exposes the source-backed zonal-scatter primitive.
"""
from __future__ import annotations

import torch

from .native import _extension, require_dcu


def _validate(zonal, radial, dst, radial_slot, inv_sqrt_deg):
    for name, tensor in (
        ("zonal", zonal),
        ("radial", radial),
        ("dst", dst),
        ("radial_slot", radial_slot),
        ("inv_sqrt_deg", inv_sqrt_deg),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"GIE {name} must be a tensor")
    if zonal.dtype != torch.float32 or radial.dtype != torch.float32 or inv_sqrt_deg.dtype != torch.float32:
        raise TypeError("GIE value tensors must be float32")
    if dst.dtype != torch.int64 or radial_slot.dtype != torch.int64:
        raise TypeError("GIE dst and radial_slot must be int64")
    if zonal.ndim != 2:
        raise ValueError("GIE zonal must have shape (n_edges, n_rows)")
    if radial.ndim != 3 or radial.shape[0] != zonal.shape[0]:
        raise ValueError("GIE radial must have shape (n_edges, n_radial, channels)")
    n_edges, n_rows = zonal.shape
    if dst.shape != (n_edges,):
        raise ValueError("GIE dst must have shape (n_edges,)")
    if radial_slot.shape != (n_rows,):
        raise ValueError("GIE radial_slot must have shape (n_rows,)")
    if inv_sqrt_deg.ndim not in (1, 3) or (
        inv_sqrt_deg.ndim == 3 and inv_sqrt_deg.shape[1:] != (1, 1)
    ):
        raise ValueError("GIE inv_sqrt_deg must have shape (n_nodes,) or (n_nodes, 1, 1)")
    if n_rows and radial.shape[1] == 0:
        raise ValueError("GIE radial must contain slots when zonal rows are present")
    if n_edges and inv_sqrt_deg.shape[0] == 0:
        raise ValueError("GIE degree normalization must contain destination nodes")
    if any(not tensor.is_contiguous() for tensor in (zonal, radial, dst, radial_slot, inv_sqrt_deg)):
        raise ValueError("GIE inputs must be contiguous")
    if len({tensor.device for tensor in (zonal, radial, dst, radial_slot, inv_sqrt_deg)}) != 1:
        raise ValueError("all GIE inputs must share a device")


class _GIEZonalScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zonal, radial, dst, radial_slot, inv_sqrt_deg):
        extension = _extension("gie_zonal_scatter_forward")
        output = extension.gie_zonal_scatter_forward(
            zonal, radial, dst, radial_slot, inv_sqrt_deg
        )
        ctx.save_for_backward(zonal, radial, dst, radial_slot, inv_sqrt_deg)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        zonal, radial, dst, radial_slot, inv_sqrt_deg = ctx.saved_tensors
        extension = _extension("gie_zonal_scatter_backward")
        grad_zonal, grad_radial, grad_inv_sqrt_deg = extension.gie_zonal_scatter_backward(
            grad_output.contiguous(), zonal, radial, dst, radial_slot, inv_sqrt_deg
        )
        return (
            grad_zonal if ctx.needs_input_grad[0] else None,
            grad_radial if ctx.needs_input_grad[1] else None,
            None,
            None,
            grad_inv_sqrt_deg if ctx.needs_input_grad[4] else None,
        )


def zonal_scatter(zonal, radial, dst, radial_slot, inv_sqrt_deg):
    """Apply DPA4's geometric initial-embedding zonal scatter on a DCU.

    All value tensors are contiguous float32 and all indices are contiguous
    int64. Indices are assumed valid and in range. The output has shape
    ``(n_nodes, n_rows + 1, channels)``; row zero is reserved. Gradients are
    defined for zonal, radial, and inverse-degree values, not the indices.
    """
    _validate(zonal, radial, dst, radial_slot, inv_sqrt_deg)
    require_dcu(zonal, radial, dst, radial_slot, inv_sqrt_deg)
    return _GIEZonalScatterFunction.apply(
        zonal, radial, dst, radial_slot, inv_sqrt_deg
    )


def gie_zonal_scatter(zonal, radial, dst, radial_slot, inv_sqrt_deg):
    """Alias emphasizing that this primitive belongs to the DPA4 GIE stage."""
    return zonal_scatter(zonal, radial, dst, radial_slot, inv_sqrt_deg)


def gie_available() -> bool:
    """Return whether this process can load both native GIE entry points."""
    try:
        require_dcu()
        extension = _extension("gie_zonal_scatter_forward")
        return hasattr(extension, "gie_zonal_scatter_backward")
    except (ImportError, OSError, RuntimeError):
        return False


__all__ = ["gie_available", "gie_zonal_scatter", "zonal_scatter"]
