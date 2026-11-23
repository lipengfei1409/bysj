"""Native slot-axis operators for the DPA4 edge stream.

Two small operators, both one pass over an edge-major tensor:

``slot_pack_scale``
    ``out[e, r, c] = slot[r] < 0 ? 0 : src[e, slot[r], c] * scale[e, r, c]``
    One pass replaces the four kernels the profiled forward uses to build the
    SO(2) input: ``index_select`` (three l-rows expanded to nine), the
    elementwise ``mul`` for the radial scale, the ``copy_`` into the
    ``(E, 10, C)`` buffer and the ``fill_`` for the reserved row. A negative
    slot entry writes zeros, which is how the reserved row is produced without a
    separate initialisation.

``slot_reduce``
    ``out[e, c] = sum_r weight[e, r] * src[e, r, c]``
    The ten-row axis goes through the generic reduction today at an effective
    56 GB/s; one block per edge keeps the whole slot axis in registers.

Neither operator is bound into DeepMD yet -- see
``docs/optimization-status.md`` (C1 / C2) for status and expected gains.
"""
from __future__ import annotations

import torch

from .native import _extension, require_dcu


def _validate_slot(slot, rows_out, name="slot pack"):
    if not isinstance(slot, torch.Tensor) or slot.dtype != torch.int64:
        raise TypeError(f"{name} slot must be an int64 tensor")
    if slot.dim() != 1 or slot.numel() != rows_out:
        raise ValueError(f"{name} slot must have shape (rows_out,)")
    if not slot.is_contiguous():
        raise ValueError(f"{name} slot must be contiguous")


def _validate_common(tensors, name):
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError(f"all {name} inputs must share a device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(f"{name} inputs must be contiguous")


def _needs_grad(*tensors) -> bool:
    return torch.is_grad_enabled() and any(
        isinstance(tensor, torch.Tensor) and tensor.requires_grad
        for tensor in tensors
    )


class _SlotPackScaleFunction(torch.autograd.Function):
    """Autograd wrapper around the forward-only native kernel.

    ``out[e, r, c] = slot[r] < 0 ? 0 : src[e, slot[r], c] * scale[e, r, c]``
    so the gradient with respect to ``src`` is a scatter-add over the slot rows
    and the gradient with respect to ``scale`` is the gathered source row times
    the incoming gradient (zero for a reserved row).
    """

    @staticmethod
    def forward(ctx, src, slot, scale):
        out = _extension("slot_pack_scale").slot_pack_scale(src, slot, scale)
        ctx.save_for_backward(src, slot, scale)
        ctx.has_scale = scale is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None
        src, slot, scale = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        valid = slot >= 0

        grad_scale = None
        if ctx.has_scale and ctx.needs_input_grad[2]:
            gathered = src.index_select(1, slot.clamp_min(0))
            product = grad_out * gathered
            grad_scale = torch.where(valid.reshape(1, -1, 1), product,
                                     torch.zeros_like(product))

        grad_src = None
        if ctx.needs_input_grad[0]:
            grad_src = torch.zeros_like(src)
            index = slot[valid]
            if index.numel() != 0:
                contribution = grad_out[:, valid, :]
                if ctx.has_scale:
                    contribution = contribution * scale[:, valid, :]
                grad_src.index_add_(1, index, contribution)
        return grad_src, None, grad_scale


class _SlotReduceFunction(torch.autograd.Function):
    """Autograd wrapper for ``sum_r weight[e, r] * src[e, r, c]``."""

    @staticmethod
    def forward(ctx, src, weight):
        out = _extension("slot_reduce").slot_reduce(src, weight)
        ctx.save_for_backward(src, weight)
        ctx.has_weight = weight is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None
        src, weight = ctx.saved_tensors
        expanded = grad_out.contiguous().unsqueeze(1)
        grad_src = None
        if ctx.needs_input_grad[0]:
            grad_src = (expanded * weight.unsqueeze(-1)
                        if ctx.has_weight else expanded.expand_as(src))
        grad_weight = None
        if ctx.has_weight and ctx.needs_input_grad[1]:
            grad_weight = (expanded * src).sum(dim=-1)
        return grad_src, grad_weight


def slot_pack_scale(src, slot, scale=None):
    """Gather ``src`` rows through ``slot`` and apply an optional per-slot scale.

    ``src`` is ``(E, rows_in, C)`` fp32, ``slot`` int64 ``(rows_out,)`` with a
    negative entry meaning "zero row", ``scale`` optional ``(E, rows_out, C)``.
    """
    for name, tensor in (("src", src), ("slot", slot)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"slot pack {name} must be a tensor")
    if src.dtype != torch.float32:
        raise TypeError("slot pack src must be float32")
    if src.dim() != 3:
        raise ValueError("slot pack src must have shape (n_edges, rows_in, channels)")
    rows_out = int(slot.numel()) if slot.dim() == 1 else -1
    _validate_slot(slot, rows_out)
    if rows_out <= 0:
        raise ValueError("slot pack requires at least one output row")
    if int(slot.max()) >= src.shape[1]:
        raise ValueError("slot pack slot entries must index src rows")
    tensors = [src, slot]
    if scale is not None:
        if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32:
            raise TypeError("slot pack scale must be a float32 tensor")
        if scale.shape != (src.shape[0], rows_out, src.shape[2]):
            raise ValueError(
                "slot pack scale must have shape (n_edges, rows_out, channels)"
            )
        tensors.append(scale)
    _validate_common(tensors, "slot pack")
    require_dcu(*tensors)
    if _needs_grad(src, scale):
        return _SlotPackScaleFunction.apply(src, slot, scale)
    return _extension("slot_pack_scale").slot_pack_scale(src, slot, scale)


def slot_reduce(src, weight=None):
    """Reduce ``src`` over its slot axis, with an optional per-slot weight."""
    if not isinstance(src, torch.Tensor):
        raise TypeError("slot reduce src must be a tensor")
    if src.dtype != torch.float32:
        raise TypeError("slot reduce src must be float32")
    if src.dim() != 3:
        raise ValueError("slot reduce src must have shape (n_edges, rows, channels)")
    edges, rows, _ = src.shape
    tensors = [src]
    if weight is not None:
        if not isinstance(weight, torch.Tensor) or weight.dtype != torch.float32:
            raise TypeError("slot reduce weight must be a float32 tensor")
        if weight.shape != (edges, rows):
            raise ValueError("slot reduce weight must have shape (n_edges, rows)")
        tensors.append(weight)
    _validate_common(tensors, "slot reduce")
    require_dcu(*tensors)
    if _needs_grad(src, weight):
        return _SlotReduceFunction.apply(src, weight)
    return _extension("slot_reduce").slot_reduce(src, weight)


def slot_available() -> bool:
    """Return whether this process can load both slot entry points."""
    try:
        require_dcu()
        extension = _extension("slot_pack_scale")
    except (ImportError, OSError, RuntimeError):
        return False
    return hasattr(extension, "slot_reduce")


__all__ = ["slot_available", "slot_pack_scale", "slot_reduce"]
