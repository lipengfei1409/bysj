"""PyTorch references for the slot-axis operators.

``slot_pack_scale`` replaces the four-pass edge-tensor construction in the
profiled forward (``index_select`` to expand the three l-rows into nine, an
elementwise ``mul`` for the radial scale, a ``copy_`` into the ``(E, 10, C)``
buffer, and a ``fill_`` for the reserved row). ``slot_reduce`` replaces the
three ``sum`` calls over that ten-row axis.

Both references are written with plain indexing so the contract is readable: the
operator must produce exactly this, in one pass.
"""
from __future__ import annotations

import torch


def _validate_slot(slot: torch.Tensor, rows_out: int) -> None:
    if slot.dtype != torch.int64:
        raise TypeError("slot pack slot must be int64")
    if slot.dim() != 1 or slot.numel() != rows_out:
        raise ValueError("slot pack slot must have shape (rows_out,)")


def slot_pack_scale_reference(src, slot, scale=None):
    """``out[e, r] = slot[r] < 0 ? 0 : src[e, slot[r]] * scale[e, r]``."""
    if src.dtype != torch.float32:
        raise TypeError("slot pack src must be float32")
    if src.dim() != 3:
        raise ValueError("slot pack src must have shape (n_edges, rows_in, channels)")
    edges, rows_in, channels = src.shape
    rows_out = int(slot.numel()) if slot.dim() == 1 else -1
    _validate_slot(slot, rows_out)
    if rows_out <= 0:
        raise ValueError("slot pack requires at least one output row")
    if rows_out and int(slot.max()) >= rows_in:
        raise ValueError("slot pack slot entries must index src rows")
    if scale is not None:
        if scale.dtype != torch.float32:
            raise TypeError("slot pack scale must be float32")
        if scale.shape != (edges, rows_out, channels):
            raise ValueError("slot pack scale must have shape (n_edges, rows_out, channels)")

    out = src.new_zeros((edges, rows_out, channels))
    gather = slot.clamp_min(0)
    gathered = src.index_select(1, gather)
    out = torch.where((slot < 0).reshape(1, rows_out, 1), out, gathered)
    if scale is not None:
        out = out * scale
    return out


def slot_reduce_reference(src, weight=None):
    """``out[e, c] = sum_r weight[e, r] * src[e, r, c]``."""
    if src.dtype != torch.float32:
        raise TypeError("slot reduce src must be float32")
    if src.dim() != 3:
        raise ValueError("slot reduce src must have shape (n_edges, rows, channels)")
    edges, rows, channels = src.shape
    if rows <= 0 or channels <= 0:
        raise ValueError("slot reduce requires at least one row and one channel")
    if weight is not None:
        if weight.dtype != torch.float32:
            raise TypeError("slot reduce weight must be float32")
        if weight.shape != (edges, rows):
            raise ValueError("slot reduce weight must have shape (n_edges, rows)")
        return (src * weight.reshape(edges, rows, 1)).sum(dim=1)
    return src.sum(dim=1)


__all__ = ["slot_pack_scale_reference", "slot_reduce_reference"]
