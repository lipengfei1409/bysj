"""Explicit PyTorch oracle for the DPA4 SO(2) envelope reduction."""
from __future__ import annotations

import torch


def so2_envelope_reduce_reference(message, edge_weight, dst, inv_sqrt_deg):
    """Scatter weighted edge messages to destination nodes and normalize.

    Args:
        message: ``(E, D, C)`` edge messages.
        edge_weight: ``(E,)`` per-edge envelope, optionally already multiplied
            by a source gate.
        dst: ``(E,)`` int64 destination indices.
        inv_sqrt_deg: ``(N,)`` or ``(N, 1, 1)`` smooth inverse-degree scale.

    Returns:
        ``(N, D, C)`` normalized destination updates.

    This is an explicit oracle only; production callers must select the native
    HIP wrapper or DeepMD's original implementation.
    """
    if not isinstance(message, torch.Tensor) or message.ndim != 3:
        raise ValueError("message must have shape (n_edges, n_coeff, n_channels)")
    if not isinstance(edge_weight, torch.Tensor) or edge_weight.ndim != 1:
        raise ValueError("edge_weight must have shape (n_edges,)")
    if not isinstance(dst, torch.Tensor) or dst.dtype != torch.int64 or dst.ndim != 1:
        raise ValueError("dst must be an int64 tensor with shape (n_edges,)")
    if message.shape[0] != edge_weight.numel() or dst.numel() != edge_weight.numel():
        raise ValueError("message, edge_weight, and dst edge dimensions must match")
    if inv_sqrt_deg.ndim not in (1, 3) or (
        inv_sqrt_deg.ndim == 3 and inv_sqrt_deg.shape[1:] != (1, 1)
    ):
        raise ValueError("inv_sqrt_deg must have shape (n_nodes,) or (n_nodes, 1, 1)")
    inverse = inv_sqrt_deg.reshape(-1)
    if dst.numel() and ((dst < 0).any() or (dst >= inverse.numel()).any()):
        raise ValueError("dst values must be within the node range")
    pre_norm = message.new_zeros((inverse.numel(), message.shape[1], message.shape[2]))
    if dst.numel():
        pre_norm.index_add_(0, dst, message * edge_weight.reshape(-1, 1, 1))
    return pre_norm * inverse.reshape(-1, 1, 1)


__all__ = ["so2_envelope_reduce_reference"]
