"""PyTorch reference for the standalone DPA4 GIE zonal scatter."""
from __future__ import annotations

import torch


def _validate(zonal, radial, dst, radial_slot, inv_sqrt_deg):
    if not isinstance(zonal, torch.Tensor) or not isinstance(radial, torch.Tensor):
        raise TypeError("GIE zonal and radial inputs must be tensors")
    if not isinstance(dst, torch.Tensor) or not isinstance(radial_slot, torch.Tensor):
        raise TypeError("GIE dst and radial_slot must be tensors")
    if not isinstance(inv_sqrt_deg, torch.Tensor):
        raise TypeError("GIE inv_sqrt_deg must be a tensor")
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
    tensors = (zonal, radial, dst, radial_slot, inv_sqrt_deg)
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("GIE inputs must be contiguous")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all GIE inputs must share a device")
    if n_rows and radial.shape[1] == 0:
        raise ValueError("GIE radial must contain slots when zonal rows are present")
    if n_edges and inv_sqrt_deg.shape[0] == 0:
        raise ValueError("GIE degree normalization must contain destination nodes")


def zonal_scatter_reference(zonal, radial, dst, radial_slot, inv_sqrt_deg):
    """Scatter normalized edge messages into ``[N, R+1, C]``.

    Row zero is reserved and stays zero. The mathematical contract is
    ``out[dst[e], r+1, c] += zonal[e,r] * radial[e,radial_slot[r],c] *
    inv_sqrt_deg[dst[e]]``. Destination and radial-slot indices must be in range.
    """
    _validate(zonal, radial, dst, radial_slot, inv_sqrt_deg)
    inverse = inv_sqrt_deg.reshape(-1)
    n_edges, n_rows = zonal.shape
    n_nodes = inverse.shape[0]
    channels = radial.shape[2]
    if n_edges and ((dst < 0).any() or (dst >= n_nodes).any()):
        raise ValueError("GIE dst values must be within the node range")
    if n_rows and ((radial_slot < 0).any() or (radial_slot >= radial.shape[1]).any()):
        raise ValueError("GIE radial_slot values must be within the radial range")

    output = radial.new_zeros((n_nodes, n_rows + 1, channels))
    if n_edges and n_rows and channels:
        indices = radial_slot.reshape(1, n_rows, 1).expand(n_edges, n_rows, channels)
        radial_values = radial.gather(1, indices)
        messages = zonal.unsqueeze(-1) * radial_values
        messages = messages * inverse.index_select(0, dst).reshape(n_edges, 1, 1)
        output[:, 1:, :].index_add_(0, dst, messages)
    return output


# Descriptive alias for callers that group the operator by model stage.
gie_zonal_scatter_reference = zonal_scatter_reference

__all__ = ["gie_zonal_scatter_reference", "zonal_scatter_reference"]
