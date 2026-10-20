"""Edge geometry with the DeePMD src/dst convention."""

from __future__ import annotations

import torch


def edge_vectors(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    shifts: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``r_src + shift @ cell - r_dst`` with shape ``[E, 3]``.

    ``edge_index[0]`` is the sender and ``edge_index[1]`` is the receiver.
    This matches the SeZM edge-vector path used by the target checkpoint.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("edge_index must contain integer indices")
    if edge_index.device != positions.device:
        raise ValueError("edge_index and positions must be on the same device")

    nedge = edge_index.shape[1]
    if shifts is None:
        shifts = torch.zeros((nedge, 3), dtype=torch.int64, device=positions.device)
    if shifts.shape != (nedge, 3):
        raise ValueError("shifts must have shape [E, 3]")
    if shifts.device != positions.device:
        raise ValueError("shifts and positions must be on the same device")
    if cell is None:
        if bool(torch.any(shifts != 0)):
            raise ValueError("nonperiodic edges cannot have nonzero shifts")
        image_offset = 0.0
    else:
        if cell.shape != (3, 3) or cell.device != positions.device:
            raise ValueError("cell must have shape [3, 3] on the positions device")
        image_offset = shifts.to(positions.dtype) @ cell

    sender, receiver = edge_index[0].long(), edge_index[1].long()
    return positions.index_select(0, sender) + image_offset - positions.index_select(
        0, receiver
    )
