"""Reference force and virial assembly from edge-vector energy gradients."""

from __future__ import annotations

import torch


def edge_force_virial(
    edge_grad: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    n_atoms: int,
    edge_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return force ``[N,3]`` and DeePMD virial ``[3,3]``.

    Each edge contributes ``+g`` to its receiver and ``-g`` to its sender;
    the virial is ``-sum(g ⊗ edge_vec)`` with force index first.
    """
    if n_atoms < 0:
        raise ValueError("n_atoms must be nonnegative")
    if edge_grad.ndim != 2 or edge_grad.shape[1] != 3:
        raise ValueError("edge_grad must have shape [E, 3]")
    if edge_vec.shape != edge_grad.shape:
        raise ValueError("edge_vec must have shape [E, 3]")
    if edge_index.shape != (2, edge_grad.shape[0]):
        raise ValueError("edge_index must have shape [2, E]")
    if edge_grad.device != edge_vec.device or edge_grad.device != edge_index.device:
        raise ValueError("all edge tensors must share a device")
    if edge_mask is not None:
        if edge_mask.shape != (edge_grad.shape[0],):
            raise ValueError("edge_mask must have shape [E]")
        edge_grad = torch.where(edge_mask.unsqueeze(1), edge_grad, 0.0)

    sender, receiver = edge_index[0].long(), edge_index[1].long()
    force = edge_grad.new_zeros((n_atoms, 3))
    force.index_add_(0, receiver, edge_grad)
    force.index_add_(0, sender, -edge_grad)
    virial = -torch.einsum("ek,ej->kj", edge_grad, edge_vec)
    return force, virial
