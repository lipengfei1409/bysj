"""Native HIP destination reduction used by attention-free DPA4 SO(2) blocks."""
from __future__ import annotations

import torch

from .native import _extension, require_dcu


def _validate(message, edge_weight, dst, inv_sqrt_deg):
    tensors = {
        "message": message,
        "edge_weight": edge_weight,
        "dst": dst,
        "inv_sqrt_deg": inv_sqrt_deg,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"SO(2) {name} must be a tensor")
    if message.dtype != torch.float32:
        raise TypeError("SO(2) message must be float32")
    if edge_weight.dtype != torch.float32 or inv_sqrt_deg.dtype != torch.float32:
        raise TypeError("SO(2) edge weights and degree scales must be float32")
    if dst.dtype != torch.int64:
        raise TypeError("SO(2) dst must be int64")
    if message.ndim != 3 or message.shape[1] <= 0 or message.shape[2] <= 0:
        raise ValueError("SO(2) message must have shape (n_edges, n_coeff, n_channels)")
    n_edges = message.shape[0]
    if edge_weight.shape not in ((n_edges,), (n_edges, 1)):
        raise ValueError("SO(2) edge_weight must have shape (n_edges,) or (n_edges, 1)")
    if dst.shape != (n_edges,):
        raise ValueError("SO(2) dst must have shape (n_edges,)")
    if inv_sqrt_deg.ndim not in (1, 3) or (
        inv_sqrt_deg.ndim == 3 and inv_sqrt_deg.shape[1:] != (1, 1)
    ):
        raise ValueError(
            "SO(2) inv_sqrt_deg must have shape (n_nodes,) or (n_nodes, 1, 1)"
        )
    if any(not tensor.is_contiguous() for tensor in tensors.values()):
        raise ValueError("SO(2) inputs must be contiguous")
    if len({tensor.device for tensor in tensors.values()}) != 1:
        raise ValueError("all SO(2) inputs must share one device")
    if edge_weight.numel() != n_edges or inv_sqrt_deg.numel() == 0 and n_edges:
        raise ValueError("SO(2) edge and node dimensions are inconsistent")


class _SO2EnvelopeReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, message, edge_weight, dst, inverse):
        extension = _extension("so2_envelope_reduce_forward")
        output, pre_norm = extension.so2_envelope_reduce_forward(
            message, edge_weight, dst, inverse
        )
        ctx.save_for_backward(message, edge_weight, dst, inverse, pre_norm)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        message, edge_weight, dst, inverse, pre_norm = ctx.saved_tensors
        extension = _extension("so2_envelope_reduce_backward")
        grad_message, grad_weight, grad_inverse = (
            extension.so2_envelope_reduce_backward(
                grad_output.contiguous(), message, edge_weight, dst, inverse, pre_norm
            )
        )
        return (
            grad_message if ctx.needs_input_grad[0] else None,
            grad_weight if ctx.needs_input_grad[1] else None,
            None,
            grad_inverse if ctx.needs_input_grad[3] else None,
        )


def so2_envelope_reduce(message, edge_weight, dst, inv_sqrt_deg):
    """Compute ``inv_sqrt_deg[dst] * scatter_sum(message * edge_weight)``.

    The primitive supports contiguous float32 DCU tensors, arbitrary coefficient
    and channel dimensions, valid int64 destination indices, and first-order
    gradients with respect to message, edge weights, and degree scales. It is
    the destination-reduction seam of an attention-free SO2Convolution; edge
    geometry, SO(2) rotations, radial mixing, and the mixing stack remain in
    DeepMD. Training-mode calls and unsupported layouts should use the original
    DeepMD method.
    """
    _validate(message, edge_weight, dst, inv_sqrt_deg)
    require_dcu(message, edge_weight, dst, inv_sqrt_deg)
    return _SO2EnvelopeReduceFunction.apply(
        message,
        edge_weight.reshape(-1).contiguous(),
        dst,
        inv_sqrt_deg.reshape(-1).contiguous(),
    )


def so2_envelope_reduce_available() -> bool:
    """Return whether both native SO(2) reduction entry points are available."""
    try:
        require_dcu()
        extension = _extension("so2_envelope_reduce_forward")
        return hasattr(extension, "so2_envelope_reduce_backward")
    except (ImportError, OSError, RuntimeError):
        return False


__all__ = ["so2_envelope_reduce", "so2_envelope_reduce_available"]
