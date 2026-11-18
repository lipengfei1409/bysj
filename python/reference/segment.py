"""PyTorch references for the fused segmented operators.

Each reference is written with the ATen chain the profiler traced (zero-init +
``index_add_`` + scale, the per-segment max/sum chain of the gated softmax, the
two endpoint scatters of the force assembly). The HIP operators in
``kernels/segment.py`` must reproduce these results with fewer kernels, so the
reference doubles as the executable specification of the fusion.
"""
from __future__ import annotations

import torch


def _node_count(offsets: torch.Tensor) -> int:
    if offsets.dim() != 1 or offsets.numel() < 2:
        raise ValueError("offsets must be a one-dimensional prefix sum of at least two entries")
    return int(offsets.numel()) - 1


def _validate_csr(dst: torch.Tensor, offsets: torch.Tensor, n_nodes: int) -> None:
    counts = torch.bincount(dst, minlength=n_nodes) if dst.numel() else dst.new_zeros(n_nodes)
    expected = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    if not torch.equal(expected, offsets):
        raise ValueError("offsets must be the exclusive prefix sum of the dst group sizes")


def _validate_grouping(dst: torch.Tensor, offsets: torch.Tensor, n_edges: int, name: str) -> None:
    if dst.dtype != torch.int64:
        raise TypeError(f"{name} dst must be int64")
    if dst.ndim != 1 or dst.shape[0] != n_edges:
        raise ValueError(f"{name} dst must have shape (n_edges,)")
    if offsets.dtype != torch.int64:
        raise TypeError(f"{name} offsets must be int64")
    if dst.numel() and (dst.min() < 0 or dst.max() >= _node_count(offsets)):
        raise ValueError(f"{name} dst values must be within the node range")
    _validate_csr(dst, offsets, _node_count(offsets))


def segment_sum_reference(src, dst, offsets, edge_scale=None, node_scale=None):
    """``out[n] = node_scale[n] * sum_{dst(e)=n} edge_scale[e] * src[e]``."""
    if src.dtype != torch.float32:
        raise TypeError("segment sum src must be float32")
    if src.ndim != 2:
        raise ValueError("segment sum src must have shape (n_edges, channels)")
    n_nodes = _node_count(offsets)
    _validate_grouping(dst, offsets, src.shape[0], "segment sum")
    if edge_scale is not None:
        if edge_scale.dtype != torch.float32 or edge_scale.shape != (src.shape[0],):
            raise ValueError("segment sum edge_scale must be float32 with shape (n_edges,)")
    if node_scale is not None:
        if node_scale.dtype != torch.float32 or node_scale.numel() != n_nodes:
            raise ValueError("segment sum node_scale must be float32 with one value per segment")

    weighted = src if edge_scale is None else src * edge_scale.reshape(-1, 1)
    out = weighted.new_zeros((n_nodes, src.shape[1]))
    if weighted.numel():
        out.index_add_(0, dst, weighted)
    if node_scale is not None:
        out = out * node_scale.reshape(n_nodes, 1)
    return out


def segment_softmax_reference(logit, env, dst, offsets, null_logit=None):
    """Per-segment, per-channel softmax with the envelope folded into the logit.

    ``l_e = logit_e + 2 ln env_e`` for ``env_e > 0`` (other edges are excluded),
    ``l_0`` the per-channel null logit. Returns ``(alpha, null_alpha)`` with
    ``alpha_e = exp(l_e - m) / D``, ``alpha_0 = exp(l_0 - m) / D`` and
    ``D = exp(l_0 - m) + sum_e exp(l_e - m)``, ``m`` the maximum over the segment
    and the null logit.
    """
    if logit.dtype != torch.float32 or env.dtype != torch.float32:
        raise TypeError("segment softmax logit and env must be float32")
    if logit.ndim != 2 or env.shape != (logit.shape[0],):
        raise ValueError("segment softmax expects logit (n_edges, channels) and env (n_edges,)")
    n_edges, channels = logit.shape
    n_nodes = _node_count(offsets)
    _validate_grouping(dst, offsets, n_edges, "segment softmax")
    if null_logit is not None and (
        null_logit.dtype != torch.float32 or null_logit.shape != (n_nodes, channels)
    ):
        raise ValueError("segment softmax null_logit must be float32 (n_nodes, channels)")

    negative_infinity = torch.full_like(logit, float("-inf"))
    positive = (env > 0).reshape(-1, 1)
    effective = torch.where(positive, logit + 2.0 * torch.log(env.clamp_min(1e-30)).reshape(-1, 1), negative_infinity)

    segment_index = dst.reshape(-1, 1).expand(n_edges, channels).contiguous()
    if n_edges:
        peak = torch.full((n_nodes, channels), float("-inf"), dtype=logit.dtype, device=logit.device)
        peak.scatter_reduce_(0, segment_index, effective, reduce="amax", include_self=True)
    else:
        peak = torch.full((n_nodes, channels), float("-inf"), dtype=logit.dtype, device=logit.device)
    if null_logit is not None:
        peak = torch.maximum(peak, null_logit)

    finite = peak > float("-inf")
    if n_edges:
        peak_per_edge = peak.index_select(0, dst)
        finite_per_edge = finite.index_select(0, dst)
        weights = torch.where(
            finite_per_edge, torch.exp(effective - peak_per_edge), torch.zeros_like(effective)
        )
    else:
        weights = effective
    total = logit.new_zeros((n_nodes, channels))
    if n_edges:
        total.index_add_(0, dst, weights)
    null_weight = (
        torch.where(finite, torch.exp(null_logit - peak), torch.zeros_like(peak))
        if null_logit is not None
        else torch.zeros_like(total)
    )
    total = total + null_weight
    safe = torch.where(total > 0, total, torch.ones_like(total))
    alpha = weights / safe.index_select(0, dst) if n_edges else weights
    null_alpha = null_weight / safe if null_logit is not None else None
    return alpha, null_alpha


def force_assembly_reference(grad_edge, edge_vec, src, dst, dst_offsets, src_offsets):
    """``F_k = sum_{dst=k} g_e - sum_{src=k} g_e`` and ``W_k = sum_{src=k} -g_e (x) v_e``."""
    if grad_edge.dtype != torch.float32 or grad_edge.ndim != 2 or grad_edge.shape[1] != 3:
        raise ValueError("force assembly grad_edge must be float32 (n_edges, 3)")
    if edge_vec.shape != grad_edge.shape:
        raise ValueError("force assembly edge_vec must match grad_edge")
    n_edges = grad_edge.shape[0]
    n_nodes = _node_count(dst_offsets)
    _validate_grouping(dst, dst_offsets, n_edges, "force assembly destination")
    _validate_grouping(src, src_offsets, n_edges, "force assembly source")
    if _node_count(src_offsets) != n_nodes:
        raise ValueError("force assembly offsets must describe the same atom count")

    force = grad_edge.new_zeros((n_nodes, 3))
    if n_edges:
        force.index_add_(0, dst, grad_edge)
        force.index_add_(0, src, -grad_edge)
    virial = grad_edge.new_zeros((n_nodes, 9))
    if n_edges:
        outer = -(grad_edge.unsqueeze(2) * edge_vec.unsqueeze(1)).reshape(n_edges, 9)
        virial.index_add_(0, src, outer)
    return force, virial


__all__ = [
    "force_assembly_reference",
    "segment_softmax_reference",
    "segment_sum_reference",
]
