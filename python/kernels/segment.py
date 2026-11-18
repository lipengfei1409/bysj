"""Fused destination-segmented HIP operators for the DPA4 edge stream.

The profiled 3000-atom DPA4 step spends 64% of its device time in the autograd
pass that turns the edge stream into per-atom forces, and most of that pass is
``fill_`` + ``index_add_`` + broadcast chains that ATen can only express one
kernel at a time. These three operators replace those chains with one kernel
each (design and measured budget: ``docs/task2-hip-fusion-plan.md``):

``segment_sum``
    Zero-init + segment scatter-add + per-node scale in one launch. One block
    owns one destination segment, so the accumulator starts in a register, no
    row atomic is issued and the normalization is folded into the store.

``segment_softmax``
    The gated softmax over each destination's incoming edges, with the cutoff
    envelope folded into the logit and a per-channel null mass. The forward
    keeps the maximum and the denominator in registers; the backward runs on
    the stored weights and needs no second read of the logits.

``force_assembly``
    ``F_k = sum_{dst=k} g - sum_{src=k} g`` and the per-atom virial outer
    product as two CSR segment walks, instead of two atomic scatters plus a
    materialized ``(E, 9)`` surface.

Grouping contract: every operator expects the DeepMD edge layout -- the
natural edge order grouped by destination (``dst`` non-decreasing) with
``offsets`` its exclusive prefix sum. Build it once per neighbour graph with
:func:`segment_offsets` and reuse it for all blocks and for the backward.
"""
from __future__ import annotations

import os

import torch

from .native import _extension, require_dcu

_CHECK_GROUPING = os.environ.get("DPA4_CHECK_GROUPING", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def segment_offsets(dst: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Exclusive prefix sum of the destination group sizes of a CSR edge list."""
    if not isinstance(dst, torch.Tensor):
        raise TypeError("segment offsets require a tensor of destinations")
    if dst.dtype != torch.int64:
        raise TypeError("segment offsets require int64 destinations")
    if n_nodes <= 0:
        raise ValueError("segment offsets require a positive node count")
    if dst.numel() and (dst.min() < 0 or dst.max() >= n_nodes):
        raise ValueError("destination values must be within the node range")
    counts = torch.bincount(dst, minlength=n_nodes)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)])


def group_by_source(src: torch.Tensor, n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable ``(order, offsets)`` pair that groups a edge list by source atom."""
    if not isinstance(src, torch.Tensor):
        raise TypeError("group_by_source requires a tensor of sources")
    if src.dtype != torch.int64:
        raise TypeError("group_by_source requires int64 sources")
    if n_nodes <= 0:
        raise ValueError("group_by_source requires a positive node count")
    if src.numel() and (src.min() < 0 or src.max() >= n_nodes):
        raise ValueError("source values must be within the node range")
    order = torch.argsort(src, stable=True)
    return order, segment_offsets(src, n_nodes)


def _validate_segment_sum(src, dst, offsets, edge_scale, node_scale):
    for name, tensor in (("src", src), ("dst", dst), ("offsets", offsets)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"segment sum {name} must be a tensor")
    if src.dtype != torch.float32:
        raise TypeError("segment sum src must be float32")
    if dst.dtype != torch.int64 or offsets.dtype != torch.int64:
        raise TypeError("segment sum dst and offsets must be int64")
    if src.ndim != 2:
        raise ValueError("segment sum src must have shape (n_edges, channels)")
    if dst.shape != (src.shape[0],):
        raise ValueError("segment sum dst must have shape (n_edges,)")
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("segment sum offsets must be a one-dimensional prefix sum")
    n_nodes = int(offsets.numel()) - 1
    if edge_scale is not None:
        if not isinstance(edge_scale, torch.Tensor) or edge_scale.dtype != torch.float32:
            raise TypeError("segment sum edge_scale must be a float32 tensor")
        if edge_scale.shape != (src.shape[0],):
            raise ValueError("segment sum edge_scale must have shape (n_edges,)")
    if node_scale is not None:
        if not isinstance(node_scale, torch.Tensor) or node_scale.dtype != torch.float32:
            raise TypeError("segment sum node_scale must be a float32 tensor")
        if node_scale.numel() != n_nodes:
            raise ValueError("segment sum node_scale must have one value per segment")
    tensors = [src, dst, offsets] + [t for t in (edge_scale, node_scale) if t is not None]
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("segment sum inputs must be contiguous")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all segment sum inputs must share a device")
    return n_nodes


def _validate_segment_softmax(logit, env, offsets, null_logit):
    for name, tensor in (("logit", logit), ("env", env), ("offsets", offsets)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"segment softmax {name} must be a tensor")
    if logit.dtype != torch.float32 or env.dtype != torch.float32:
        raise TypeError("segment softmax logit and env must be float32")
    if offsets.dtype != torch.int64:
        raise TypeError("segment softmax offsets must be int64")
    if logit.ndim != 2:
        raise ValueError("segment softmax logit must have shape (n_edges, channels)")
    if env.shape != (logit.shape[0],):
        raise ValueError("segment softmax env must have shape (n_edges,)")
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("segment softmax offsets must be a one-dimensional prefix sum")
    n_nodes, channels = int(offsets.numel()) - 1, logit.shape[1]
    if null_logit is not None:
        if not isinstance(null_logit, torch.Tensor) or null_logit.dtype != torch.float32:
            raise TypeError("segment softmax null_logit must be a float32 tensor")
        if null_logit.shape != (n_nodes, channels):
            raise ValueError("segment softmax null_logit must have shape (n_nodes, channels)")
    tensors = [logit, env, offsets] + [t for t in (null_logit,) if t is not None]
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("segment softmax inputs must be contiguous")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all segment softmax inputs must share a device")
    return n_nodes, channels


def _validate_force_assembly(grad_edge, edge_vec, dst_offsets, src_offsets, src_order):
    for name, tensor in (
        ("grad_edge", grad_edge),
        ("edge_vec", edge_vec),
        ("dst_offsets", dst_offsets),
        ("src_offsets", src_offsets),
        ("src_order", src_order),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"force assembly {name} must be a tensor")
    if grad_edge.dtype != torch.float32 or edge_vec.dtype != torch.float32:
        raise TypeError("force assembly grad_edge and edge_vec must be float32")
    if not grad_edge.dim() == 2 or grad_edge.shape[1] != 3:
        raise ValueError("force assembly grad_edge must have shape (n_edges, 3)")
    if edge_vec.shape != grad_edge.shape:
        raise ValueError("force assembly edge_vec must match grad_edge")
    for name, tensor in (("dst_offsets", dst_offsets), ("src_offsets", src_offsets)):
        if tensor.dtype != torch.int64 or tensor.ndim != 1 or tensor.numel() < 2:
            raise ValueError(f"force assembly {name} must be an int64 prefix sum")
    if dst_offsets.numel() != src_offsets.numel():
        raise ValueError("force assembly offsets must describe the same atom count")
    if src_order.dtype != torch.int64 or src_order.shape != (grad_edge.shape[0],):
        raise ValueError("force assembly src_order must be int64 with shape (n_edges,)")
    tensors = [grad_edge, edge_vec, dst_offsets, src_offsets, src_order]
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("force assembly inputs must be contiguous")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all force assembly inputs must share a device")
    return int(dst_offsets.numel()) - 1


def _should_check(explicit):
    """Grouping validation is opt-in: it costs a bincount over the edge list."""
    return _CHECK_GROUPING if explicit is None else bool(explicit)


def _check_csr(dst: torch.Tensor, offsets: torch.Tensor, n_nodes: int, name: str) -> None:
    expected = segment_offsets(dst, n_nodes)
    if not torch.equal(expected, offsets):
        raise ValueError(
            f"{name} requires the natural edge order grouped by destination; "
            "offsets must be the exclusive prefix sum of the dst group sizes"
        )


class _SegmentSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, dst, offsets, edge_scale, node_scale):
        extension = _extension("segment_sum_forward")
        output = extension.segment_sum_forward(src, offsets, edge_scale, node_scale)
        ctx.save_for_backward(src, dst, offsets, edge_scale, node_scale)
        ctx.needs_node_scale_grad = node_scale is not None and node_scale.requires_grad
        return output

    @staticmethod
    def backward(ctx, grad_output):
        src, dst, offsets, edge_scale, node_scale = ctx.saved_tensors
        extension = _extension("segment_sum_backward")
        grad_src, grad_node_scale = extension.segment_sum_backward(
            grad_output.contiguous(),
            src,
            dst,
            offsets,
            edge_scale,
            node_scale,
            ctx.needs_node_scale_grad,
        )
        if not ctx.needs_node_scale_grad:
            return grad_src, None, None, None, None
        return grad_src, None, None, None, grad_node_scale.reshape(node_scale.shape)


class _SegmentSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit, env, offsets):
        extension = _extension("segment_softmax_forward")
        alpha, _ = extension.segment_softmax_forward(logit, env, offsets, None)
        ctx.save_for_backward(alpha, env, offsets)
        return alpha

    @staticmethod
    def backward(ctx, grad_alpha):
        alpha, env, offsets = ctx.saved_tensors
        extension = _extension("segment_softmax_backward")
        grad_logit, _, grad_env = extension.segment_softmax_backward(
            alpha, grad_alpha.contiguous(), env, offsets, None, None
        )
        return grad_logit, grad_env, None


class _SegmentSoftmaxNullFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit, env, offsets, null_logit):
        extension = _extension("segment_softmax_forward")
        alpha, null_alpha = extension.segment_softmax_forward(
            logit, env, offsets, null_logit
        )
        ctx.save_for_backward(alpha, null_alpha, env, offsets)
        return alpha, null_alpha

    @staticmethod
    def backward(ctx, grad_alpha, grad_null_alpha):
        alpha, null_alpha, env, offsets = ctx.saved_tensors
        if grad_alpha is None:
            grad_alpha = torch.zeros_like(alpha)
        if grad_null_alpha is None:
            grad_null_alpha = torch.zeros_like(null_alpha)
        extension = _extension("segment_softmax_backward")
        grad_logit, grad_null_logit, grad_env = extension.segment_softmax_backward(
            alpha,
            grad_alpha.contiguous(),
            env,
            offsets,
            null_alpha,
            grad_null_alpha.contiguous(),
        )
        return grad_logit, grad_env, None, grad_null_logit


def segment_sum(src, dst, offsets, edge_scale=None, node_scale=None, *, check_grouping=None):
    """``out[n] = node_scale[n] * sum_{dst(e)=n} edge_scale[e] * src[e]``.

    ``src`` is contiguous float32 ``(n_edges, channels)``, ``dst`` int64
    ``(n_edges,)`` non-decreasing, ``offsets`` its exclusive prefix sum.
    Gradients are defined for ``src``, ``edge_scale`` (not yet implemented) and
    ``node_scale``, never for the indices.
    """
    n_nodes = _validate_segment_sum(src, dst, offsets, edge_scale, node_scale)
    if _should_check(check_grouping):
        _check_csr(dst, offsets, n_nodes, "segment sum")
    require_dcu(src, dst, offsets, edge_scale, node_scale)
    return _SegmentSumFunction.apply(src, dst, offsets, edge_scale, node_scale)


def segment_softmax(logit, env, offsets, null_logit=None, *, check_grouping=None):
    """Gated destination-segmented softmax over the incoming edges.

    Returns ``alpha`` with shape ``(n_edges, channels)``; when ``null_logit``
    is given also returns the per-channel null mass ``(n_nodes, channels)``.
    Edges with a non-positive envelope carry no mass.
    """
    _validate_segment_softmax(logit, env, offsets, null_logit)
    require_dcu(logit, env, offsets, null_logit)
    if null_logit is None:
        return _SegmentSoftmaxFunction.apply(logit, env, offsets)
    return _SegmentSoftmaxNullFunction.apply(logit, env, offsets, null_logit)


def force_assembly(grad_edge, edge_vec, src, dst, dst_offsets, src_offsets=None, src_order=None):
    """Assemble ``(force, virial)`` from per-edge energy gradients.

    ``src_order`` and ``src_offsets`` group the edge list by source atom; when
    omitted they are derived from ``src`` (one stable sort -- cache them across
    invocations when this runs per step). Inference-only: the caller keeps the
    reference path whenever the force graph must stay differentiable.
    """
    n_nodes = int(dst_offsets.numel()) - 1
    if src_order is None or src_offsets is None:
        order, offsets = group_by_source(src, n_nodes)
        src_order = order if src_order is None else src_order
        src_offsets = offsets if src_offsets is None else src_offsets
    _validate_force_assembly(grad_edge, edge_vec, dst_offsets, src_offsets, src_order)
    require_dcu(grad_edge, edge_vec, dst_offsets, src_offsets, src_order)
    extension = _extension("force_assembly")
    return extension.force_assembly(
        grad_edge, edge_vec, dst_offsets, src_offsets, src_order
    )


def segment_available() -> bool:
    """Return whether this process can load all three fused segmented entries."""
    required = (
        "segment_sum_forward",
        "segment_sum_backward",
        "segment_softmax_forward",
        "segment_softmax_backward",
        "force_assembly",
    )
    try:
        require_dcu()
        extension = _extension("segment_sum_forward")
    except (ImportError, OSError, RuntimeError):
        return False
    return all(hasattr(extension, symbol) for symbol in required)


__all__ = [
    "force_assembly",
    "group_by_source",
    "segment_available",
    "segment_offsets",
    "segment_softmax",
    "segment_sum",
]
