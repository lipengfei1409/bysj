"""Standalone TileLang attention-weighted destination aggregation for DPA4."""

from functools import lru_cache

import torch

_BLOCK_EDGES = 4
_BLOCK_FEATURES = 32


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang attention aggregation requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def attention_aggregate_forward(
        n_edges,
        n_nodes,
        n_degree,
        n_focus,
        n_heads,
        head_dim,
        block_edges,
        block_features,
        dtype=T.float32,
    ):
        n_channels = n_focus * n_heads * head_dim
        n_features = n_degree * n_channels

        @T.prim_func
        def main(
            values: T.Tensor((n_edges, n_degree, n_focus, n_heads, head_dim), dtype),
            alpha: T.Tensor((n_edges, n_focus, n_heads), dtype),
            head_gate: T.Tensor((n_nodes, n_focus, n_heads), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            output: T.Tensor((n_nodes, n_degree, n_focus, n_heads, head_dim), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                T.ceildiv(n_features, block_features),
                threads=128,
            ) as (pid_edge, pid_feature):
                for edge_offset, feature_offset in T.Parallel(
                    block_edges, block_features
                ):
                    edge = pid_edge * block_edges + edge_offset
                    feature = pid_feature * block_features + feature_offset
                    if edge < n_edges and feature < n_features:
                        degree = feature // n_channels
                        channel = feature - degree * n_channels
                        focus = channel // (n_heads * head_dim)
                        focus_channel = channel - focus * (n_heads * head_dim)
                        head = focus_channel // head_dim
                        head_channel = focus_channel - head * head_dim
                        node = dst[edge]
                        value = (
                            values[edge, degree, focus, head, head_channel]
                            * alpha[edge, focus, head]
                            * head_gate[node, focus, head]
                        )
                        T.atomic_add(
                            output[node, degree, focus, head, head_channel], value
                        )

        return main

    return attention_aggregate_forward


@lru_cache(maxsize=1)
def _backward_values_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def attention_aggregate_backward_values(
        n_edges,
        n_nodes,
        n_degree,
        n_focus,
        n_heads,
        head_dim,
        block_edges,
        block_features,
        dtype=T.float32,
    ):
        n_channels = n_focus * n_heads * head_dim
        n_features = n_degree * n_channels

        @T.prim_func
        def main(
            grad_output: T.Tensor(
                (n_nodes, n_degree, n_focus, n_heads, head_dim), dtype
            ),
            alpha: T.Tensor((n_edges, n_focus, n_heads), dtype),
            head_gate: T.Tensor((n_nodes, n_focus, n_heads), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            grad_values: T.Tensor(
                (n_edges, n_degree, n_focus, n_heads, head_dim), dtype
            ),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                T.ceildiv(n_features, block_features),
                threads=128,
            ) as (pid_edge, pid_feature):
                for edge_offset, feature_offset in T.Parallel(
                    block_edges, block_features
                ):
                    edge = pid_edge * block_edges + edge_offset
                    feature = pid_feature * block_features + feature_offset
                    if edge < n_edges and feature < n_features:
                        degree = feature // n_channels
                        channel = feature - degree * n_channels
                        focus = channel // (n_heads * head_dim)
                        focus_channel = channel - focus * (n_heads * head_dim)
                        head = focus_channel // head_dim
                        head_channel = focus_channel - head * head_dim
                        node = dst[edge]
                        grad_values[edge, degree, focus, head, head_channel] = (
                            grad_output[node, degree, focus, head, head_channel]
                            * alpha[edge, focus, head]
                            * head_gate[node, focus, head]
                        )

        return main

    return attention_aggregate_backward_values


@lru_cache(maxsize=1)
def _backward_alpha_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def attention_aggregate_backward_alpha(
        n_edges,
        n_nodes,
        n_degree,
        n_focus,
        n_heads,
        head_dim,
        block_edges,
        dtype=T.float32,
    ):
        @T.prim_func
        def main(
            grad_output: T.Tensor(
                (n_nodes, n_degree, n_focus, n_heads, head_dim), dtype
            ),
            values: T.Tensor((n_edges, n_degree, n_focus, n_heads, head_dim), dtype),
            head_gate: T.Tensor((n_nodes, n_focus, n_heads), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            grad_alpha: T.Tensor((n_edges, n_focus, n_heads), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges), n_focus, n_heads, threads=128
            ) as (pid, focus, head):
                for edge_offset in T.Parallel(block_edges):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges:
                        node = dst[edge]
                        total = 0.0
                        for degree in T.serial(n_degree):
                            for channel in T.serial(head_dim):
                                total += (
                                    grad_output[node, degree, focus, head, channel]
                                    * values[edge, degree, focus, head, channel]
                                    * head_gate[node, focus, head]
                                )
                        grad_alpha[edge, focus, head] = total

        return main

    return attention_aggregate_backward_alpha


@lru_cache(maxsize=1)
def _backward_gate_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def attention_aggregate_backward_gate(
        n_edges,
        n_nodes,
        n_degree,
        n_focus,
        n_heads,
        head_dim,
        block_edges,
        block_features,
        dtype=T.float32,
    ):
        n_channels = n_focus * n_heads * head_dim
        n_features = n_degree * n_channels

        @T.prim_func
        def main(
            grad_output: T.Tensor(
                (n_nodes, n_degree, n_focus, n_heads, head_dim), dtype
            ),
            values: T.Tensor((n_edges, n_degree, n_focus, n_heads, head_dim), dtype),
            alpha: T.Tensor((n_edges, n_focus, n_heads), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            grad_gate: T.Tensor((n_nodes, n_focus, n_heads), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                T.ceildiv(n_features, block_features),
                threads=128,
            ) as (pid_edge, pid_feature):
                for edge_offset, feature_offset in T.Parallel(
                    block_edges, block_features
                ):
                    edge = pid_edge * block_edges + edge_offset
                    feature = pid_feature * block_features + feature_offset
                    if edge < n_edges and feature < n_features:
                        degree = feature // n_channels
                        channel = feature - degree * n_channels
                        focus = channel // (n_heads * head_dim)
                        focus_channel = channel - focus * (n_heads * head_dim)
                        head = focus_channel // head_dim
                        head_channel = focus_channel - head * head_dim
                        node = dst[edge]
                        value = (
                            grad_output[node, degree, focus, head, head_channel]
                            * values[edge, degree, focus, head, head_channel]
                            * alpha[edge, focus, head]
                        )
                        T.atomic_add(grad_gate[node, focus, head], value)

        return main

    return attention_aggregate_backward_gate


def _check_inputs(values, alpha, head_gate, dst):
    if values.ndim != 5:
        raise ValueError(
            "values must have shape (n_edges, n_degree, n_focus, n_heads, head_dim)"
        )
    n_edges, _, n_focus, n_heads, _ = values.shape
    if alpha.shape != (n_edges, n_focus, n_heads):
        raise ValueError("alpha must have shape (n_edges, n_focus, n_heads)")
    if head_gate.ndim != 3 or head_gate.shape[1:] != (n_focus, n_heads):
        raise ValueError("head_gate must have shape (n_nodes, n_focus, n_heads)")
    if dst.shape != (n_edges,) or dst.dtype != torch.int64:
        raise TypeError("dst must be an int64 tensor with shape (n_edges,)")
    if any(t.dtype != torch.float32 for t in (values, alpha, head_gate)):
        raise TypeError("TileLang attention aggregation requires float32 values")
    if len({values.device, alpha.device, head_gate.device, dst.device}) != 1:
        raise ValueError("all attention aggregation inputs must share a device")
    if not values.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang attention aggregation requires a HIP/DCU tensor")


def _forward(values, alpha, head_gate, dst):
    _check_inputs(values, alpha, head_gate, dst)
    n_edges, n_degree, n_focus, n_heads, head_dim = values.shape
    n_nodes = head_gate.shape[0]
    output = torch.zeros(
        (n_nodes, n_degree, n_focus, n_heads, head_dim),
        dtype=values.dtype,
        device=values.device,
    )
    if n_edges:
        _, T = _tilelang_modules()
        _forward_factory()(
            n_edges,
            n_nodes,
            n_degree,
            n_focus,
            n_heads,
            head_dim,
            _BLOCK_EDGES,
            _BLOCK_FEATURES,
            T.float32,
        )(
            values.contiguous(),
            alpha.contiguous(),
            head_gate.contiguous(),
            dst.contiguous(),
            output,
        )
    return output


def _backward(grad_output, values, alpha, head_gate, dst):
    n_edges, n_degree, n_focus, n_heads, head_dim = values.shape
    n_nodes = head_gate.shape[0]
    grad_values = torch.empty_like(values)
    grad_alpha = torch.empty_like(alpha)
    grad_gate = torch.zeros_like(head_gate)
    if n_edges:
        _, T = _tilelang_modules()
        _backward_values_factory()(
            n_edges,
            n_nodes,
            n_degree,
            n_focus,
            n_heads,
            head_dim,
            _BLOCK_EDGES,
            _BLOCK_FEATURES,
            T.float32,
        )(
            grad_output.contiguous(),
            alpha.contiguous(),
            head_gate.contiguous(),
            dst.contiguous(),
            grad_values,
        )
        _backward_alpha_factory()(
            n_edges,
            n_nodes,
            n_degree,
            n_focus,
            n_heads,
            head_dim,
            _BLOCK_EDGES,
            T.float32,
        )(
            grad_output.contiguous(),
            values.contiguous(),
            head_gate.contiguous(),
            dst.contiguous(),
            grad_alpha,
        )
        _backward_gate_factory()(
            n_edges,
            n_nodes,
            n_degree,
            n_focus,
            n_heads,
            head_dim,
            _BLOCK_EDGES,
            _BLOCK_FEATURES,
            T.float32,
        )(
            grad_output.contiguous(),
            values.contiguous(),
            alpha.contiguous(),
            dst.contiguous(),
            grad_gate,
        )
    return grad_values, grad_alpha, grad_gate


class TileLangAttentionAggregate(torch.autograd.Function):
    """Autograd wrapper for attention-weighted node aggregation."""

    @staticmethod
    def forward(ctx, values, alpha, head_gate, dst):
        _check_inputs(values, alpha, head_gate, dst)
        values, alpha, head_gate, dst = (
            values.contiguous(),
            alpha.contiguous(),
            head_gate.contiguous(),
            dst.contiguous(),
        )
        ctx.save_for_backward(values, alpha, head_gate, dst)
        return _forward(values, alpha, head_gate, dst)

    @staticmethod
    def backward(ctx, grad_output):
        values, alpha, head_gate, dst = ctx.saved_tensors
        grad_values, grad_alpha, grad_gate = _backward(
            grad_output, values, alpha, head_gate, dst
        )
        return (
            grad_values if ctx.needs_input_grad[0] else None,
            grad_alpha if ctx.needs_input_grad[1] else None,
            grad_gate if ctx.needs_input_grad[2] else None,
            None,
        )


def tilelang_attention_aggregate(values, alpha, head_gate, dst):
    """Aggregate weighted DPA4 messages by destination node."""
    return TileLangAttentionAggregate.apply(values, alpha, head_gate, dst)


def attention_aggregate_reference(values, alpha, head_gate, dst):
    """Differentiable PyTorch reference for attention-weighted aggregation."""
    n_nodes = head_gate.shape[0]
    messages = (
        values
        * alpha[:, None, :, :, None]
        * head_gate.index_select(0, dst)[:, None, :, :, None]
    )
    output = values.new_zeros((n_nodes, *values.shape[1:]))
    if values.shape[0]:
        output.index_add_(0, dst, messages)
    return output


__all__ = [
    "TileLangAttentionAggregate",
    "attention_aggregate_reference",
    "tilelang_attention_aggregate",
]
