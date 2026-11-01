"""Standalone TileLang destination-segmented envelope-gated attention softmax."""

from functools import lru_cache

import torch


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang segment softmax requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def segment_softmax_forward(n_nodes, n_edges, n_focus, n_heads, dtype=T.float32):
        @T.prim_func
        def main(
            logits: T.Tensor((n_edges, n_focus, n_heads), dtype),
            edge_env: T.Tensor((n_edges,), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            order: T.Tensor((n_edges,), T.int64),
            row_ptr: T.Tensor((n_nodes + 1,), T.int64),
            null_logit: T.Tensor((n_focus, n_heads), dtype),
            segment_max: T.Tensor((n_nodes, n_focus, n_heads), dtype),
            denominator: T.Tensor((n_nodes, n_focus, n_heads), dtype),
        ):
            with T.Kernel(n_nodes, n_focus, n_heads, threads=128) as (
                node,
                focus,
                head,
            ):
                maximum = null_logit[focus, head]
                for position in T.serial(row_ptr[node], row_ptr[node + 1]):
                    edge = order[position]
                    env = edge_env[edge]
                    if env > 0.0:
                        value = logits[edge, focus, head] + 2.0 * T.log(env)
                        maximum = T.if_then_else(value > maximum, value, maximum)
                total = T.exp(null_logit[focus, head] - maximum)
                for position in T.serial(row_ptr[node], row_ptr[node + 1]):
                    edge = order[position]
                    env = edge_env[edge]
                    if env > 0.0:
                        value = logits[edge, focus, head] + 2.0 * T.log(env)
                        total += T.exp(value - maximum)
                segment_max[node, focus, head] = maximum
                denominator[node, focus, head] = total

        return main

    return segment_softmax_forward


@lru_cache(maxsize=1)
def _edge_forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def segment_softmax_edges(
        n_edges, n_nodes, n_focus, n_heads, block_edges, dtype=T.float32
    ):
        @T.prim_func
        def main(
            logits: T.Tensor((n_edges, n_focus, n_heads), dtype),
            edge_env: T.Tensor((n_edges,), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            segment_max: T.Tensor((n_nodes, n_focus, n_heads), dtype),
            denominator: T.Tensor((n_nodes, n_focus, n_heads), dtype),
            alpha: T.Tensor((n_edges, n_focus, n_heads), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges), n_focus, n_heads, threads=128
            ) as (pid, focus, head):
                for edge_offset in T.Parallel(block_edges):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges:
                        env = edge_env[edge]
                        if env > 0.0:
                            node = dst[edge]
                            value = logits[edge, focus, head] + 2.0 * T.log(env)
                            alpha[edge, focus, head] = (
                                T.exp(value - segment_max[node, focus, head])
                                / denominator[node, focus, head]
                            )
                        else:
                            alpha[edge, focus, head] = 0.0

        return main

    return segment_softmax_edges


@lru_cache(maxsize=1)
def _backward_segment_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def segment_softmax_backward_segment(
        n_nodes, n_edges, n_focus, n_heads, dtype=T.float32
    ):
        @T.prim_func
        def main(
            alpha: T.Tensor((n_edges, n_focus, n_heads), dtype),
            grad_alpha: T.Tensor((n_edges, n_focus, n_heads), dtype),
            order: T.Tensor((n_edges,), T.int64),
            row_ptr: T.Tensor((n_nodes + 1,), T.int64),
            segment_dot: T.Tensor((n_nodes, n_focus, n_heads), dtype),
        ):
            with T.Kernel(n_nodes, n_focus, n_heads, threads=128) as (
                node,
                focus,
                head,
            ):
                total = 0.0
                for position in T.serial(row_ptr[node], row_ptr[node + 1]):
                    edge = order[position]
                    total += alpha[edge, focus, head] * grad_alpha[edge, focus, head]
                segment_dot[node, focus, head] = total

        return main

    return segment_softmax_backward_segment


@lru_cache(maxsize=1)
def _backward_edge_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def segment_softmax_backward_edges(
        n_edges, n_nodes, n_focus, n_heads, block_edges, dtype=T.float32
    ):
        @T.prim_func
        def main(
            alpha: T.Tensor((n_edges, n_focus, n_heads), dtype),
            grad_alpha: T.Tensor((n_edges, n_focus, n_heads), dtype),
            edge_env: T.Tensor((n_edges,), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            segment_dot: T.Tensor((n_nodes, n_focus, n_heads), dtype),
            grad_logits: T.Tensor((n_edges, n_focus, n_heads), dtype),
            grad_edge_env: T.Tensor((n_edges,), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges), n_focus, n_heads, threads=128
            ) as (pid, focus, head):
                for edge_offset in T.Parallel(block_edges):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges:
                        node = dst[edge]
                        value = alpha[edge, focus, head] * (
                            grad_alpha[edge, focus, head]
                            - segment_dot[node, focus, head]
                        )
                        grad_logits[edge, focus, head] = value
                        if focus == 0 and head == 0:
                            env = edge_env[edge]
                            grad_edge_env[edge] = T.if_then_else(
                                env > 0.0, 2.0 * value / env, 0.0
                            )

        return main

    return segment_softmax_backward_edges


def _check_inputs(logits, edge_env, dst, order, row_ptr, null_logit):
    if logits.ndim != 3:
        raise ValueError("logits must have shape (n_edges, n_focus, n_heads)")
    if edge_env.ndim == 2 and edge_env.shape[1] == 1:
        edge_env = edge_env.reshape(-1)
    if edge_env.ndim != 1 or edge_env.shape[0] != logits.shape[0]:
        raise ValueError("edge_env must have shape (n_edges,) or (n_edges, 1)")
    if dst.ndim != 1 or dst.shape[0] != logits.shape[0]:
        raise ValueError("dst must have shape (n_edges,)")
    if order.ndim != 1 or order.shape != dst.shape:
        raise ValueError("order must have shape (n_edges,)")
    if row_ptr.ndim != 1:
        raise ValueError("row_ptr must have shape (n_nodes + 1,)")
    if null_logit.shape != logits.shape[1:]:
        raise ValueError("null_logit must have shape (n_focus, n_heads)")
    if any(t.dtype != torch.float32 for t in (logits, edge_env, null_logit)):
        raise TypeError("TileLang segment softmax requires float32 value tensors")
    if any(t.dtype != torch.int64 for t in (dst, order, row_ptr)):
        raise TypeError("dst, order and row_ptr must use torch.int64")
    if (
        len(
            {
                logits.device,
                edge_env.device,
                dst.device,
                order.device,
                row_ptr.device,
                null_logit.device,
            }
        )
        != 1
    ):
        raise ValueError("all segment softmax inputs must share a device")
    if not logits.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang segment softmax requires a HIP/DCU tensor")
    return edge_env


def _forward(logits, edge_env, dst, order, row_ptr, null_logit):
    edge_env = _check_inputs(logits, edge_env, dst, order, row_ptr, null_logit)
    n_edges, n_focus, n_heads = logits.shape
    n_nodes = row_ptr.shape[0] - 1
    alpha = torch.empty_like(logits)
    if n_edges == 0:
        return alpha
    _, T = _tilelang_modules()
    segment_max = torch.empty(
        (n_nodes, n_focus, n_heads), dtype=logits.dtype, device=logits.device
    )
    denominator = torch.empty_like(segment_max)
    _forward_factory()(n_nodes, n_edges, n_focus, n_heads, T.float32)(
        logits.contiguous(),
        edge_env.contiguous(),
        dst.contiguous(),
        order.contiguous(),
        row_ptr.contiguous(),
        null_logit.contiguous(),
        segment_max,
        denominator,
    )
    _edge_forward_factory()(n_edges, n_nodes, n_focus, n_heads, 32, T.float32)(
        logits.contiguous(),
        edge_env.contiguous(),
        dst.contiguous(),
        segment_max,
        denominator,
        alpha,
    )
    return alpha


def _backward(grad_alpha, logits, edge_env, dst, order, row_ptr, null_logit, alpha):
    n_edges, n_focus, n_heads = logits.shape
    n_nodes = row_ptr.shape[0] - 1
    grad_logits = torch.empty_like(logits)
    grad_edge_env = torch.empty_like(edge_env)
    if n_edges == 0:
        return grad_logits, grad_edge_env
    _, T = _tilelang_modules()
    segment_dot = torch.empty(
        (n_nodes, n_focus, n_heads), dtype=logits.dtype, device=logits.device
    )
    _backward_segment_factory()(n_nodes, n_edges, n_focus, n_heads, T.float32)(
        alpha.contiguous(),
        grad_alpha.contiguous(),
        order.contiguous(),
        row_ptr.contiguous(),
        segment_dot,
    )
    _backward_edge_factory()(n_edges, n_nodes, n_focus, n_heads, 32, T.float32)(
        alpha.contiguous(),
        grad_alpha.contiguous(),
        edge_env.contiguous(),
        dst.contiguous(),
        segment_dot,
        grad_logits,
        grad_edge_env,
    )
    return grad_logits, grad_edge_env


class TileLangSegmentSoftmax(torch.autograd.Function):
    """Autograd wrapper for envelope-squared segment softmax with null mass."""

    @staticmethod
    def forward(ctx, logits, edge_env, dst, order, row_ptr, null_logit):
        edge_env_shape = tuple(edge_env.shape)
        edge_env = _check_inputs(logits, edge_env, dst, order, row_ptr, null_logit)
        logits = logits.contiguous()
        edge_env = edge_env.contiguous()
        dst = dst.contiguous()
        order = order.contiguous()
        row_ptr = row_ptr.contiguous()
        null_logit = null_logit.contiguous()
        alpha = _forward(logits, edge_env, dst, order, row_ptr, null_logit)
        ctx.edge_env_shape = edge_env_shape
        ctx.save_for_backward(logits, edge_env, dst, order, row_ptr, null_logit, alpha)
        return alpha

    @staticmethod
    def backward(ctx, grad_alpha):
        logits, edge_env, dst, order, row_ptr, null_logit, alpha = ctx.saved_tensors
        grad_logits, grad_edge_env = _backward(
            grad_alpha, logits, edge_env, dst, order, row_ptr, null_logit, alpha
        )
        return (
            grad_logits if ctx.needs_input_grad[0] else None,
            grad_edge_env.reshape(ctx.edge_env_shape)
            if ctx.needs_input_grad[1]
            else None,
            None,
            None,
            None,
            None,
        )


def tilelang_segment_envelope_softmax(
    logits, edge_env, dst, order, row_ptr, null_logit
):
    """Normalize edge attention over destination segments using TileLang."""
    return TileLangSegmentSoftmax.apply(
        logits, edge_env, dst, order, row_ptr, null_logit
    )


def segment_envelope_softmax_reference(logits, edge_env, dst, n_nodes, null_logit):
    """Simple differentiable PyTorch reference for the selected no-mask path."""
    edge_env = edge_env.reshape(-1)
    _, n_focus, n_heads = logits.shape
    output = torch.zeros_like(logits)
    for node in range(n_nodes):
        mask = dst == node
        if not bool(mask.any()):
            continue
        env = edge_env[mask]
        active = env > 0.0
        for focus in range(n_focus):
            for head in range(n_heads):
                null = null_logit[focus, head]
                if bool(active.any()):
                    effective = logits[mask, focus, head][active] + 2.0 * torch.log(
                        env[active]
                    )
                    maximum = torch.maximum(null, effective.max())
                    weights = torch.exp(effective - maximum)
                    denom = weights.sum() + torch.exp(null - maximum)
                    edge_alpha = torch.zeros_like(env)
                    edge_alpha[active] = weights / denom
                else:
                    edge_alpha = torch.zeros_like(env)
                output[mask, focus, head] = edge_alpha
    return output


__all__ = [
    "TileLangSegmentSoftmax",
    "segment_envelope_softmax_reference",
    "tilelang_segment_envelope_softmax",
]
