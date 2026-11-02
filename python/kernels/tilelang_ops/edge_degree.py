"""Standalone TileLang DPA4 destination degree and normalization operator."""

from functools import lru_cache

import torch

_BLOCK_EDGES = 32


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang edge degree requires TileLang to be installed"
        ) from error
    return tilelang, T


def _build_csr(dst, n_nodes):
    order = torch.argsort(dst, stable=True)
    counts = torch.bincount(dst, minlength=n_nodes)
    row_ptr = torch.cat(
        (torch.zeros((1,), dtype=torch.int64, device=dst.device), counts.cumsum(0))
    )
    return order, row_ptr


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def edge_degree_forward(n_nodes, n_edges, deg_norm_floor, dtype=T.float32):
        @T.prim_func
        def main(
            edge_env: T.Tensor((n_edges, 1), dtype),
            order: T.Tensor((n_edges,), T.int64),
            row_ptr: T.Tensor((n_nodes + 1,), T.int64),
            deg: T.Tensor((n_nodes,), dtype),
            inv_sqrt_deg: T.Tensor((n_nodes, 1, 1), dtype),
        ):
            with T.Kernel(n_nodes, threads=128) as (node,):
                total = 0.0
                for position in T.serial(row_ptr[node], row_ptr[node + 1]):
                    edge = order[position]
                    env = edge_env[edge, 0]
                    total += env * env
                deg[node] = total
                inv_sqrt_deg[node, 0, 0] = 1.0 / T.sqrt(total + deg_norm_floor)

        return main

    return edge_degree_forward


@lru_cache(maxsize=1)
def _backward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def edge_degree_backward(n_nodes, n_edges, block_edges, dtype=T.float32):
        @T.prim_func
        def main(
            edge_env: T.Tensor((n_edges, 1), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            inv_sqrt_deg: T.Tensor((n_nodes, 1, 1), dtype),
            grad_deg: T.Tensor((n_nodes,), dtype),
            grad_inv_sqrt_deg: T.Tensor((n_nodes, 1, 1), dtype),
            grad_edge_env: T.Tensor((n_edges, 1), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block_edges), threads=128) as (pid,):
                for edge_offset in T.Parallel(block_edges):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges:
                        node = dst[edge]
                        inv = inv_sqrt_deg[node, 0, 0]
                        grad_deg_total = grad_deg[node] - (
                            0.5 * grad_inv_sqrt_deg[node, 0, 0] * inv * inv * inv
                        )
                        grad_edge_env[edge, 0] = (
                            2.0 * edge_env[edge, 0] * grad_deg_total
                        )

        return main

    return edge_degree_backward


def _check_inputs(edge_env, dst, n_nodes, deg_norm_floor):
    if edge_env.ndim != 2 or edge_env.shape[1] != 1:
        raise ValueError("edge_env must have shape (n_edges, 1)")
    if dst.ndim != 1 or dst.shape[0] != edge_env.shape[0]:
        raise ValueError("dst must have shape (n_edges,)")
    if dst.dtype != torch.int64:
        raise TypeError("dst must use torch.int64")
    if edge_env.dtype != torch.float32:
        raise TypeError("TileLang edge degree requires float32 input")
    if edge_env.device != dst.device:
        raise ValueError("edge_env and dst must be on the same device")
    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative")
    if deg_norm_floor < 0.0:
        raise ValueError("deg_norm_floor must be non-negative")
    if not edge_env.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang edge degree requires a HIP/DCU tensor")


def _forward(edge_env, dst, n_nodes, deg_norm_floor):
    deg = torch.empty((n_nodes,), dtype=edge_env.dtype, device=edge_env.device)
    inv_sqrt_deg = torch.empty(
        (n_nodes, 1, 1), dtype=edge_env.dtype, device=edge_env.device
    )
    if n_nodes:
        order, row_ptr = _build_csr(dst, n_nodes)
        _, T = _tilelang_modules()
        _forward_factory()(n_nodes, dst.shape[0], deg_norm_floor, T.float32)(
            edge_env.contiguous(),
            order.contiguous(),
            row_ptr.contiguous(),
            deg,
            inv_sqrt_deg,
        )
    return deg, inv_sqrt_deg


def _backward(grad_deg, grad_inv_sqrt_deg, edge_env, dst, inv_sqrt_deg, n_nodes):
    grad_edge_env = torch.empty_like(edge_env)
    if edge_env.shape[0]:
        _, T = _tilelang_modules()
        _backward_factory()(n_nodes, edge_env.shape[0], _BLOCK_EDGES, T.float32)(
            edge_env.contiguous(),
            dst.contiguous(),
            inv_sqrt_deg.contiguous(),
            grad_deg.contiguous(),
            grad_inv_sqrt_deg.contiguous(),
            grad_edge_env,
        )
    return grad_edge_env


class TileLangEdgeDegree(torch.autograd.Function):
    """Compute smooth destination degrees and inverse-square-root scaling."""

    @staticmethod
    def forward(ctx, edge_env, dst, n_nodes, deg_norm_floor):
        _check_inputs(edge_env, dst, n_nodes, deg_norm_floor)
        edge_env = edge_env.contiguous()
        dst = dst.contiguous()
        ctx.n_nodes = int(n_nodes)
        ctx.deg_norm_floor = float(deg_norm_floor)
        deg, inv_sqrt_deg = _forward(edge_env, dst, ctx.n_nodes, ctx.deg_norm_floor)
        ctx.save_for_backward(edge_env, dst, inv_sqrt_deg)
        return deg, inv_sqrt_deg

    @staticmethod
    def backward(ctx, grad_deg, grad_inv_sqrt_deg):
        edge_env, dst, inv_sqrt_deg = ctx.saved_tensors
        if grad_deg is None:
            grad_deg = torch.zeros(
                (ctx.n_nodes,), dtype=edge_env.dtype, device=edge_env.device
            )
        if grad_inv_sqrt_deg is None:
            grad_inv_sqrt_deg = torch.zeros_like(inv_sqrt_deg)
        grad_edge_env = _backward(
            grad_deg,
            grad_inv_sqrt_deg,
            edge_env,
            dst,
            inv_sqrt_deg,
            ctx.n_nodes,
        )
        return (
            grad_edge_env if ctx.needs_input_grad[0] else None,
            None,
            None,
            None,
        )


def tilelang_edge_degree(edge_env, dst, n_nodes, deg_norm_floor):
    """Compute destination ``sum(edge_env**2)`` and its inverse-sqrt scale."""
    return TileLangEdgeDegree.apply(edge_env, dst, int(n_nodes), float(deg_norm_floor))


def edge_degree_reference(edge_env, dst, n_nodes, deg_norm_floor):
    """Differentiable PyTorch reference for smooth destination degrees."""
    deg = torch.zeros((n_nodes,), dtype=edge_env.dtype, device=edge_env.device)
    deg = deg.scatter_add(0, dst, edge_env.reshape(-1).square())
    inv_sqrt_deg = torch.rsqrt(deg + deg_norm_floor).reshape(n_nodes, 1, 1)
    return deg, inv_sqrt_deg


__all__ = ["TileLangEdgeDegree", "edge_degree_reference", "tilelang_edge_degree"]
