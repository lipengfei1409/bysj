"""Standalone TileLang regularized edge length and unit-vector operator."""

from functools import lru_cache

import torch

_BLOCK = 128


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang edge geometry requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def edge_geometry_forward(n_edges, block, eps, dtype=T.float32):
        @T.prim_func
        def main(
            edge_vec: T.Tensor((n_edges, 3), dtype),
            edge_len: T.Tensor((n_edges, 1), dtype),
            edge_unit: T.Tensor((n_edges, 3), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block), threads=128) as (pid,):
                for edge_offset in T.Parallel(block):
                    edge = pid * block + edge_offset
                    if edge < n_edges:
                        x = edge_vec[edge, 0]
                        y = edge_vec[edge, 1]
                        z = edge_vec[edge, 2]
                        length = T.sqrt(x * x + y * y + z * z + eps * eps)
                        edge_len[edge, 0] = length
                        edge_unit[edge, 0] = x / length
                        edge_unit[edge, 1] = y / length
                        edge_unit[edge, 2] = z / length

        return main

    return edge_geometry_forward


@lru_cache(maxsize=1)
def _backward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def edge_geometry_backward(n_edges, block, eps, dtype=T.float32):
        @T.prim_func
        def main(
            grad_len: T.Tensor((n_edges, 1), dtype),
            grad_unit: T.Tensor((n_edges, 3), dtype),
            edge_vec: T.Tensor((n_edges, 3), dtype),
            grad_edge_vec: T.Tensor((n_edges, 3), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block), threads=128) as (pid,):
                for edge_offset in T.Parallel(block):
                    edge = pid * block + edge_offset
                    if edge < n_edges:
                        x = edge_vec[edge, 0]
                        y = edge_vec[edge, 1]
                        z = edge_vec[edge, 2]
                        length = T.sqrt(x * x + y * y + z * z + eps * eps)
                        dot = (
                            grad_unit[edge, 0] * x
                            + grad_unit[edge, 1] * y
                            + grad_unit[edge, 2] * z
                        )
                        inv_length = 1.0 / length
                        inv_length3 = inv_length * inv_length * inv_length
                        grad_edge_vec[edge, 0] = (
                            grad_len[edge, 0] * x * inv_length
                            + grad_unit[edge, 0] * inv_length
                            - x * dot * inv_length3
                        )
                        grad_edge_vec[edge, 1] = (
                            grad_len[edge, 0] * y * inv_length
                            + grad_unit[edge, 1] * inv_length
                            - y * dot * inv_length3
                        )
                        grad_edge_vec[edge, 2] = (
                            grad_len[edge, 0] * z * inv_length
                            + grad_unit[edge, 2] * inv_length
                            - z * dot * inv_length3
                        )

        return main

    return edge_geometry_backward


def _check_inputs(edge_vec):
    if edge_vec.ndim != 2 or edge_vec.shape[1] != 3:
        raise ValueError("edge_vec must have shape (n_edges, 3)")
    if edge_vec.dtype != torch.float32:
        raise TypeError("TileLang edge geometry requires float32 input")
    if not edge_vec.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang edge geometry requires a HIP/DCU tensor")


def _forward(edge_vec, eps):
    _check_inputs(edge_vec)
    n_edges = edge_vec.shape[0]
    edge_len = torch.empty((n_edges, 1), dtype=edge_vec.dtype, device=edge_vec.device)
    edge_unit = torch.empty_like(edge_vec)
    if n_edges:
        _, T = _tilelang_modules()
        kernel = _forward_factory()(n_edges, _BLOCK, eps, T.float32)
        kernel(edge_vec.contiguous(), edge_len, edge_unit)
    return edge_len, edge_unit


def _backward(grad_len, grad_unit, edge_vec, eps):
    grad_edge_vec = torch.empty_like(edge_vec)
    if edge_vec.shape[0]:
        _, T = _tilelang_modules()
        kernel = _backward_factory()(edge_vec.shape[0], _BLOCK, eps, T.float32)
        kernel(
            grad_len.contiguous(),
            grad_unit.contiguous(),
            edge_vec.contiguous(),
            grad_edge_vec,
        )
    return grad_edge_vec


class TileLangEdgeGeometry(torch.autograd.Function):
    """Compute safe edge lengths and unit vectors with analytic vector backward."""

    @staticmethod
    def forward(ctx, edge_vec, eps=1.0e-7):
        _check_inputs(edge_vec)
        edge_vec = edge_vec.contiguous()
        ctx.eps = float(eps)
        ctx.save_for_backward(edge_vec)
        return _forward(edge_vec, ctx.eps)

    @staticmethod
    def backward(ctx, grad_len, grad_unit):
        (edge_vec,) = ctx.saved_tensors
        if grad_len is None:
            grad_len = torch.zeros(
                (edge_vec.shape[0], 1), dtype=edge_vec.dtype, device=edge_vec.device
            )
        if grad_unit is None:
            grad_unit = torch.zeros_like(edge_vec)
        return _backward(grad_len, grad_unit, edge_vec, ctx.eps), None


def tilelang_edge_geometry(edge_vec, eps=1.0e-7):
    """Return regularized lengths and unit vectors for an edge-vector array."""
    return TileLangEdgeGeometry.apply(edge_vec, float(eps))


def edge_geometry_reference(edge_vec, eps=1.0e-7):
    """Differentiable PyTorch reference matching DPA4's regularized norm."""
    length = torch.sqrt((edge_vec * edge_vec).sum(dim=-1, keepdim=True) + eps * eps)
    return length, edge_vec / length


__all__ = ["TileLangEdgeGeometry", "edge_geometry_reference", "tilelang_edge_geometry"]
