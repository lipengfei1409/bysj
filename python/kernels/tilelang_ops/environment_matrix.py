"""Standalone TileLang DPA4 environment-matrix geometry transform."""

from functools import lru_cache

import torch

_BLOCK = 128


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang environment matrix requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def environment_matrix_forward(n_edges, block, eps, dtype=T.float32):
        @T.prim_func
        def main(
            edge_vec: T.Tensor((n_edges, 3), dtype),
            edge_env: T.Tensor((n_edges, 1), dtype),
            r_tilde: T.Tensor((n_edges, 4), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block), threads=128) as (pid,):
                for edge_offset in T.Parallel(block):
                    edge = pid * block + edge_offset
                    if edge < n_edges:
                        x = edge_vec[edge, 0]
                        y = edge_vec[edge, 1]
                        z = edge_vec[edge, 2]
                        inv_r = 1.0 / T.sqrt(x * x + y * y + z * z + eps * eps)
                        scale = edge_env[edge, 0] * inv_r
                        r_tilde[edge, 0] = scale
                        r_tilde[edge, 1] = scale * (x * inv_r)
                        r_tilde[edge, 2] = scale * (y * inv_r)
                        r_tilde[edge, 3] = scale * (z * inv_r)

        return main

    return environment_matrix_forward


@lru_cache(maxsize=1)
def _backward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def environment_matrix_backward(n_edges, block, eps, dtype=T.float32):
        @T.prim_func
        def main(
            grad_r_tilde: T.Tensor((n_edges, 4), dtype),
            edge_vec: T.Tensor((n_edges, 3), dtype),
            edge_env: T.Tensor((n_edges, 1), dtype),
            grad_edge_vec: T.Tensor((n_edges, 3), dtype),
            grad_edge_env: T.Tensor((n_edges, 1), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block), threads=128) as (pid,):
                for edge_offset in T.Parallel(block):
                    edge = pid * block + edge_offset
                    if edge < n_edges:
                        x = edge_vec[edge, 0]
                        y = edge_vec[edge, 1]
                        z = edge_vec[edge, 2]
                        inv_r = 1.0 / T.sqrt(x * x + y * y + z * z + eps * eps)
                        inv_r2 = inv_r * inv_r
                        inv_r3 = inv_r2 * inv_r
                        inv_r4 = inv_r2 * inv_r2
                        scale = edge_env[edge, 0]
                        g0 = grad_r_tilde[edge, 0]
                        gx = grad_r_tilde[edge, 1]
                        gy = grad_r_tilde[edge, 2]
                        gz = grad_r_tilde[edge, 3]
                        dot = gx * x + gy * y + gz * z
                        grad_edge_vec[edge, 0] = -g0 * scale * x * inv_r3 + scale * (
                            gx * inv_r2 - 2.0 * x * dot * inv_r4
                        )
                        grad_edge_vec[edge, 1] = -g0 * scale * y * inv_r3 + scale * (
                            gy * inv_r2 - 2.0 * y * dot * inv_r4
                        )
                        grad_edge_vec[edge, 2] = -g0 * scale * z * inv_r3 + scale * (
                            gz * inv_r2 - 2.0 * z * dot * inv_r4
                        )
                        grad_edge_env[edge, 0] = g0 * inv_r + dot * inv_r2

        return main

    return environment_matrix_backward


def _check_inputs(edge_vec, edge_env):
    if edge_vec.ndim != 2 or edge_vec.shape[1] != 3:
        raise ValueError("edge_vec must have shape (n_edges, 3)")
    if edge_env.shape != (edge_vec.shape[0], 1):
        raise ValueError("edge_env must have shape (n_edges, 1)")
    if edge_vec.dtype != torch.float32 or edge_env.dtype != torch.float32:
        raise TypeError("TileLang environment matrix requires float32 inputs")
    if edge_vec.device != edge_env.device:
        raise ValueError("edge_vec and edge_env must be on the same device")
    if not edge_vec.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang environment matrix requires a HIP/DCU tensor")


def _forward(edge_vec, edge_env, eps):
    n_edges = edge_vec.shape[0]
    r_tilde = torch.empty((n_edges, 4), dtype=edge_vec.dtype, device=edge_vec.device)
    if n_edges:
        _, T = _tilelang_modules()
        _forward_factory()(n_edges, _BLOCK, eps, T.float32)(
            edge_vec.contiguous(), edge_env.contiguous(), r_tilde
        )
    return r_tilde


def _backward(grad_r_tilde, edge_vec, edge_env, eps):
    grad_edge_vec = torch.empty_like(edge_vec)
    grad_edge_env = torch.empty_like(edge_env)
    if edge_vec.shape[0]:
        _, T = _tilelang_modules()
        _backward_factory()(edge_vec.shape[0], _BLOCK, eps, T.float32)(
            grad_r_tilde.contiguous(),
            edge_vec.contiguous(),
            edge_env.contiguous(),
            grad_edge_vec,
            grad_edge_env,
        )
    return grad_edge_vec, grad_edge_env


class TileLangEnvironmentMatrix(torch.autograd.Function):
    """Build ``[s, s*r_hat]`` with coordinate and envelope backward."""

    @staticmethod
    def forward(ctx, edge_vec, edge_env, eps=1.0e-7):
        _check_inputs(edge_vec, edge_env)
        edge_vec = edge_vec.contiguous()
        edge_env = edge_env.contiguous()
        ctx.eps = float(eps)
        ctx.save_for_backward(edge_vec, edge_env)
        return _forward(edge_vec, edge_env, ctx.eps)

    @staticmethod
    def backward(ctx, grad_r_tilde):
        edge_vec, edge_env = ctx.saved_tensors
        grad_edge_vec, grad_edge_env = _backward(
            grad_r_tilde, edge_vec, edge_env, ctx.eps
        )
        return (
            grad_edge_vec if ctx.needs_input_grad[0] else None,
            grad_edge_env if ctx.needs_input_grad[1] else None,
            None,
        )


def tilelang_environment_matrix(edge_vec, edge_env, eps=1.0e-7):
    """Return the regularized scalar/vector environment matrix channels."""
    return TileLangEnvironmentMatrix.apply(edge_vec, edge_env, float(eps))


def environment_matrix_reference(edge_vec, edge_env, eps=1.0e-7):
    """Differentiable PyTorch reference for DPA4's environment matrix."""
    inv_r = torch.rsqrt((edge_vec * edge_vec).sum(-1, keepdim=True) + eps * eps)
    scale = edge_env * inv_r
    r_hat = edge_vec * inv_r
    return torch.cat((scale, scale * r_hat), dim=-1)


__all__ = [
    "TileLangEnvironmentMatrix",
    "environment_matrix_reference",
    "tilelang_environment_matrix",
]
