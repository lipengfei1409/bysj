"""Standalone TileLang environment-seed invariant Gram projection."""

from functools import lru_cache

import torch

_BLOCK = 128


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError("TileLang environment Gram requires TileLang") from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def environment_gram_forward(
        n_nodes, coord_dim, embed_dim, axis_dim, block, dtype=T.float32
    ):
        n_outputs = n_nodes * embed_dim * axis_dim

        @T.prim_func
        def main(
            env_agg: T.Tensor((n_nodes, coord_dim, embed_dim), dtype),
            gram: T.Tensor((n_nodes, embed_dim, axis_dim), dtype),
        ):
            with T.Kernel(T.ceildiv(n_outputs, block), threads=128) as (pid,):
                for output_offset in T.Parallel(block):
                    index = pid * block + output_offset
                    if index < n_outputs:
                        node = index // (embed_dim * axis_dim)
                        rem = index % (embed_dim * axis_dim)
                        row = rem // axis_dim
                        col = rem % axis_dim
                        total = 0.0
                        for coord in T.serial(coord_dim):
                            total += (
                                env_agg[node, coord, row] * env_agg[node, coord, col]
                            )
                        gram[node, row, col] = total

        return main

    return environment_gram_forward


@lru_cache(maxsize=1)
def _backward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def environment_gram_backward(
        n_nodes, coord_dim, embed_dim, axis_dim, block, dtype=T.float32
    ):
        n_inputs = n_nodes * coord_dim * embed_dim

        @T.prim_func
        def main(
            grad_gram: T.Tensor((n_nodes, embed_dim, axis_dim), dtype),
            env_agg: T.Tensor((n_nodes, coord_dim, embed_dim), dtype),
            grad_env_agg: T.Tensor((n_nodes, coord_dim, embed_dim), dtype),
        ):
            with T.Kernel(T.ceildiv(n_inputs, block), threads=128) as (pid,):
                for input_offset in T.Parallel(block):
                    index = pid * block + input_offset
                    if index < n_inputs:
                        node = index // (coord_dim * embed_dim)
                        rem = index % (coord_dim * embed_dim)
                        coord = rem // embed_dim
                        column = rem % embed_dim
                        total = 0.0
                        for axis in T.serial(axis_dim):
                            total += (
                                grad_gram[node, column, axis]
                                * env_agg[node, coord, axis]
                            )
                        if column < axis_dim:
                            for row in T.serial(embed_dim):
                                total += (
                                    env_agg[node, coord, row]
                                    * grad_gram[node, row, column]
                                )
                        grad_env_agg[node, coord, column] = total

        return main

    return environment_gram_backward


def _check_inputs(env_agg, axis_dim):
    if env_agg.ndim != 3:
        raise ValueError("env_agg must have shape (n_nodes, coord_dim, embed_dim)")
    if not 0 < axis_dim <= env_agg.shape[2]:
        raise ValueError("axis_dim must be in [1, embed_dim]")
    if env_agg.dtype != torch.float32:
        raise TypeError("TileLang environment Gram requires float32 input")
    if not env_agg.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang environment Gram requires a HIP/DCU tensor")


def _forward(env_agg, axis_dim):
    n_nodes, coord_dim, embed_dim = env_agg.shape
    gram = torch.empty(
        (n_nodes, embed_dim, axis_dim), dtype=env_agg.dtype, device=env_agg.device
    )
    n_outputs = n_nodes * embed_dim * axis_dim
    if n_outputs:
        _, T = _tilelang_modules()
        _forward_factory()(n_nodes, coord_dim, embed_dim, axis_dim, _BLOCK, T.float32)(
            env_agg.contiguous(), gram
        )
    return gram


def _backward(grad_gram, env_agg, axis_dim):
    grad_env_agg = torch.empty_like(env_agg)
    if env_agg.numel():
        _, T = _tilelang_modules()
        _backward_factory()(
            env_agg.shape[0],
            env_agg.shape[1],
            env_agg.shape[2],
            axis_dim,
            _BLOCK,
            T.float32,
        )(grad_gram.contiguous(), env_agg.contiguous(), grad_env_agg)
    return grad_env_agg


class TileLangEnvironmentSeedGram(torch.autograd.Function):
    """Form the DPA4 environment-seed invariant Gram matrix."""

    @staticmethod
    def forward(ctx, env_agg, axis_dim):
        _check_inputs(env_agg, axis_dim)
        env_agg = env_agg.contiguous()
        ctx.axis_dim = int(axis_dim)
        ctx.save_for_backward(env_agg)
        return _forward(env_agg, ctx.axis_dim)

    @staticmethod
    def backward(ctx, grad_gram):
        (env_agg,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        return _backward(grad_gram, env_agg, ctx.axis_dim), None


def tilelang_environment_seed_gram(env_agg, axis_dim):
    """Compute ``env_agg.T @ env_agg[..., :axis_dim]`` using TileLang."""
    return TileLangEnvironmentSeedGram.apply(env_agg, int(axis_dim))


def environment_seed_gram_reference(env_agg, axis_dim):
    """Differentiable PyTorch reference for the invariant Gram projection."""
    return torch.matmul(env_agg.transpose(-1, -2), env_agg[..., :axis_dim])


__all__ = [
    "TileLangEnvironmentSeedGram",
    "environment_seed_gram_reference",
    "tilelang_environment_seed_gram",
]
