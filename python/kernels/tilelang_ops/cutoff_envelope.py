"""Standalone TileLang C3 polynomial cutoff envelope and distance backward."""

import math
from functools import lru_cache

import torch

_BLOCK = 128


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang C3 cutoff envelope requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=16)
def _kernel_factory(exponent: int, backward: bool):
    tilelang, T = _tilelang_modules()
    coefficients = tuple(float(math.comb(k + 3, 3)) for k in range(exponent))

    if backward:

        @tilelang.jit(target="hip")
        def envelope_backward(n_edges, rcut, block, dtype=T.float32):
            @T.prim_func
            def main(
                distance: T.Tensor((n_edges, 1), dtype),
                grad_output: T.Tensor((n_edges, 1), dtype),
                grad_distance: T.Tensor((n_edges, 1), dtype),
            ):
                with T.Kernel(T.ceildiv(n_edges, block), threads=128) as (pid,):
                    for i in T.Parallel(block):
                        edge = pid * block + i
                        if edge < n_edges:
                            r = distance[edge, 0]
                            x = T.if_then_else(
                                r <= 0.0,
                                0.0,
                                T.if_then_else(r >= rcut, 1.0, r / rcut),
                            )
                            u = 1.0 - x
                            p = coefficients[-1]
                            for k in T.static_range(exponent - 2, -1, -1):
                                p = coefficients[k] + x * p
                            if exponent == 1:
                                dp = 0.0
                            else:
                                dp = (exponent - 1) * coefficients[-1]
                                for k in T.static_range(exponent - 2, 0, -1):
                                    dp = k * coefficients[k] + x * dp
                            value = (-4.0 * u * u * u * p + u * u * u * u * dp) / rcut
                            active = (r >= 0.0) & (r < rcut)
                            grad_distance[edge, 0] = T.if_then_else(
                                active, grad_output[edge, 0] * value, 0.0
                            )

            return main

        return envelope_backward

    @tilelang.jit(target="hip")
    def envelope_forward(n_edges, rcut, block, dtype=T.float32):
        @T.prim_func
        def main(
            distance: T.Tensor((n_edges, 1), dtype),
            output: T.Tensor((n_edges, 1), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block), threads=128) as (pid,):
                for i in T.Parallel(block):
                    edge = pid * block + i
                    if edge < n_edges:
                        r = distance[edge, 0]
                        x = T.if_then_else(
                            r <= 0.0,
                            0.0,
                            T.if_then_else(r >= rcut, 1.0, r / rcut),
                        )
                        u = 1.0 - x
                        p = coefficients[-1]
                        for k in T.static_range(exponent - 2, -1, -1):
                            p = coefficients[k] + x * p
                        value = u * u * u * u * p
                        output[edge, 0] = T.if_then_else(
                            r <= 0.0,
                            1.0,
                            T.if_then_else(r >= rcut, 0.0, value),
                        )

        return main

    return envelope_forward


def _check_inputs(distance: torch.Tensor, rcut: float) -> None:
    if distance.ndim != 2 or distance.shape[1] != 1:
        raise ValueError("distance must have shape (n_edges, 1)")
    if distance.dtype != torch.float32:
        raise TypeError("TileLang C3 cutoff envelope requires float32 input")
    if rcut <= 0.0:
        raise ValueError("rcut must be positive")
    if not distance.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang C3 cutoff envelope requires a HIP/DCU tensor")


def _forward(distance: torch.Tensor, rcut: float, exponent: int) -> torch.Tensor:
    _check_inputs(distance, rcut)
    output = torch.empty_like(distance)
    if distance.numel() == 0:
        return output
    _, T = _tilelang_modules()
    kernel = _kernel_factory(exponent, False)(
        distance.shape[0], rcut, _BLOCK, T.float32
    )
    kernel(distance.contiguous(), output)
    return output


def _backward(
    grad_output: torch.Tensor,
    distance: torch.Tensor,
    rcut: float,
    exponent: int,
) -> torch.Tensor:
    grad_distance = torch.empty_like(distance)
    if distance.numel() == 0:
        return grad_distance
    _, T = _tilelang_modules()
    kernel = _kernel_factory(exponent, True)(distance.shape[0], rcut, _BLOCK, T.float32)
    kernel(grad_output.contiguous(), distance.contiguous(), grad_distance)
    return grad_distance


class TileLangC3CutoffEnvelope(torch.autograd.Function):
    """Autograd wrapper for a scalar-parameter C3 cutoff envelope."""

    @staticmethod
    def forward(ctx, distance: torch.Tensor, rcut: float, exponent: int):
        _check_inputs(distance, rcut)
        if exponent <= 0:
            raise ValueError("exponent must be positive")
        ctx.rcut = float(rcut)
        ctx.exponent = int(exponent)
        distance = distance.contiguous()
        ctx.save_for_backward(distance)
        return _forward(distance, ctx.rcut, ctx.exponent)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (distance,) = ctx.saved_tensors
        grad_distance = _backward(grad_output, distance, ctx.rcut, ctx.exponent)
        return grad_distance, None, None


def tilelang_c3_cutoff_envelope(
    distance: torch.Tensor, rcut: float, exponent: int = 5
) -> torch.Tensor:
    """Evaluate the C3 polynomial envelope with a TileLang HIP/DCU kernel."""
    return TileLangC3CutoffEnvelope.apply(distance, float(rcut), int(exponent))


def c3_cutoff_envelope_reference(
    distance: torch.Tensor, rcut: float, exponent: int = 5
) -> torch.Tensor:
    """Differentiable PyTorch reference matching the dpmodel factorization."""
    if rcut <= 0.0 or exponent <= 0:
        raise ValueError("rcut and exponent must be positive")
    u = ((rcut - distance) / rcut).clamp(0.0, 1.0)
    x = 1.0 - u
    series = torch.full_like(x, float(math.comb(exponent + 2, 3)))
    for k in range(exponent - 2, -1, -1):
        series = float(math.comb(k + 3, 3)) + x * series
    return u.pow(4) * series


__all__ = [
    "TileLangC3CutoffEnvelope",
    "c3_cutoff_envelope_reference",
    "tilelang_c3_cutoff_envelope",
]
