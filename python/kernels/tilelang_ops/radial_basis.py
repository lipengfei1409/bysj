"""Standalone TileLang Bessel radial-basis operator for inference.

The cutoff envelope remains a separate PyTorch operation. This module owns only
``phi_n(r) = w_n * sinc(w_n * r / pi)`` and its distance backward.
"""

import math
from functools import lru_cache

import torch

_BLOCK_EDGES = 32
_SMALL_X = 1.0e-3


@lru_cache(maxsize=1)
def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang radial basis requires TileLang to be installed "
            "in the active environment"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _radial_forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def radial_forward(n_edges, n_radial, block_edges, block_radial, dtype=T.float32):
        @T.prim_func
        def main(
            distance: T.Tensor((n_edges, 1), dtype),
            frequencies: T.Tensor((n_radial,), dtype),
            output: T.Tensor((n_edges, n_radial), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block_edges), threads=128) as (pid,):
                for edge_offset, radial_offset in T.Parallel(block_edges, block_radial):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges and radial_offset < n_radial:
                        frequency = frequencies[radial_offset]
                        x = distance[edge, 0] * frequency
                        if T.abs(x) < _SMALL_X:
                            x2 = x * x
                            sinc = 1.0 - x2 / 6.0 + x2 * x2 / 120.0
                        else:
                            sinc = T.sin(x) / x
                        output[edge, radial_offset] = frequency * sinc

        return main

    return radial_forward


@lru_cache(maxsize=1)
def _radial_backward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def radial_backward(n_edges, n_radial, block_edges, block_radial, dtype=T.float32):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_edges, n_radial), dtype),
            distance: T.Tensor((n_edges, 1), dtype),
            frequencies: T.Tensor((n_radial,), dtype),
            grad_distance: T.Tensor((n_edges, 1), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block_edges), threads=128) as (pid,):
                contributions = T.alloc_fragment((block_edges, block_radial), T.float32)
                grad_rows = T.alloc_fragment((block_edges,), T.float32)
                for edge_offset, radial_offset in T.Parallel(block_edges, block_radial):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges and radial_offset < n_radial:
                        frequency = frequencies[radial_offset]
                        x = distance[edge, 0] * frequency
                        if T.abs(x) < _SMALL_X:
                            x2 = x * x
                            derivative = (
                                frequency * frequency * (-x / 3.0 + x * x2 / 30.0)
                            )
                        else:
                            derivative = (
                                frequency
                                * frequency
                                * ((x * T.cos(x) - T.sin(x)) / (x * x))
                            )
                        contributions[edge_offset, radial_offset] = (
                            grad_output[edge, radial_offset] * derivative
                        )
                    else:
                        contributions[edge_offset, radial_offset] = 0.0

                T.reduce_sum(contributions, grad_rows, dim=1, clear=True)
                for edge_offset in T.Parallel(block_edges):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges:
                        grad_distance[edge, 0] = grad_rows[edge_offset]

        return main

    return radial_backward


def _check_inputs(distance: torch.Tensor, frequencies: torch.Tensor) -> None:
    if distance.ndim != 2 or distance.shape[1] != 1:
        raise ValueError("distance must have shape (n_edges, 1)")
    if frequencies.ndim not in (1, 2) or (
        frequencies.ndim == 2 and frequencies.shape[0] != 1
    ):
        raise ValueError("frequencies must have shape (n_radial,) or (1, n_radial)")
    if distance.dtype != torch.float32 or frequencies.dtype != torch.float32:
        raise TypeError(
            "TileLang Bessel radial basis currently requires float32 inputs"
        )
    if distance.device != frequencies.device:
        raise ValueError("distance and frequencies must be on the same device")
    if not distance.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang Bessel radial basis requires a HIP/DCU tensor")


def _tilelang_bessel_forward(
    distance: torch.Tensor, frequencies: torch.Tensor
) -> torch.Tensor:
    _check_inputs(distance, frequencies)
    n_edges = distance.shape[0]
    frequencies = frequencies.reshape(-1).contiguous()
    n_radial = frequencies.numel()
    if n_radial == 0:
        raise ValueError("frequencies must contain at least one value")
    output = torch.empty(
        (n_edges, n_radial), dtype=distance.dtype, device=distance.device
    )
    if n_edges == 0:
        return output
    block_radial = 1 << (n_radial - 1).bit_length()
    if block_radial > 128:
        raise ValueError(
            "TileLang Bessel radial basis supports at most 128 frequencies"
        )
    _, T = _tilelang_modules()
    kernel = _radial_forward_factory()(
        n_edges, n_radial, _BLOCK_EDGES, block_radial, T.float32
    )
    kernel(distance.contiguous(), frequencies, output)
    return output


def _tilelang_bessel_backward(
    grad_output: torch.Tensor,
    distance: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    n_edges = distance.shape[0]
    frequencies = frequencies.reshape(-1).contiguous()
    n_radial = frequencies.numel()
    grad_distance = torch.empty_like(distance)
    if n_edges == 0:
        return grad_distance
    block_radial = 1 << (n_radial - 1).bit_length()
    _, T = _tilelang_modules()
    kernel = _radial_backward_factory()(
        n_edges, n_radial, _BLOCK_EDGES, block_radial, T.float32
    )
    kernel(grad_output.contiguous(), distance.contiguous(), frequencies, grad_distance)
    return grad_distance


class TileLangBesselRadialBasis(torch.autograd.Function):
    """PyTorch autograd wrapper around separate TileLang forward/backward kernels.

    This first implementation supports inference forces only: the derivative
    with respect to distance is implemented; radial-frequency gradients are
    intentionally not returned. Keep training on the PyTorch reference path.
    """

    @staticmethod
    def forward(ctx, distance: torch.Tensor, frequencies: torch.Tensor):
        _check_inputs(distance, frequencies)
        distance_contiguous = distance.contiguous()
        frequencies_contiguous = frequencies.reshape(-1).contiguous()
        ctx.input_shape = tuple(distance.shape)
        ctx.save_for_backward(distance_contiguous, frequencies_contiguous)
        return _tilelang_bessel_forward(distance_contiguous, frequencies_contiguous)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        distance, frequencies = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        grad_distance = _tilelang_bessel_backward(grad_output, distance, frequencies)
        return grad_distance.reshape(ctx.input_shape), None


def tilelang_bessel_radial_basis(
    distance: torch.Tensor, frequencies: torch.Tensor
) -> torch.Tensor:
    """Evaluate the raw Bessel radial basis with an explicit PyTorch autograd op."""
    return TileLangBesselRadialBasis.apply(distance, frequencies)


def bessel_radial_basis_reference(
    distance: torch.Tensor, frequencies: torch.Tensor
) -> torch.Tensor:
    """Differentiable PyTorch reference for tests and numerical comparison."""
    frequencies = frequencies.reshape(-1)
    return frequencies * torch.sinc(distance * frequencies / math.pi)


__all__ = [
    "TileLangBesselRadialBasis",
    "bessel_radial_basis_reference",
    "tilelang_bessel_radial_basis",
]
