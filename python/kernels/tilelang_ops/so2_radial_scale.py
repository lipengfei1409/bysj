"""Standalone TileLang per-degree radial modulation in the SO(2) local frame."""

from functools import lru_cache

import torch

_BLOCK = 128


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang SO2 radial scale requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def radial_scale_forward(n_edges, n_degree, channels, block, dtype=T.float32):
        @T.prim_func
        def main(
            values: T.Tensor((n_edges, n_degree, channels), dtype),
            radial: T.Tensor((n_edges, n_degree, channels), dtype),
            output: T.Tensor((n_edges, n_degree, channels), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges * n_degree * channels, block), threads=128
            ) as (pid,):
                for offset in T.Parallel(block):
                    index = pid * block + offset
                    if index < n_edges * n_degree * channels:
                        output[
                            index // (n_degree * channels),
                            (index // channels) % n_degree,
                            index % channels,
                        ] = (
                            values[
                                index // (n_degree * channels),
                                (index // channels) % n_degree,
                                index % channels,
                            ]
                            * radial[
                                index // (n_degree * channels),
                                (index // channels) % n_degree,
                                index % channels,
                            ]
                        )

        return main

    return radial_scale_forward


@lru_cache(maxsize=1)
def _backward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def radial_scale_backward(n_edges, n_degree, channels, block, dtype=T.float32):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_edges, n_degree, channels), dtype),
            values: T.Tensor((n_edges, n_degree, channels), dtype),
            radial: T.Tensor((n_edges, n_degree, channels), dtype),
            grad_values: T.Tensor((n_edges, n_degree, channels), dtype),
            grad_radial: T.Tensor((n_edges, n_degree, channels), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges * n_degree * channels, block), threads=128
            ) as (pid,):
                for offset in T.Parallel(block):
                    index = pid * block + offset
                    if index < n_edges * n_degree * channels:
                        edge = index // (n_degree * channels)
                        degree = (index // channels) % n_degree
                        channel = index % channels
                        upstream = grad_output[edge, degree, channel]
                        grad_values[edge, degree, channel] = (
                            upstream * radial[edge, degree, channel]
                        )
                        grad_radial[edge, degree, channel] = (
                            upstream * values[edge, degree, channel]
                        )

        return main

    return radial_scale_backward


def _check_inputs(values, radial):
    if values.ndim != 3 or radial.shape != values.shape:
        raise ValueError(
            "values and radial must have the same (n_edges, n_degree, channels) shape"
        )
    if values.dtype != torch.float32 or radial.dtype != torch.float32:
        raise TypeError("TileLang SO2 radial scale requires float32 inputs")
    if values.device != radial.device:
        raise ValueError("values and radial must be on the same device")
    if not values.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang SO2 radial scale requires a HIP/DCU tensor")


def _forward(values, radial):
    _check_inputs(values, radial)
    n_edges, n_degree, channels = values.shape
    output = torch.empty_like(values)
    if values.numel():
        _, T = _tilelang_modules()
        kernel = _forward_factory()(n_edges, n_degree, channels, _BLOCK, T.float32)
        kernel(values.contiguous(), radial.contiguous(), output)
    return output


def _backward(grad_output, values, radial):
    grad_values = torch.empty_like(values)
    grad_radial = torch.empty_like(radial)
    if values.numel():
        _, T = _tilelang_modules()
        kernel = _backward_factory()(*values.shape, _BLOCK, T.float32)
        kernel(
            grad_output.contiguous(),
            values.contiguous(),
            radial.contiguous(),
            grad_values,
            grad_radial,
        )
    return grad_values, grad_radial


class TileLangSO2RadialScale(torch.autograd.Function):
    """Elementwise radial modulation with gradients for both feature inputs."""

    @staticmethod
    def forward(ctx, values, radial):
        _check_inputs(values, radial)
        values, radial = values.contiguous(), radial.contiguous()
        ctx.save_for_backward(values, radial)
        return _forward(values, radial)

    @staticmethod
    def backward(ctx, grad_output):
        values, radial = ctx.saved_tensors
        if not any(ctx.needs_input_grad):
            return None, None
        return _backward(grad_output, values, radial)


def tilelang_so2_radial_scale(values, radial):
    """Apply the l/m degree-expanded radial scale to edge-local features."""
    return TileLangSO2RadialScale.apply(values, radial)


def so2_radial_scale_reference(values, radial):
    """Differentiable PyTorch reference for SO(2) radial modulation."""
    if values.shape != radial.shape or values.ndim != 3:
        raise ValueError("values and radial must have the same rank-3 shape")
    return values * radial


__all__ = [
    "TileLangSO2RadialScale",
    "so2_radial_scale_reference",
    "tilelang_so2_radial_scale",
]
