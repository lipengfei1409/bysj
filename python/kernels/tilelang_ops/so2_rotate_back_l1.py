"""Standalone TileLang inverse l=1 rotation from the local SO(2) frame."""

from functools import lru_cache

import torch

_BLOCK_EDGES = 8
_BLOCK_CHANNELS = 32


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang SO2 rotation requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def rotate_back_forward(
        n_edges, n_focus, channels, block_edges, block_channels, dtype=T.float32
    ):
        @T.prim_func
        def main(
            x_local: T.Tensor((n_edges, n_focus, 4, channels), dtype),
            rotation: T.Tensor((n_edges, 4, 4), dtype),
            output: T.Tensor((n_edges, 4, n_focus * channels), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                T.ceildiv(n_focus * channels, block_channels),
                threads=128,
            ) as (pid_edge, pid_channel):
                for edge_offset, row, channel_offset in T.Parallel(
                    block_edges, 4, block_channels
                ):
                    edge = pid_edge * block_edges + edge_offset
                    flat_channel = pid_channel * block_channels + channel_offset
                    if edge < n_edges and flat_channel < n_focus * channels:
                        focus = flat_channel // channels
                        channel = flat_channel - focus * channels
                        acc = 0.0
                        for col in T.serial(4):
                            acc += (
                                rotation[
                                    edge,
                                    row,
                                    T.if_then_else(
                                        col == 1, 2, T.if_then_else(col == 2, 1, col)
                                    ),
                                ]
                                * x_local[edge, focus, col, channel]
                            )
                        output[edge, row, flat_channel] = acc

        return main

    return rotate_back_forward


@lru_cache(maxsize=1)
def _backward_local_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def rotate_back_backward_local(
        n_edges, n_focus, channels, block_edges, block_channels, dtype=T.float32
    ):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_edges, 4, n_focus * channels), dtype),
            rotation: T.Tensor((n_edges, 4, 4), dtype),
            grad_x_local: T.Tensor((n_edges, n_focus, 4, channels), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                T.ceildiv(n_focus * channels, block_channels),
                threads=128,
            ) as (pid_edge, pid_channel):
                for edge_offset, col, channel_offset in T.Parallel(
                    block_edges, 4, block_channels
                ):
                    edge = pid_edge * block_edges + edge_offset
                    flat_channel = pid_channel * block_channels + channel_offset
                    if edge < n_edges and flat_channel < n_focus * channels:
                        focus = flat_channel // channels
                        channel = flat_channel - focus * channels
                        acc = 0.0
                        for row in T.serial(4):
                            projected_col = T.if_then_else(
                                col == 1, 2, T.if_then_else(col == 2, 1, col)
                            )
                            acc += (
                                rotation[edge, row, projected_col]
                                * grad_output[edge, row, flat_channel]
                            )
                        grad_x_local[edge, focus, col, channel] = acc

        return main

    return rotate_back_backward_local


@lru_cache(maxsize=1)
def _backward_rotation_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def rotate_back_backward_rotation(
        n_edges, n_focus, channels, block_edges, dtype=T.float32
    ):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_edges, 4, n_focus * channels), dtype),
            x_local: T.Tensor((n_edges, n_focus, 4, channels), dtype),
            grad_rotation: T.Tensor((n_edges, 4, 4), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block_edges), threads=128) as (pid_edge,):
                for edge_offset, row, col in T.Parallel(block_edges, 4, 4):
                    edge = pid_edge * block_edges + edge_offset
                    if edge < n_edges:
                        acc = 0.0
                        for focus in T.serial(n_focus):
                            for channel in T.serial(channels):
                                flat_channel = focus * channels + channel
                                acc += (
                                    grad_output[edge, row, flat_channel]
                                    * x_local[edge, focus, col, channel]
                                )
                        projected_col = T.if_then_else(
                            col == 1, 2, T.if_then_else(col == 2, 1, col)
                        )
                        grad_rotation[edge, row, projected_col] = acc

        return main

    return rotate_back_backward_rotation


def _check_inputs(x_local, rotation):
    if x_local.ndim != 4 or x_local.shape[2] != 4:
        raise ValueError("x_local must have shape (n_edges, n_focus, 4, channels)")
    if rotation.ndim != 3 or rotation.shape != (x_local.shape[0], 4, 4):
        raise ValueError("rotation must have shape (n_edges, 4, 4)")
    if x_local.dtype != torch.float32 or rotation.dtype != torch.float32:
        raise TypeError("TileLang SO2 rotation requires float32 values")
    if x_local.device != rotation.device:
        raise ValueError("x_local and rotation must share a device")
    if not x_local.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang SO2 rotation requires a HIP/DCU tensor")


def _forward(x_local, rotation):
    _check_inputs(x_local, rotation)
    n_edges, n_focus, _, channels = x_local.shape
    output = torch.empty(
        (n_edges, 4, n_focus * channels), dtype=x_local.dtype, device=x_local.device
    )
    if n_edges:
        _, T = _tilelang_modules()
        kernel = _forward_factory()(
            n_edges, n_focus, channels, _BLOCK_EDGES, _BLOCK_CHANNELS, T.float32
        )
        kernel(x_local.contiguous(), rotation.contiguous(), output)
    return output


def _backward(grad_output, x_local, rotation):
    n_edges, n_focus, _, channels = x_local.shape
    grad_x_local = torch.empty_like(x_local)
    grad_rotation = torch.empty_like(rotation)
    if n_edges:
        _, T = _tilelang_modules()
        _backward_local_factory()(
            n_edges, n_focus, channels, _BLOCK_EDGES, _BLOCK_CHANNELS, T.float32
        )(grad_output.contiguous(), rotation.contiguous(), grad_x_local)
        _backward_rotation_factory()(
            n_edges, n_focus, channels, _BLOCK_EDGES, T.float32
        )(grad_output.contiguous(), x_local.contiguous(), grad_rotation)
    return grad_x_local, grad_rotation


class TileLangRotateBackL1(torch.autograd.Function):
    """Autograd wrapper for the local-to-global l=1 rotation."""

    @staticmethod
    def forward(ctx, x_local, rotation):
        _check_inputs(x_local, rotation)
        x_local = x_local.contiguous()
        rotation = rotation.contiguous()
        ctx.save_for_backward(x_local, rotation)
        return _forward(x_local, rotation)

    @staticmethod
    def backward(ctx, grad_output):
        x_local, rotation = ctx.saved_tensors
        grad_x_local, grad_rotation = _backward(grad_output, x_local, rotation)
        return (
            grad_x_local if ctx.needs_input_grad[0] else None,
            grad_rotation if ctx.needs_input_grad[1] else None,
        )


def tilelang_rotate_back_l1(x_local, rotation):
    """Rotate focus-major l=1 messages back to the packed global layout."""
    return TileLangRotateBackL1.apply(x_local, rotation)


def rotate_back_l1_reference(x_local, rotation):
    """Differentiable PyTorch reference for l=1 rotate-back."""
    x_flat = x_local.permute(0, 2, 1, 3).reshape(
        x_local.shape[0], 4, x_local.shape[1] * x_local.shape[3]
    )
    col_index = torch.tensor([0, 2, 1, 3], dtype=torch.int64, device=x_local.device)
    projected = rotation.index_select(2, col_index)
    return torch.matmul(projected, x_flat)


__all__ = [
    "TileLangRotateBackL1",
    "rotate_back_l1_reference",
    "tilelang_rotate_back_l1",
]
