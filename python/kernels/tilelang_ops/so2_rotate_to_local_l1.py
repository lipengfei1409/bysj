"""Standalone TileLang source gather and l=1 rotation into edge-local frames."""

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
    def rotate_to_local_forward(
        n_nodes, n_edges, channels, block_edges, block_channels, dtype=T.float32
    ):
        @T.prim_func
        def main(
            x: T.Tensor((n_nodes, 4, channels), dtype),
            rotation: T.Tensor((n_edges, 4, 4), dtype),
            src: T.Tensor((n_edges,), T.int64),
            output: T.Tensor((n_edges, 4, channels), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                T.ceildiv(channels, block_channels),
                threads=128,
            ) as (pid_edge, pid_channel):
                for edge_offset, row, channel_offset in T.Parallel(
                    block_edges, 4, block_channels
                ):
                    edge = pid_edge * block_edges + edge_offset
                    channel = pid_channel * block_channels + channel_offset
                    if edge < n_edges and channel < channels:
                        source = src[edge]
                        acc = 0.0
                        for col in T.serial(4):
                            acc += (
                                rotation[
                                    edge,
                                    T.if_then_else(
                                        row == 1, 2, T.if_then_else(row == 2, 1, row)
                                    ),
                                    col,
                                ]
                                * x[source, col, channel]
                            )
                        output[edge, row, channel] = acc

        return main

    return rotate_to_local_forward


@lru_cache(maxsize=1)
def _backward_rotation_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def rotate_to_local_backward_rotation(
        n_nodes, n_edges, channels, block_edges, dtype=T.float32
    ):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_edges, 4, channels), dtype),
            x: T.Tensor((n_nodes, 4, channels), dtype),
            src: T.Tensor((n_edges,), T.int64),
            grad_rotation: T.Tensor((n_edges, 4, 4), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block_edges), threads=128) as (pid_edge,):
                for edge_offset, row, col in T.Parallel(block_edges, 4, 4):
                    edge = pid_edge * block_edges + edge_offset
                    if edge < n_edges:
                        source = src[edge]
                        acc = 0.0
                        for channel in T.serial(channels):
                            acc += (
                                grad_output[edge, row, channel]
                                * x[source, col, channel]
                            )
                        reduced_row = T.if_then_else(
                            row == 1, 2, T.if_then_else(row == 2, 1, row)
                        )
                        grad_rotation[edge, reduced_row, col] = acc

        return main

    return rotate_to_local_backward_rotation


@lru_cache(maxsize=1)
def _backward_x_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def rotate_to_local_backward_x(
        n_nodes, n_edges, channels, block_edges, block_channels, dtype=T.float32
    ):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_edges, 4, channels), dtype),
            rotation: T.Tensor((n_edges, 4, 4), dtype),
            src: T.Tensor((n_edges,), T.int64),
            grad_x: T.Tensor((n_nodes, 4, channels), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                T.ceildiv(channels, block_channels),
                threads=128,
            ) as (pid_edge, pid_channel):
                for edge_offset, col, channel_offset in T.Parallel(
                    block_edges, 4, block_channels
                ):
                    edge = pid_edge * block_edges + edge_offset
                    channel = pid_channel * block_channels + channel_offset
                    if edge < n_edges and channel < channels:
                        value = 0.0
                        for row in T.serial(4):
                            reduced_row = T.if_then_else(
                                row == 1, 2, T.if_then_else(row == 2, 1, row)
                            )
                            value += (
                                rotation[edge, reduced_row, col]
                                * grad_output[edge, row, channel]
                            )
                        T.atomic_add(grad_x[src[edge], col, channel], value)

        return main

    return rotate_to_local_backward_x


def _check_inputs(x, rotation, src):
    if x.ndim != 3 or x.shape[1] != 4:
        raise ValueError("x must have shape (n_nodes, 4, channels)")
    if rotation.ndim != 3 or rotation.shape[1:] != (4, 4):
        raise ValueError("rotation must have shape (n_edges, 4, 4)")
    if src.ndim != 1 or src.shape[0] != rotation.shape[0]:
        raise ValueError("src must have shape (n_edges,)")
    if src.dtype != torch.int64:
        raise TypeError("src must use torch.int64")
    if x.dtype != torch.float32 or rotation.dtype != torch.float32:
        raise TypeError("TileLang SO2 rotation requires float32 values")
    if x.device != rotation.device or x.device != src.device:
        raise ValueError("all SO2 rotation inputs must share a device")
    if not x.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang SO2 rotation requires a HIP/DCU tensor")


def _forward(x, rotation, src):
    _check_inputs(x, rotation, src)
    n_edges, channels = src.shape[0], x.shape[-1]
    output = torch.empty((n_edges, 4, channels), dtype=x.dtype, device=x.device)
    if n_edges:
        _, T = _tilelang_modules()
        kernel = _forward_factory()(
            x.shape[0], n_edges, channels, _BLOCK_EDGES, _BLOCK_CHANNELS, T.float32
        )
        kernel(x.contiguous(), rotation.contiguous(), src.contiguous(), output)
    return output


def _backward(grad_output, x, rotation, src):
    n_edges, channels = src.shape[0], x.shape[-1]
    grad_x = torch.zeros_like(x)
    grad_rotation = torch.empty_like(rotation)
    if n_edges:
        _, T = _tilelang_modules()
        _backward_rotation_factory()(
            x.shape[0], n_edges, channels, _BLOCK_EDGES, T.float32
        )(grad_output.contiguous(), x.contiguous(), src.contiguous(), grad_rotation)
        _backward_x_factory()(
            x.shape[0], n_edges, channels, _BLOCK_EDGES, _BLOCK_CHANNELS, T.float32
        )(grad_output.contiguous(), rotation.contiguous(), src.contiguous(), grad_x)
    return grad_x, grad_rotation


class TileLangRotateToLocalL1(torch.autograd.Function):
    """Autograd wrapper for edge gather and local-frame rotation."""

    @staticmethod
    def forward(ctx, x, rotation, src):
        _check_inputs(x, rotation, src)
        x, rotation, src = x.contiguous(), rotation.contiguous(), src.contiguous()
        ctx.save_for_backward(x, rotation, src)
        return _forward(x, rotation, src)

    @staticmethod
    def backward(ctx, grad_output):
        x, rotation, src = ctx.saved_tensors
        grad_x, grad_rotation = _backward(grad_output, x, rotation, src)
        return (
            grad_x if ctx.needs_input_grad[0] else None,
            grad_rotation if ctx.needs_input_grad[1] else None,
            None,
        )


def tilelang_rotate_to_local_l1(x, rotation, src):
    """Gather source nodes and apply their edge-local l=1 rotation."""
    return TileLangRotateToLocalL1.apply(x, rotation, src)


def rotate_to_local_l1_reference(x, rotation, src):
    """Differentiable PyTorch reference for the l=1 local rotation."""
    row_index = torch.tensor([0, 2, 1, 3], dtype=torch.int64, device=x.device)
    projected = rotation.index_select(1, row_index)
    return torch.matmul(projected, x.index_select(0, src))


__all__ = [
    "TileLangRotateToLocalL1",
    "rotate_to_local_l1_reference",
    "tilelang_rotate_to_local_l1",
]
