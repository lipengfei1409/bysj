"""Standalone TileLang geometric-initial-embedding scatter operator."""

from functools import lru_cache

import torch

_BLOCK_EDGES = 8


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang zonal scatter requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def zonal_scatter_forward(
        n_edges, n_rows, n_radial, channels, n_nodes, block_edges, dtype=T.float32
    ):
        @T.prim_func
        def main(
            zonal: T.Tensor((n_edges, n_rows), dtype),
            radial: T.Tensor((n_edges, n_radial, channels), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            radial_slot: T.Tensor((n_rows,), T.int64),
            inv_sqrt_deg: T.Tensor((n_nodes,), dtype),
            output: T.Tensor((n_nodes, n_rows + 1, channels), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block_edges), threads=128) as (pid,):
                for edge_offset, row, channel in T.Parallel(
                    block_edges, n_rows, channels
                ):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges:
                        node = dst[edge]
                        radial_index = radial_slot[row]
                        value = (
                            zonal[edge, row]
                            * radial[edge, radial_index, channel]
                            * inv_sqrt_deg[node]
                        )
                        T.atomic_add(output[node, row + 1, channel], value)

        return main

    return zonal_scatter_forward


@lru_cache(maxsize=1)
def _backward_zonal_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def zonal_scatter_backward_zonal(
        n_edges, n_rows, n_radial, channels, n_nodes, block_edges, dtype=T.float32
    ):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_nodes, n_rows + 1, channels), dtype),
            radial: T.Tensor((n_edges, n_radial, channels), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            radial_slot: T.Tensor((n_rows,), T.int64),
            inv_sqrt_deg: T.Tensor((n_nodes,), dtype),
            zonal: T.Tensor((n_edges, n_rows), dtype),
            grad_zonal: T.Tensor((n_edges, n_rows), dtype),
            grad_inv_sqrt_deg: T.Tensor((n_nodes,), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block_edges), threads=128) as (pid,):
                for edge_offset, row in T.Parallel(block_edges, n_rows):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges:
                        node = dst[edge]
                        radial_index = radial_slot[row]
                        inverse = inv_sqrt_deg[node]
                        grad_value = 0.0
                        grad_inverse = 0.0
                        for channel in T.serial(channels):
                            upstream = grad_output[node, row + 1, channel]
                            radial_value = radial[edge, radial_index, channel]
                            grad_value += upstream * radial_value * inverse
                            grad_inverse += upstream * zonal[edge, row] * radial_value
                        grad_zonal[edge, row] = grad_value
                        T.atomic_add(grad_inv_sqrt_deg[node], grad_inverse)

        return main

    return zonal_scatter_backward_zonal


@lru_cache(maxsize=1)
def _backward_radial_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def zonal_scatter_backward_radial(
        n_edges, n_rows, n_radial, channels, n_nodes, block_edges, dtype=T.float32
    ):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_nodes, n_rows + 1, channels), dtype),
            zonal: T.Tensor((n_edges, n_rows), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            radial_slot: T.Tensor((n_rows,), T.int64),
            inv_sqrt_deg: T.Tensor((n_nodes,), dtype),
            grad_radial: T.Tensor((n_edges, n_radial, channels), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges), T.ceildiv(channels, 32), threads=128
            ) as (pid_edge, pid_channel):
                for edge_offset, radial_index, channel_offset in T.Parallel(
                    block_edges, n_radial, 32
                ):
                    edge = pid_edge * block_edges + edge_offset
                    channel = pid_channel * 32 + channel_offset
                    if edge < n_edges and channel < channels:
                        node = dst[edge]
                        inverse = inv_sqrt_deg[node]
                        grad_value = 0.0
                        for row in T.serial(n_rows):
                            if radial_slot[row] == radial_index:
                                grad_value += (
                                    grad_output[node, row + 1, channel]
                                    * zonal[edge, row]
                                    * inverse
                                )
                        grad_radial[edge, radial_index, channel] = grad_value

        return main

    return zonal_scatter_backward_radial


def _check_inputs(zonal, radial, dst, radial_slot, inv_sqrt_deg):
    if zonal.ndim != 2:
        raise ValueError("zonal must have shape (n_edges, n_rows)")
    if radial.ndim != 3 or radial.shape[0] != zonal.shape[0]:
        raise ValueError("radial must have shape (n_edges, n_radial, channels)")
    if dst.ndim != 1 or dst.shape[0] != zonal.shape[0]:
        raise ValueError("dst must have shape (n_edges,)")
    if radial_slot.ndim != 1 or radial_slot.shape[0] != zonal.shape[1]:
        raise ValueError("radial_slot must have shape (n_rows,)")
    if inv_sqrt_deg.ndim not in (1, 3):
        raise ValueError("inv_sqrt_deg must have shape (n_nodes,) or (n_nodes, 1, 1)")
    if inv_sqrt_deg.ndim == 3 and inv_sqrt_deg.shape[1:] != (1, 1):
        raise ValueError("inv_sqrt_deg must have shape (n_nodes, 1, 1)")
    if any(t.dtype != torch.float32 for t in (zonal, radial, inv_sqrt_deg)):
        raise TypeError("TileLang zonal scatter requires float32 value tensors")
    if dst.dtype != torch.int64 or radial_slot.dtype != torch.int64:
        raise TypeError("dst and radial_slot must use torch.int64")
    if (
        len(
            {
                zonal.device,
                radial.device,
                dst.device,
                radial_slot.device,
                inv_sqrt_deg.device,
            }
        )
        != 1
    ):
        raise ValueError("all zonal scatter inputs must be on the same device")
    if not zonal.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang zonal scatter requires a HIP/DCU tensor")


def _flatten_inv_sqrt(inv_sqrt_deg: torch.Tensor) -> torch.Tensor:
    return inv_sqrt_deg.reshape(-1).contiguous()


def _forward(zonal, radial, dst, radial_slot, inv_sqrt_deg):
    _check_inputs(zonal, radial, dst, radial_slot, inv_sqrt_deg)
    n_edges, n_rows = zonal.shape
    n_nodes = inv_sqrt_deg.shape[0]
    channels = radial.shape[2]
    output = torch.zeros(
        (n_nodes, n_rows + 1, channels), dtype=zonal.dtype, device=zonal.device
    )
    if n_edges:
        _, T = _tilelang_modules()
        kernel = _forward_factory()(
            n_edges,
            n_rows,
            radial.shape[1],
            channels,
            n_nodes,
            _BLOCK_EDGES,
            T.float32,
        )
        kernel(
            zonal.contiguous(),
            radial.contiguous(),
            dst.contiguous(),
            radial_slot.contiguous(),
            _flatten_inv_sqrt(inv_sqrt_deg),
            output,
        )
    return output


def _backward(grad_output, zonal, radial, dst, radial_slot, inv_sqrt_deg):
    n_edges, n_rows = zonal.shape
    n_nodes = inv_sqrt_deg.shape[0]
    n_radial = radial.shape[1]
    channels = radial.shape[2]
    grad_zonal = torch.empty_like(zonal)
    grad_radial = torch.empty_like(radial)
    grad_inv_sqrt_deg = torch.zeros_like(_flatten_inv_sqrt(inv_sqrt_deg))
    if n_edges:
        _, T = _tilelang_modules()
        args = (
            n_edges,
            n_rows,
            n_radial,
            channels,
            n_nodes,
            _BLOCK_EDGES,
            T.float32,
        )
        common = (
            grad_output.contiguous(),
            radial.contiguous(),
            dst.contiguous(),
            radial_slot.contiguous(),
            _flatten_inv_sqrt(inv_sqrt_deg),
            zonal.contiguous(),
        )
        _backward_zonal_factory()(*args)(*common, grad_zonal, grad_inv_sqrt_deg)
        _backward_radial_factory()(*args)(
            grad_output.contiguous(),
            zonal.contiguous(),
            dst.contiguous(),
            radial_slot.contiguous(),
            _flatten_inv_sqrt(inv_sqrt_deg),
            grad_radial,
        )
    return grad_zonal, grad_radial, grad_inv_sqrt_deg.reshape(inv_sqrt_deg.shape)


class TileLangZonalScatter(torch.autograd.Function):
    """Scatter radial-weighted zonal messages and normalize by node degree."""

    @staticmethod
    def forward(ctx, zonal, radial, dst, radial_slot, inv_sqrt_deg):
        _check_inputs(zonal, radial, dst, radial_slot, inv_sqrt_deg)
        zonal = zonal.contiguous()
        radial = radial.contiguous()
        dst = dst.contiguous()
        radial_slot = radial_slot.contiguous()
        inv_sqrt_deg = inv_sqrt_deg.contiguous()
        ctx.save_for_backward(zonal, radial, dst, radial_slot, inv_sqrt_deg)
        return _forward(zonal, radial, dst, radial_slot, inv_sqrt_deg)

    @staticmethod
    def backward(ctx, grad_output):
        zonal, radial, dst, radial_slot, inv_sqrt_deg = ctx.saved_tensors
        if not any(ctx.needs_input_grad[:2]) and not ctx.needs_input_grad[4]:
            return None, None, None, None, None
        grad_zonal, grad_radial, grad_inv_sqrt_deg = _backward(
            grad_output, zonal, radial, dst, radial_slot, inv_sqrt_deg
        )
        return (
            grad_zonal if ctx.needs_input_grad[0] else None,
            grad_radial if ctx.needs_input_grad[1] else None,
            None,
            None,
            grad_inv_sqrt_deg if ctx.needs_input_grad[4] else None,
        )


def tilelang_zonal_scatter(zonal, radial, dst, radial_slot, inv_sqrt_deg):
    """Build and reduce the DPA4 geometric initial-embedding messages."""
    return TileLangZonalScatter.apply(zonal, radial, dst, radial_slot, inv_sqrt_deg)


def zonal_scatter_reference(zonal, radial, dst, radial_slot, inv_sqrt_deg):
    """Differentiable PyTorch reference for the non-fused GIE scatter."""
    if inv_sqrt_deg.ndim == 3:
        inv_sqrt_deg = inv_sqrt_deg.reshape(-1)
    n_edges, n_rows = zonal.shape
    n_nodes = inv_sqrt_deg.shape[0]
    channels = radial.shape[-1]
    radial_index = radial_slot[None, :, None].expand(n_edges, n_rows, channels)
    radial_values = radial.gather(1, radial_index)
    messages = zonal[:, :, None] * radial_values
    output = radial.new_zeros((n_nodes, n_rows + 1, channels))
    if n_edges:
        output[:, 1:, :].index_add_(
            0,
            dst,
            messages * inv_sqrt_deg.index_select(0, dst)[:, None, None],
        )
    return output


__all__ = [
    "TileLangZonalScatter",
    "tilelang_zonal_scatter",
    "zonal_scatter_reference",
]
