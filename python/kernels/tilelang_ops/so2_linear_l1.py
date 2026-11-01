"""Standalone TileLang block-diagonal SO(2) linear for the lmax=mmax=1 path."""

from functools import lru_cache

import torch

_BLOCK_EDGES = 4
_BLOCK_WIDTH = 32


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang SO2Linear requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def block_diag_forward(
        n_focus, n_edges, in_dim, out_dim, block_edges, block_width, dtype=T.float32
    ):
        @T.prim_func
        def main(
            x: T.Tensor((n_focus, n_edges, in_dim), dtype),
            weight: T.Tensor((in_dim, n_focus, out_dim), dtype),
            output: T.Tensor((n_focus, n_edges, out_dim), dtype),
        ):
            in_block = in_dim // 2
            out_block = out_dim // 2
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                n_focus,
                T.ceildiv(out_dim, block_width),
                threads=128,
            ) as (pid_edge, focus, pid_out):
                for edge_offset, out_offset in T.Parallel(block_edges, block_width):
                    edge = pid_edge * block_edges + edge_offset
                    out_index = pid_out * block_width + out_offset
                    if edge < n_edges and out_index < out_dim:
                        block = out_index // out_block
                        in_start = block * in_block
                        acc = 0.0
                        for local_in in T.serial(in_block):
                            in_index = in_start + local_in
                            acc += (
                                x[focus, edge, in_index]
                                * weight[in_index, focus, out_index]
                            )
                        output[focus, edge, out_index] = acc

        return main

    return block_diag_forward


@lru_cache(maxsize=1)
def _backward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def block_diag_backward(
        n_focus, n_edges, in_dim, out_dim, block_edges, block_width, dtype=T.float32
    ):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_focus, n_edges, out_dim), dtype),
            weight: T.Tensor((in_dim, n_focus, out_dim), dtype),
            grad_x: T.Tensor((n_focus, n_edges, in_dim), dtype),
        ):
            in_block = in_dim // 2
            out_block = out_dim // 2
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                n_focus,
                T.ceildiv(in_dim, block_width),
                threads=128,
            ) as (pid_edge, focus, pid_in):
                for edge_offset, in_offset in T.Parallel(block_edges, block_width):
                    edge = pid_edge * block_edges + edge_offset
                    in_index = pid_in * block_width + in_offset
                    if edge < n_edges and in_index < in_dim:
                        block = in_index // in_block
                        out_start = block * out_block
                        acc = 0.0
                        for local_out in T.serial(out_block):
                            out_index = out_start + local_out
                            acc += (
                                grad_output[focus, edge, out_index]
                                * weight[in_index, focus, out_index]
                            )
                        grad_x[focus, edge, in_index] = acc

        return main

    return block_diag_backward


def _check_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.ndim != 3 or weight.ndim != 3:
        raise ValueError("x and weight must have rank 3")
    if weight.shape[0] != x.shape[2] or weight.shape[1] != x.shape[0]:
        raise ValueError("weight shape must be (input_dim, n_focus, output_dim)")
    if x.shape[2] % 2 or weight.shape[2] % 2:
        raise ValueError("lmax=mmax=1 inputs must have two equal SO(2) blocks")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("TileLang SO2Linear requires float32 inputs")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if not x.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang SO2Linear requires a HIP/DCU tensor")


def _forward(x, weight):
    _check_inputs(x, weight)
    n_focus, n_edges, in_dim = x.shape
    out_dim = weight.shape[2]
    output = torch.empty((n_focus, n_edges, out_dim), dtype=x.dtype, device=x.device)
    if n_edges:
        _, T = _tilelang_modules()
        kernel = _forward_factory()(
            n_focus, n_edges, in_dim, out_dim, _BLOCK_EDGES, _BLOCK_WIDTH, T.float32
        )
        kernel(x.contiguous(), weight.contiguous(), output)
    return output


def _backward(grad_output, x, weight):
    grad_x = torch.empty_like(x)
    if x.shape[1]:
        _, T = _tilelang_modules()
        kernel = _backward_factory()(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            weight.shape[2],
            _BLOCK_EDGES,
            _BLOCK_WIDTH,
            T.float32,
        )
        kernel(grad_output.contiguous(), weight.contiguous(), grad_x)
    return grad_x


class TileLangSO2LinearL1(torch.autograd.Function):
    """Autograd wrapper; inference force gradients flow to x, not frozen weights."""

    @staticmethod
    def forward(ctx, x, weight):
        _check_inputs(x, weight)
        x = x.contiguous()
        weight = weight.contiguous()
        ctx.save_for_backward(x, weight)
        return _forward(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        return _backward(grad_output, x, weight), None


def tilelang_so2_linear_l1(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply only the two diagonal |m| groups in the DPA4 lmax=1 layout."""
    return TileLangSO2LinearL1.apply(x, weight)


def so2_linear_l1_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """PyTorch reference that contracts only the two diagonal |m| groups."""
    if x.ndim != 3 or weight.ndim != 3:
        raise ValueError("x and weight must have rank 3")
    in_block = x.shape[-1] // 2
    out_block = weight.shape[-1] // 2
    outputs = []
    for block in range(2):
        in_start = block * in_block
        out_start = block * out_block
        outputs.append(
            torch.matmul(
                x[..., in_start : in_start + in_block],
                weight[
                    in_start : in_start + in_block, :, out_start : out_start + out_block
                ].permute(1, 0, 2),
            )
        )
    return torch.cat(outputs, dim=-1)


__all__ = ["TileLangSO2LinearL1", "so2_linear_l1_reference", "tilelang_so2_linear_l1"]
