"""Standalone TileLang pointwise grid product used by the DPA4 S2 FFN."""

from functools import lru_cache

import torch

_BLOCK = 128


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang grid product requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def grid_product_forward(n_elements, block, dtype=T.float32):
        @T.prim_func
        def main(
            left: T.Tensor((n_elements,), dtype),
            right: T.Tensor((n_elements,), dtype),
            output: T.Tensor((n_elements,), dtype),
        ):
            with T.Kernel(T.ceildiv(n_elements, block), threads=128) as (pid,):
                for offset in T.Parallel(block):
                    index = pid * block + offset
                    if index < n_elements:
                        output[index] = left[index] * right[index]

        return main

    return grid_product_forward


@lru_cache(maxsize=1)
def _backward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def grid_product_backward(n_elements, block, dtype=T.float32):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_elements,), dtype),
            left: T.Tensor((n_elements,), dtype),
            right: T.Tensor((n_elements,), dtype),
            grad_left: T.Tensor((n_elements,), dtype),
            grad_right: T.Tensor((n_elements,), dtype),
        ):
            with T.Kernel(T.ceildiv(n_elements, block), threads=128) as (pid,):
                for offset in T.Parallel(block):
                    index = pid * block + offset
                    if index < n_elements:
                        grad_left[index] = grad_output[index] * right[index]
                        grad_right[index] = grad_output[index] * left[index]

        return main

    return grid_product_backward


def _check_inputs(left, right):
    if left.shape != right.shape:
        raise ValueError("left and right grid tensors must have the same shape")
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise TypeError("TileLang grid product requires float32 tensors")
    if left.device != right.device:
        raise ValueError("left and right must be on the same device")
    if not left.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang grid product requires a HIP/DCU tensor")


def _forward(left, right):
    _check_inputs(left, right)
    output = torch.empty_like(left)
    if left.numel():
        _, T = _tilelang_modules()
        kernel = _forward_factory()(left.numel(), _BLOCK, T.float32)
        kernel(
            left.contiguous().reshape(-1),
            right.contiguous().reshape(-1),
            output.reshape(-1),
        )
    return output


def _backward(grad_output, left, right):
    grad_left = torch.empty_like(left)
    grad_right = torch.empty_like(right)
    if left.numel():
        _, T = _tilelang_modules()
        kernel = _backward_factory()(left.numel(), _BLOCK, T.float32)
        kernel(
            grad_output.contiguous().reshape(-1),
            left.contiguous().reshape(-1),
            right.contiguous().reshape(-1),
            grad_left.reshape(-1),
            grad_right.reshape(-1),
        )
    return grad_left, grad_right


class TileLangGridProduct(torch.autograd.Function):
    """Autograd wrapper for a pointwise product of projected grid fields."""

    @staticmethod
    def forward(ctx, left, right):
        _check_inputs(left, right)
        left, right = left.contiguous(), right.contiguous()
        ctx.save_for_backward(left, right)
        return _forward(left, right)

    @staticmethod
    def backward(ctx, grad_output):
        left, right = ctx.saved_tensors
        if not any(ctx.needs_input_grad):
            return None, None
        return _backward(grad_output, left, right)


def tilelang_grid_product(left, right):
    """Compute the S2 grid pointwise quadratic product with TileLang."""
    return TileLangGridProduct.apply(left, right)


def grid_product_reference(left, right):
    """Differentiable PyTorch reference for the pointwise grid product."""
    return left * right


__all__ = ["TileLangGridProduct", "grid_product_reference", "tilelang_grid_product"]
