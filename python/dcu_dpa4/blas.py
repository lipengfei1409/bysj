"""Row-major FP32/FP64 GEMM. rocBLAS is explicit, never a silent fallback."""
from __future__ import annotations

import math
import numbers
import struct

import torch


def _validate(a, b, c, trans_a, trans_b, alpha, beta, backend):
    if backend not in ('reference', 'rocblas'):
        raise ValueError('BLAS backend must be reference or rocblas')
    if type(trans_a) is not bool or type(trans_b) is not bool:
        raise TypeError('trans_a and trans_b must be bool')
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError('A and B must be tensors')
    if a.ndim not in (2, 3) or b.ndim != a.ndim:
        raise ValueError('GEMM requires two 2D or two 3D tensors')
    if a.dtype not in (torch.float32, torch.float64) or b.dtype != a.dtype:
        raise TypeError('GEMM requires matching float32 or float64 tensors')
    if b.device != a.device:
        raise ValueError('GEMM tensors must share a device')
    if torch.is_autocast_enabled(a.device.type):
        raise RuntimeError('BLAS interface requires autocast disabled')
    if a.ndim == 3 and a.shape[0] != b.shape[0]:
        raise ValueError('GEMM batch sizes must match; no batch broadcasting')
    m, k = (a.shape[-1], a.shape[-2]) if trans_a else a.shape[-2:]
    kb, n = (b.shape[-1], b.shape[-2]) if trans_b else b.shape[-2:]
    if k != kb:
        raise ValueError('GEMM inner dimensions disagree')
    shape = (*a.shape[:-2], m, n)
    if c is not None:
        if not isinstance(c, torch.Tensor) or c.dtype != a.dtype:
            raise TypeError('C must have the same dtype as A/B')
        if c.shape != shape or c.device != a.device:
            raise ValueError('C must match GEMM output shape/device exactly')
    for name, value in (('alpha', alpha), ('beta', beta)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(name + ' must be a real scalar')
        if not math.isfinite(float(value)) or abs(float(value)) > torch.finfo(a.dtype).max:
            raise ValueError(name + ' must be finite and within input dtype range')
    if c is None and beta != 0:
        raise ValueError('nonzero beta requires C')
    return shape


def _rocblas_extension():
    from .native import _extension
    extension = _extension()
    if not hasattr(extension, 'blas_available') or not extension.blas_available():
        raise RuntimeError('rebuild with DPA4_BUILD_HIP=1 DPA4_BUILD_ROCBLAS=1')
    return extension


class _Gemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, ta, tb, alpha, beta, backend):
        ctx.save_for_backward(a, b)
        ctx.ta, ctx.tb, ctx.alpha, ctx.beta, ctx.has_c = ta, tb, alpha, beta, c is not None
        if backend == 'rocblas':
            extension = _rocblas_extension()
            return extension.blas_gemm(a.contiguous(), b.contiguous(),
                                      c.contiguous() if c is not None else a.new_empty(0),
                                      c is not None, ta, tb, alpha, beta)
        ao = a.transpose(-1, -2) if ta else a
        bo = b.transpose(-1, -2) if tb else b
        if alpha == 0 or ao.shape[-1] == 0:
            result = a.new_zeros((*ao.shape[:-2], ao.shape[-2], bo.shape[-1]))
        else:
            result = alpha * (ao @ bo)
        if c is not None and beta != 0:
            result = result + beta * c
        return result

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        ao = a.transpose(-1, -2) if ctx.ta else a
        bo = b.transpose(-1, -2) if ctx.tb else b
        ga = gb = gc = None
        if ctx.needs_input_grad[0]:
            if ctx.alpha == 0:
                ga = torch.zeros_like(a)
            else:
                ga = ctx.alpha * (g @ bo.transpose(-1, -2))
                if ctx.ta:
                    ga = ga.transpose(-1, -2)
        if ctx.needs_input_grad[1]:
            if ctx.alpha == 0:
                gb = torch.zeros_like(b)
            else:
                gb = ctx.alpha * (ao.transpose(-1, -2) @ g)
                if ctx.tb:
                    gb = gb.transpose(-1, -2)
        if ctx.has_c and ctx.needs_input_grad[2]:
            gc = torch.zeros_like(g) if ctx.beta == 0 else ctx.beta * g
        return ga, gb, gc, None, None, None, None, None


def gemm(a, b, c=None, *, trans_a=False, trans_b=False,
         alpha=1.0, beta=0.0, backend='reference'):
    """Return alpha*op(A)@op(B)+beta*C, without modifying any input.

    2D shapes: op(A)[M,K], op(B)[K,N], C/output[M,N].
    3D shapes: [B,M,K], [B,K,N], [B,M,N]; no batch broadcasting.
    Native forward is rocBLAS; backward uses Torch matmul, not custom BLAS VJP.
    """
    _validate(a, b, c, trans_a, trans_b, alpha, beta, backend)
    alpha, beta = float(alpha), float(beta)
    if a.dtype == torch.float32:
        # Match the FP32 pointers passed to SGEMM, including scalar underflow.
        alpha = struct.unpack('f', struct.pack('f', alpha))[0]
        beta = struct.unpack('f', struct.pack('f', beta))[0]
    return _Gemm.apply(a, b, c, trans_a, trans_b, alpha, beta, backend)


def linear(x, weight, bias=None, *, backend='reference'):
    """x[N,K] @ weight[K,C] + bias[C]; bias epilogue is a Torch operation."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError('linear requires 2D x and weight')
    if bias is not None and (bias.shape != (weight.shape[1],) or
                             bias.dtype != x.dtype or bias.device != x.device):
        raise ValueError('linear bias must be [C], matching dtype/device')
    result = gemm(x, weight, backend=backend)
    return result if bias is None else result + bias
