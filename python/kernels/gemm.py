"""DCU-only FP32 GEMM backed by the DUMMA 16x16x4 tile."""
from __future__ import annotations

import torch


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``A[M, K] @ B[K, N]`` for contiguous 2D FP32 DCU tensors.

    The initial DUMMA operator is forward-only. Use ``python.reference.gemm``
    for a PyTorch oracle; the production API never falls back to it.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A and B must be tensors")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("DUMMA GEMM requires float32 tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("DUMMA GEMM requires two 2D tensors A[M,K] and B[K,N]")
    if a.shape[1] != b.shape[0]:
        raise ValueError("DUMMA GEMM inner dimensions must match")
    if a.device != b.device:
        raise ValueError("DUMMA GEMM tensors must share a device")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("DUMMA GEMM requires contiguous tensors")
    if a.requires_grad or b.requires_grad:
        raise RuntimeError("the initial DUMMA GEMM does not support autograd")

    from .native import _extension, require_dcu
    require_dcu(a, b)
    return _extension().gemm(a, b)
