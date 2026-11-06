"""PyTorch oracle for the initial contiguous 2D FP32 GEMM contract."""
from __future__ import annotations

import torch


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``A[M, K] @ B[K, N]`` with PyTorch on the input device."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A and B must be tensors")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("reference GEMM requires float32 tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("reference GEMM requires two 2D tensors A[M,K] and B[K,N]")
    if a.shape[1] != b.shape[0]:
        raise ValueError("reference GEMM inner dimensions must match")
    if a.device != b.device:
        raise ValueError("reference GEMM tensors must share a device")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("reference GEMM requires contiguous tensors")
    return torch.matmul(a, b)
