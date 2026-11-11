"""Test-local dense GEMM kernel compiled explicitly for the HIP backend."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch


_BLOCK_M = 32
_BLOCK_N = 32
_BLOCK_K = 32
_THREADS = 64


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
        from tilelang.tileop.base import GemmWarpPolicy
    except Exception as error:
        raise RuntimeError(
            "TileLang dense GEMM requires a TileLang installation with HIP support"
        ) from error
    return tilelang, T, GemmWarpPolicy


def _tilelang_dtype(dtype_name: str, T: Any):
    if dtype_name == "float16":
        return T.float16
    if dtype_name == "bfloat16":
        tilelang_bfloat16 = getattr(T, "bfloat16", None)
        if tilelang_bfloat16 is None:
            raise RuntimeError("this TileLang build does not expose bfloat16")
        return tilelang_bfloat16
    raise ValueError(f"unsupported dense GEMM dtype: {dtype_name}")


@lru_cache(maxsize=32)
def dense_gemm_kernel(
    m: int,
    n: int,
    k: int,
    dtype_name: str = "float16",
    block_m: int = _BLOCK_M,
    block_n: int = _BLOCK_N,
    block_k: int = _BLOCK_K,
    threads: int = _THREADS,
):
    """Build/cache a HIP TileLang kernel returning ``A[M,K] @ B[K,N]``."""
    tilelang, T, GemmWarpPolicy = _tilelang_modules()
    dtype = _tilelang_dtype(dtype_name, T)

    @tilelang.jit(target="hip", out_idx=[2])
    def dense_gemm(M, N, K, bm, bn, bk, num_threads, dtype=dtype):
        @T.prim_func
        def main(
            A: T.Tensor([M, K], dtype),
            B: T.Tensor([K, N], dtype),
            C: T.Tensor([M, N], dtype),
        ):
            with T.Kernel(
                T.ceildiv(M, bm), T.ceildiv(N, bn), threads=num_threads
            ) as (pid_m, pid_n):
                a_shared = T.alloc_shared([bm, bk], dtype)
                b_shared = T.alloc_shared([bk, bn], dtype)
                accum = T.alloc_fragment([bm, bn], T.float32)
                T.clear(accum)

                for pid_k in T.serial(T.ceildiv(K, bk)):
                    k_offset = pid_k * bk
                    for i, j in T.Parallel(bm, bk):
                        row = pid_m * bm + i
                        col = k_offset + j
                        if row < M and col < K:
                            a_shared[i, j] = A[row, col]
                        else:
                            a_shared[i, j] = 0.0

                    for i, j in T.Parallel(bk, bn):
                        row = k_offset + i
                        col = pid_n * bn + j
                        if row < K and col < N:
                            b_shared[i, j] = B[row, col]
                        else:
                            b_shared[i, j] = 0.0

                    T.gemm(
                        a_shared,
                        b_shared,
                        accum,
                        k_pack=1,
                        policy=GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(bm, bn):
                    row = pid_m * bm + i
                    col = pid_n * bn + j
                    if row < M and col < N:
                        C[row, col] = accum[i, j]

        return main

    return dense_gemm(m, n, k, block_m, block_n, block_k, threads, dtype)


def _unwrap_output(output):
    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise RuntimeError(f"expected one GEMM output, got {len(output)}")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f"TileLang returned unexpected output type: {type(output)!r}")
    return output


def tilelang_dense_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run the test-local HIP GEMM kernel on device-resident matrices."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N] with matching inner dimensions")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("TileLang dense GEMM supports matching float16 or bfloat16 inputs")
    if a.device != b.device or a.device.type != "cuda":
        raise ValueError("A and B must be on the same HIP/DCU device")
    if torch.version.hip is None:
        raise RuntimeError("TileLang dense GEMM requires a HIP-enabled PyTorch build")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("TileLang dense GEMM requires contiguous input tensors")

    dtype_name = "float16" if a.dtype == torch.float16 else "bfloat16"
    kernel = dense_gemm_kernel(int(a.shape[0]), int(b.shape[1]), int(a.shape[1]), dtype_name)
    return _unwrap_output(kernel(a, b))


__all__ = ["dense_gemm_kernel", "tilelang_dense_gemm"]
