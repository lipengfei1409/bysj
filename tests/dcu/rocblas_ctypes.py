"""Test-only ctypes adapter for direct rocBLAS single-precision GEMM.

rocBLAS uses column-major storage. For row-major inputs ``A[M,K]`` and
``B[K,N]``, this calls ``C_col[N,M] = B_col[N,K] * A_col[K,M]`` using the
same storage, then returns C as a row-major PyTorch tensor.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Optional

import torch


_ROCBLAS_STATUS_SUCCESS = 0
_ROCBLAS_OPERATION_NONE = 111  # rocblas_operation_none from rocblas_types.h
_INT32_MAX = (1 << 31) - 1


class RocblasError(RuntimeError):
    """Direct rocBLAS loading or execution failed."""


def _load_library(library_path: Optional[str]):
    if library_path:
        candidates = [library_path]
    else:
        candidates = []
        discovered = ctypes.util.find_library("rocblas")
        if discovered:
            candidates.append(discovered)
        candidates.extend(("librocblas.so", "rocblas"))

    failures = []
    for candidate in candidates:
        try:
            return candidate, ctypes.CDLL(candidate)
        except OSError as error:
            failures.append(f"{candidate}: {error}")
    details = "; ".join(failures) if failures else "no library candidate found"
    raise RocblasError(
        "could not load rocBLAS; source the DTK/ROCm environment or pass "
        f"--rocblas-library /path/to/librocblas.so ({details})"
    )


class RocblasSgemm:
    """Call the direct ``rocblas_sgemm`` API on PyTorch's current HIP stream."""

    def __init__(self, library_path: Optional[str] = None):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RocblasError("direct rocBLAS requires HIP-enabled PyTorch and a visible DCU")
        self.library_path, self._library = _load_library(library_path)
        self._bind_api()
        self._handle = ctypes.c_void_p()
        self._check(self._create_handle(ctypes.byref(self._handle)), "rocblas_create_handle")
        self._stream_id = None
        self._device = None
        self._closed = False

    def _bind_api(self) -> None:
        try:
            self._create_handle = self._library.rocblas_create_handle
            self._destroy_handle = self._library.rocblas_destroy_handle
            self._set_stream = self._library.rocblas_set_stream
            self._sgemm = self._library.rocblas_sgemm
        except AttributeError as error:
            raise RocblasError(
                f"{self.library_path} does not expose the required rocBLAS C API"
            ) from error

        self._create_handle.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._create_handle.restype = ctypes.c_int
        self._destroy_handle.argtypes = [ctypes.c_void_p]
        self._destroy_handle.restype = ctypes.c_int
        self._set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._set_stream.restype = ctypes.c_int
        self._sgemm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._sgemm.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != _ROCBLAS_STATUS_SUCCESS:
            raise RocblasError(f"{operation} failed with rocBLAS status {status}")

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RocblasError("rocBLAS handle is closed")
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError("rocBLAS SGEMM inputs must be torch tensors")
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("expected A[M,K] and B[K,N] with matching inner dimensions")
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise TypeError("direct rocblas_sgemm requires float32 tensors")
        if a.device != b.device or a.device.type != "cuda":
            raise ValueError("A and B must be on the same HIP/DCU device")
        if torch.version.hip is None:
            raise RocblasError("direct rocBLAS requires HIP-enabled PyTorch")
        if not a.is_contiguous() or not b.is_contiguous():
            raise ValueError("direct rocBLAS requires contiguous inputs")

        m, k = (int(value) for value in a.shape)
        _, n = (int(value) for value in b.shape)
        if max(m, n, k) > _INT32_MAX:
            raise ValueError("rocblas_sgemm uses 32-bit dimensions; dimensions are too large")

        output = torch.empty((m, n), device=a.device, dtype=a.dtype)
        if m == 0 or n == 0:
            return output

        stream_id = int(torch.cuda.current_stream(a.device).cuda_stream)
        if stream_id != self._stream_id:
            stream = ctypes.c_void_p(stream_id) if stream_id else ctypes.c_void_p()
            self._check(self._set_stream(self._handle, stream), "rocblas_set_stream")
            self._stream_id = stream_id
            self._device = a.device

        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        # Row-major A(M,K), B(K,N), C(M,N) are column-major Aᵀ(K,M),
        # Bᵀ(N,K), Cᵀ(N,M). Compute Cᵀ = Bᵀ * Aᵀ.
        self._check(
            self._sgemm(
                self._handle,
                _ROCBLAS_OPERATION_NONE,
                _ROCBLAS_OPERATION_NONE,
                n,
                m,
                k,
                ctypes.byref(alpha),
                ctypes.c_void_p(int(b.data_ptr())),
                n,
                ctypes.c_void_p(int(a.data_ptr())),
                k,
                ctypes.byref(beta),
                ctypes.c_void_p(int(output.data_ptr())),
                n,
            ),
            "rocblas_sgemm",
        )
        return output

    def close(self) -> None:
        if not self._closed:
            if self._device is not None:
                torch.cuda.synchronize(self._device)
            self._check(self._destroy_handle(self._handle), "rocblas_destroy_handle")
            self._closed = True

    def __enter__(self) -> "RocblasSgemm":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["RocblasError", "RocblasSgemm"]
