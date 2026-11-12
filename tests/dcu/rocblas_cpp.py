"""Lazy test-only PyTorch C++ extension for direct rocBLAS SGEMM."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch


class RocblasError(RuntimeError):
    """Direct rocBLAS extension loading or execution failed."""


def _library_arguments(library_path: Optional[str]):
    if library_path is None:
        return ["-lrocblas"], []

    resolved = Path(library_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"rocBLAS library does not exist: {resolved}")
    link_flags = [str(resolved), f"-Wl,-rpath,{resolved.parent}"]
    include_path = resolved.parent.parent / "include"
    include_dirs = [str(include_path)] if include_path.is_dir() else []
    return link_flags, include_dirs


@lru_cache(maxsize=4)
def _load_extension(library_path: Optional[str]):
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RocblasError(
            "direct rocBLAS requires HIP-enabled PyTorch and a visible DCU"
        )

    try:
        from torch.utils.cpp_extension import load
    except ImportError as error:
        raise RocblasError(
            "loading the C++ rocBLAS wrapper requires torch.utils.cpp_extension"
        ) from error

    source = Path(__file__).with_name("rocblas_sgemm.cpp")
    if not source.is_file():
        raise RocblasError(f"C++ rocBLAS source is missing: {source}")

    compiler = (
        os.environ.get("DPA4_HIP_CXX")
        or os.environ.get("HIPCXX")
        or "hipcc"
    )
    link_flags, include_dirs = _library_arguments(library_path)
    cache_key = "system" if library_path is None else str(Path(library_path).expanduser().resolve())
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:10]
    extension_name = f"dpa4_rocblas_sgemm_{digest}"

    previous_cxx = os.environ.get("CXX")
    os.environ["CXX"] = compiler
    try:
        return load(
            name=extension_name,
            sources=[str(source)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_ldflags=link_flags,
            extra_include_paths=include_dirs,
            with_cuda=True,
            verbose=os.environ.get("DPA4_ROCBLAS_CPP_VERBOSE", "0") == "1",
        )
    except Exception as error:
        raise RocblasError(
            "failed to build the C++ rocBLAS wrapper; source the DTK/ROCm "
            "environment and ensure hipcc, rocBLAS headers, and libraries are "
            f"available ({type(error).__name__}: {error})"
        ) from error
    finally:
        if previous_cxx is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = previous_cxx


class RocblasSgemm:
    """Call a C++ ``rocblas_sgemm`` wrapper on the current PyTorch HIP stream."""

    def __init__(self, library_path: Optional[str] = None):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RocblasError(
                "direct rocBLAS requires HIP-enabled PyTorch and a visible DCU"
            )
        self.library_path = (
            str(Path(library_path).expanduser().resolve())
            if library_path is not None
            else "librocblas from DTK/ROCm environment"
        )
        self._module = _load_extension(
            None if library_path is None else str(Path(library_path).expanduser().resolve())
        )
        self._handle = self._module.RocblasSgemm()
        self._closed = False

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RocblasError("rocBLAS handle is closed")
        return self._handle(a, b)

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> "RocblasSgemm":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["RocblasError", "RocblasSgemm"]
