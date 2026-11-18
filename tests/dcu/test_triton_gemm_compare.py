"""Triton GEMM versus direct rocBLAS SGEMM across a wide shape sweep.

The DCU tuning plan needs one number the existing comparisons do not produce:
how far a Triton ``tl.dot`` GEMM is from the vendor library as a function of
shape.  That gap decides whether the DPA4 GEMMs are worth hand-writing (or
megakernel-fusing) or whether Triton is already close enough, and it is not a
single number -- Triton's tile heuristic and rocBLAS's Tensile kernel selection
both move with ``M``, ``N``, ``K`` and the arithmetic intensity, so a square
1024 only says something about square 1024.

What is measured
----------------
Three arms per shape, on the same tensors and the same stream:

``triton``
    the tiled GEMM in this file, optionally autotuned over a tile grid, with
    the ``input_precision`` the DPA4 kernels use (``ieee`` by default, because
    ``tl.dot(..., "ieee")`` is what the SeZM stack requests).
``rocblas``
    direct ``rocblas_sgemm`` through :mod:`rocblas_cpp`, i.e. the vendor
    library with no PyTorch dispatch in between.
``aten``
    ``torch.matmul`` / ``torch.bmm``, reported for reference only: it is the
    API the model actually calls, and its distance to ``rocblas`` is the
    PyTorch dispatch overhead.

For a batched shape the ``rocblas`` arm is unavailable (the test wrapper is a
single 2-D SGEMM), so the reference is ``torch.bmm`` labelled ``aten``; the
report says so per row and in the aggregate section.

Coverage
--------
``--kinds`` selects any of:

``square``
    ``N x N x N`` for the classic sizes, 128 .. 8192 (capped by memory).
``rect``
    aspect-ratio extremes, including the "tall-skinny" and "wide-short"
    shapes where GEMM is memory bound on the DCU.
``ktail``
    ``K`` at and around tile boundaries (1, 3, 31, 33, 100, 1000, 4097) so a
    kernel that assumes ``K % BLOCK_K == 0`` shows up as wrong rather than
    fast.
``dpa4``
    the shapes the DPA4/SeZM graph actually issues: ``(E, C) x (C, C)`` edge
    GEMMs and ``(N, C) x (C, C)`` node GEMMs for the edge counts and channel
    widths seen in the profiling trace.
``batch``
    ``bmm`` with the batch counts the SO(2) block GEMMs use.

Every timed shape is also checked against the reference arm, so a fast but
wrong tile configuration cannot win.

Usage
-----
::

    source /opt/dtk/env.sh
    python tests/dcu/test_triton_gemm_compare.py --kinds all \\
        --output-json /tmp/triton-vs-rocblas.json

Exit code is 0 when every shape ran and every ``ieee`` result agreed with its
reference; a numerical disagreement or an arm failure returns 1.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import unittest
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from rocblas_cpp import RocblasError, RocblasSgemm

try:  # Triton is optional: the host-side tests below must run without it.
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except Exception as error:  # noqa: BLE001 - any import failure means unusable
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = error

_HAS_HIP_DEVICE = torch.cuda.is_available() and torch.version.hip is not None
_TRITON_AVAILABLE = triton is not None

DEFAULT_PRECISION = "ieee"
_PRECISION_CODES = {"default": 0, "ieee": 1, "tf32": 2}
_DEFAULT_MEMORY_FRACTION = 0.80
_DEFAULT_TILE_CONFIGS = (
    (128, 128, 64, 8, 3),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
    (128, 128, 32, 4, 4),
    (64, 128, 64, 4, 3),
    (128, 64, 64, 4, 3),
    (64, 64, 32, 4, 3),
    (32, 64, 32, 4, 2),
)

# The shapes a Triton GEMM is expected to handle well, and the ones where the
# vendor library historically wins by a wide margin.  Kept explicit rather
# than generated so the report is comparable between runs.
_SQUARE_DIMS = (128, 256, 512, 1024, 2048, 4096, 8192)
_RECT_SHAPES = (
    (256, 1024, 128),
    (1024, 256, 128),
    (512, 2048, 256),
    (2048, 512, 256),
    (64, 4096, 64),
    (4096, 64, 64),
    (257, 511, 129),
    (1023, 2049, 513),
)
_KTAIL_M_N = (512, 512)
_KTAIL_DIMS = (1, 3, 31, 33, 100, 1000, 4097)
_DPA4_EDGES = (3000, 36000, 181838)
_DPA4_CHANNELS = (32, 64, 128, 256, 512)
_BATCH_SHAPES = (
    (8, 512, 512, 128),
    (64, 512, 512, 128),
    (512, 512, 512, 64),
    (64, 1024, 1024, 128),
    (512, 3000, 64, 64),
)


# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        GROUP_SIZE_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_K)):
            remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=offs_k[None, :] < remaining,
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < remaining,
                other=0.0,
            )
            if PRECISION == 1:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            elif PRECISION == 2:
                acc = tl.dot(a, b, acc, input_precision="tf32")
            else:
                acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

    @triton.jit
    def _bmm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_bk,
        stride_bb,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        GROUP_SIZE_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch = tl.program_id(axis=1)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        base_a = a_ptr + batch * stride_ab
        base_b = b_ptr + batch * stride_bb
        a_ptrs = base_a + (offs_am[:, None] * stride_am + offs_k[None, :])
        b_ptrs = base_b + (offs_k[:, None] * stride_bk + offs_bn[None, :])

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_K)):
            remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < remaining, other=0.0)
            if PRECISION == 1:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            elif PRECISION == 2:
                acc = tl.dot(a, b, acc, input_precision="tf32")
            else:
                acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = (
            c_ptr
            + batch * stride_cb
            + offs_cm[:, None] * stride_cm
            + offs_cn[None, :] * stride_cn
        )
        tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def _autotuned(kernel, configs: Iterable[tuple[int, int, int, int, int]]):
    """Wrap a kernel in an autotuner over ``(BM, BN, BK, warps, stages)``."""
    return triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                num_warps=warps,
                num_stages=stages,
            )
            for bm, bn, bk, warps, stages in configs
        ],
        key=["M", "N", "K"],
    )(kernel)


# ----------------------------------------------------------------------
# Launch glue
# ----------------------------------------------------------------------
def _gemm_grid(meta: dict[str, Any], size: tuple[int, int, int]) -> tuple[int]:
    m, n, _k = size
    return (math.ceil(m / meta["BLOCK_M"]) * math.ceil(n / meta["BLOCK_N"]),)


def _reject_unsupported_precision(error: TypeError, precision: str) -> None:
    """Turn a Triton version mismatch into an actionable error."""
    if "input_precision" in str(error) or "allow_tf32" in str(error):
        raise RuntimeError(
            f"this Triton version cannot request input_precision={precision!r}; "
            "re-run with --precision default"
        ) from error
    raise error


def launch_triton_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    precision: str = DEFAULT_PRECISION,
    config: tuple[int, int, int, int, int] | None = None,
    group_size_m: int = 8,
) -> torch.Tensor:
    """Return ``a @ b`` computed by the Triton kernel in this module."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR!r}")
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    code = _PRECISION_CODES[precision]
    try:
        if config is None:
            grid = lambda meta: _gemm_grid(meta, (m, n, k))
            _gemm_autotuned[grid](
                a,
                b,
                c,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                GROUP_SIZE_M=group_size_m,
                PRECISION=code,
            )
        else:
            bm, bn, bk, warps, stages = config
            _gemm_kernel[(math.ceil(m / bm) * math.ceil(n / bn),)](
                a,
                b,
                c,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                GROUP_SIZE_M=group_size_m,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                PRECISION=code,
                num_warps=warps,
                num_stages=stages,
            )
    except TypeError as error:
        _reject_unsupported_precision(error, precision)
    return c


def launch_triton_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    precision: str = DEFAULT_PRECISION,
    config: tuple[int, int, int, int, int] | None = None,
    group_size_m: int = 8,
) -> torch.Tensor:
    """Return the batched product of ``a[b] @ b[b]`` with the Triton kernel."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR!r}")
    batch, m, k = a.shape
    n = b.shape[2]
    c = torch.empty((batch, m, n), device=a.device, dtype=torch.float32)
    code = _PRECISION_CODES[precision]
    try:
        if config is None:
            grid = lambda meta: (
                math.ceil(m / meta["BLOCK_M"]) * math.ceil(n / meta["BLOCK_N"]),
                batch,
            )
            _bmm_autotuned[grid](
                a,
                b,
                c,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(1),
                b.stride(0),
                b.stride(2),
                c.stride(0),
                c.stride(1),
                c.stride(2),
                GROUP_SIZE_M=group_size_m,
                PRECISION=code,
            )
        else:
            bm, bn, bk, warps, stages = config
            _bmm_kernel[(math.ceil(m / bm) * math.ceil(n / bn), batch)](
                a,
                b,
                c,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(1),
                b.stride(0),
                b.stride(2),
                c.stride(0),
                c.stride(1),
                c.stride(2),
                GROUP_SIZE_M=group_size_m,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                PRECISION=code,
                num_warps=warps,
                num_stages=stages,
            )
    except TypeError as error:
        _reject_unsupported_precision(error, precision)
    return c


def configure_tile_grid(configs: Iterable[tuple[int, int, int, int, int]]) -> None:
    """Rebuild the module autotuners over ``configs``.

    Triton's autotuner is created once at import; a custom ``--tile-configs``
    grid has to replace it before any launch, so the runner calls this instead
    of mutating the module state from the argument parser.
    """
    global _gemm_autotuned, _bmm_autotuned
    if triton is None:  # pragma: no cover - guarded by the parser
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR!r}")
    _gemm_autotuned = _autotuned(_gemm_kernel, configs)
    _bmm_autotuned = _autotuned(_bmm_kernel, configs)


if triton is not None:
    _gemm_autotuned = _autotuned(_gemm_kernel, _DEFAULT_TILE_CONFIGS)
    _bmm_autotuned = _autotuned(_bmm_kernel, _DEFAULT_TILE_CONFIGS)
else:  # pragma: no cover - exercised by the host-side tests
    _gemm_autotuned = None
    _bmm_autotuned = None


def best_tile_config(autotuner) -> list[int] | None:
    """Return the autotuner's winning tile as a plain list, if it has one."""
    best = getattr(autotuner, "best_config", None)
    if best is None:
        return None
    kwargs = dict(getattr(best, "kwargs", {}) or {})
    return [
        kwargs.get("BLOCK_M"),
        kwargs.get("BLOCK_N"),
        kwargs.get("BLOCK_K"),
        getattr(best, "num_warps", None),
        getattr(best, "num_stages", None),
    ]


# ----------------------------------------------------------------------
# Shape coverage
# ----------------------------------------------------------------------
def square_shapes() -> list[tuple[int, int, int]]:
    """Return the ``N x N x N`` sweep."""
    return [(size, size, size) for size in _SQUARE_DIMS]


def rect_shapes() -> list[tuple[int, int, int]]:
    """Return the aspect-ratio extreme shapes."""
    return list(_RECT_SHAPES)


def ktail_shapes() -> list[tuple[int, int, int]]:
    """Return the block-boundary ``K`` sweep at a fixed output size."""
    m, n = _KTAIL_M_N
    return [(m, n, k) for k in _KTAIL_DIMS]


def dpa4_shapes() -> list[tuple[int, int, int]]:
    """Return the ``(rows, C) x (C, C)`` shapes the DPA4 graph issues."""
    return [(rows, c, c) for rows in _DPA4_EDGES for c in _DPA4_CHANNELS]


def batched_shapes() -> list[tuple[int, int, int, int]]:
    """Return the ``(batch, M, N, K)`` sweep."""
    return list(_BATCH_SHAPES)


_KIND_BUILDERS = {
    "square": square_shapes,
    "rect": rect_shapes,
    "ktail": ktail_shapes,
    "dpa4": dpa4_shapes,
    "batch": batched_shapes,
}


def _selected_kinds(raw: str) -> list[str]:
    if raw == "all":
        return list(_KIND_BUILDERS)
    kinds = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(kinds) - set(_KIND_BUILDERS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown kinds {unknown}; expected any of {sorted(_KIND_BUILDERS)} or 'all'"
        )
    return kinds


def collect_cases(kinds: Iterable[str]) -> list[dict[str, Any]]:
    """Return the case list for ``kinds`` as ``kind/m/n/k/batch`` records."""
    cases: list[dict[str, Any]] = []
    for kind in kinds:
        builder = _KIND_BUILDERS[kind]
        for shape in builder():
            if kind == "batch":
                batch, m, n, k = shape
            else:
                batch, (m, n, k) = 1, shape
            cases.append(
                {
                    "kind": kind,
                    "batch": int(batch),
                    "m": int(m),
                    "n": int(n),
                    "k": int(k),
                    "flops": 2 * int(batch) * int(m) * int(n) * int(k),
                }
            )
    unique: dict[tuple, dict[str, Any]] = {}
    for case in cases:
        key = (case["kind"], case["batch"], case["m"], case["n"], case["k"])
        unique.setdefault(key, case)
    return list(unique.values())


def case_bytes(case: dict[str, Any]) -> int:
    """Return the live fp32 tensor bytes one case needs (inputs + output)."""
    batch, m, n, k = case["batch"], case["m"], case["n"], case["k"]
    return 4 * batch * (m * k + k * n + m * n)


def filter_by_memory(
    cases: list[dict[str, Any]],
    *,
    budget_bytes: int,
    max_elements_per_tensor: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``cases`` into (kept, skipped) against a memory budget."""
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for case in cases:
        largest = case["batch"] * max(case["m"] * case["k"], case["k"] * case["n"])
        too_large = case_bytes(case) > budget_bytes
        too_wide = (
            max_elements_per_tensor is not None and largest > max_elements_per_tensor
        )
        (skipped if too_large or too_wide else kept).append(case)
    return kept, skipped


# ----------------------------------------------------------------------
# Measurement and accuracy
# ----------------------------------------------------------------------
def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def measure(
    function,
    device: torch.device,
    *,
    warmup: int,
    samples: int,
    repeats: int,
) -> dict[str, float]:
    """Return the median device-event milliseconds of ``function``."""
    for _ in range(warmup):
        function()
    _synchronize(device)
    per_call = []
    wall_per_call = []
    for _ in range(samples):
        _synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        for _ in range(repeats):
            function()
        stop.record()
        stop.synchronize()
        wall_per_call.append((time.perf_counter() - wall_start) * 1000.0 / repeats)
        per_call.append(start.elapsed_time(stop) / repeats)
    return {
        "median_device_ms": float(statistics.median(per_call)),
        "min_device_ms": float(min(per_call)),
        "median_wall_ms": float(statistics.median(wall_per_call)),
    }


def accuracy(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Return the max-absolute / max-relative / scaled-RMSE error of ``actual``."""
    difference = (actual - expected).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    scale = float(expected.abs().max().item()) if expected.numel() else 0.0
    if expected.numel() > 8_388_608:  # keep the reduction cheap on wide shapes
        stride = max(expected.numel() // 8_388_608, 1)
        sampled_actual = actual.flatten()[::stride].double()
        sampled_expected = expected.flatten()[::stride].double()
        rmse = float(
            torch.sqrt(((sampled_actual - sampled_expected) ** 2).mean()).item()
        )
    else:
        rmse = float(
            torch.sqrt(((actual.double() - expected.double()) ** 2).mean()).item()
        )
    return {
        "max_abs": max_abs,
        "max_rel": max_abs / scale if scale else 0.0,
        "rmse_scaled": rmse / scale if scale else 0.0,
    }


def _with_peak(arm: dict[str, Any], peak_tflops: float) -> dict[str, Any]:
    """Add the achieved percentage of a declared peak, when one is given."""
    if peak_tflops > 0:
        arm["percent_of_peak"] = arm["tflops"] / peak_tflops * 100.0
    return arm


def _tflops(flops: int, milliseconds: float) -> float:
    if milliseconds <= 0:
        return float("inf")
    return flops / (milliseconds * 1.0e6)


def _geomean_ratio(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    ratios = [
        row["ratio_triton_over_reference"]
        for row in rows
        if row.get(field) is not None and row["ratio_triton_over_reference"] > 0
    ]
    if not ratios:
        return None
    return float(math.exp(sum(math.log(value) for value in ratios) / len(ratios)))


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def _run_case(
    case: dict[str, Any],
    *,
    device: torch.device,
    rocblas: RocblasSgemm | None,
    args,
) -> dict[str, Any]:
    batch, m, n, k = case["batch"], case["m"], case["n"], case["k"]
    if batch == 1:
        a = torch.randn(m, k, device=device)
        b = torch.randn(k, n, device=device)
        triton_call = lambda: launch_triton_gemm(
            a, b, precision=args.precision, config=args.pinned_config
        )
        reference_call = (
            (lambda: rocblas(a, b)) if rocblas is not None else (lambda: a @ b)
        )
        reference_arm = "rocblas" if rocblas is not None else "aten_matmul"
        aten_call = lambda: a @ b
    else:
        a = torch.randn(batch, m, k, device=device)
        b = torch.randn(batch, k, n, device=device)
        triton_call = lambda: launch_triton_bmm(
            a, b, precision=args.precision, config=args.pinned_config
        )
        reference_call = lambda: torch.bmm(a, b)
        reference_arm = "aten_bmm"
        aten_call = reference_call

    triton_out = triton_call()
    reference_out = reference_call()
    _synchronize(device)
    error = accuracy(triton_out, reference_out)

    timing_kwargs = {
        "device": device,
        "warmup": args.warmup,
        "samples": args.samples,
        "repeats": args.repeats,
    }
    triton_timing = measure(triton_call, **timing_kwargs)
    reference_timing = measure(reference_call, **timing_kwargs)
    arms = {
        "triton": {
            **triton_timing,
            "tflops": _tflops(case["flops"], triton_timing["median_device_ms"]),
            "tile_config": best_tile_config(_gemm_autotuned)
            if batch == 1 and args.pinned_config is None
            else list(args.pinned_config or ()),
        },
        reference_arm: {
            **reference_timing,
            "tflops": _tflops(case["flops"], reference_timing["median_device_ms"]),
        },
    }
    if reference_arm != "aten_matmul":
        aten_timing = measure(aten_call, **timing_kwargs)
        arms["aten"] = {
            **aten_timing,
            "tflops": _tflops(case["flops"], aten_timing["median_device_ms"]),
        }

    for arm in arms.values():
        _with_peak(arm, args.peak_tflops)
    ratio = (
        arms["triton"]["median_device_ms"] / arms[reference_arm]["median_device_ms"]
        if arms[reference_arm]["median_device_ms"] > 0
        else float("inf")
    )
    tolerance = 5.0e-3 if args.precision == "tf32" else 1.0e-3
    row = {
        "kind": case["kind"],
        "shape": {"batch": batch, "m": m, "n": n, "k": k},
        "shape_text": (f"{m},{n},{k}" if batch == 1 else f"{batch}x{m},{n},{k}"),
        "flops": case["flops"],
        "reference_arm": reference_arm,
        "arms": arms,
        "accuracy_vs_reference": error,
        "accuracy_ok": error["max_rel"] <= tolerance,
        "accuracy_tolerance_max_rel": tolerance,
        "ratio_triton_over_reference": ratio,
        "triton_gap_percent": (1.0 - 1.0 / ratio) * 100.0 if ratio else None,
    }
    return row


def _report_table(rows: list[dict[str, Any]], stdout) -> None:
    header = (
        f"{'kind':<7} {'shape':<18} {'triton ms':>10} {'triton TF':>10} "
        f"{'ref ms':>10} {'ref TF':>10} {'triton/ref':>11}  acc"
    )
    print(header, file=stdout)
    for row in rows:
        arms = row["arms"]
        reference = arms[row["reference_arm"]]
        triton_arm = arms["triton"]
        print(
            f"{row['kind']:<7} {row['shape_text']:<18} "
            f"{triton_arm['median_device_ms']:>10.4f} {triton_arm['tflops']:>10.2f} "
            f"{reference['median_device_ms']:>10.4f} {reference['tflops']:>10.2f} "
            f"{row['ratio_triton_over_reference']:>10.3f}x  "
            f"{'ok' if row['accuracy_ok'] else 'FAIL'}",
            file=stdout,
        )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, Any] = {}
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        by_kind[kind] = {
            "cases": len(subset),
            "geomean_triton_over_reference": _geomean_ratio(subset, "median_device_ms"),
            "best_ratio": min(row["ratio_triton_over_reference"] for row in subset),
            "worst_ratio": max(row["ratio_triton_over_reference"] for row in subset),
            "triton_tflops_max": max(row["arms"]["triton"]["tflops"] for row in subset),
            "reference_tflops_max": max(
                row["arms"][row["reference_arm"]]["tflops"] for row in subset
            ),
            "triton_percent_of_peak_max": max(
                (row["arms"]["triton"].get("percent_of_peak", 0.0) for row in subset),
                default=0.0,
            ),
            "reference_percent_of_peak_max": max(
                (
                    row["arms"][row["reference_arm"]].get("percent_of_peak", 0.0)
                    for row in subset
                ),
                default=0.0,
            ),
        }
    return {
        "cases": len(rows),
        "cases_within_10_percent": sum(
            1 for row in rows if row["ratio_triton_over_reference"] <= 1.10
        ),
        "cases_accuracy_ok": sum(1 for row in rows if row["accuracy_ok"]),
        "geomean_triton_over_reference": _geomean_ratio(rows, "median_device_ms"),
        "by_kind": by_kind,
        "interpretation": (
            "geomean_triton_over_reference > 1 means Triton is slower than the "
            "reference arm; the ratio is per-shape time, so a value of 2 means "
            "the Triton GEMM needs twice as long and reaches half the TFLOP/s."
        ),
    }


def run_comparison(args) -> dict[str, Any]:
    """Run the sweep and return (also print and optionally write) the report."""
    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    if args.custom_tile_grids:
        configure_tile_grid(args.tile_configs)

    kinds = _selected_kinds(args.kinds)
    if args.cases:
        kinds = ["custom"]
        cases = [
            {
                "kind": "custom",
                "batch": batch,
                "m": m,
                "n": n,
                "k": k,
                "flops": 2 * batch * m * n * k,
            }
            for batch, m, n, k in args.cases
        ]
    else:
        cases = collect_cases(kinds)

    free_bytes, total_bytes = (int(value) for value in torch.cuda.mem_get_info(device))
    budget = int(free_bytes * args.memory_fraction)
    kept, skipped = filter_by_memory(
        cases, budget_bytes=budget, max_elements_per_tensor=args.max_elements_per_tensor
    )

    needs_rocblas = any(case["batch"] == 1 for case in kept)
    rocblas = None
    if needs_rocblas:
        try:
            rocblas = RocblasSgemm(args.rocblas_library)
        except RocblasError as error:
            if args.require_rocblas:
                raise
            print(
                f"warning: direct rocBLAS unavailable ({error}); the 2-D "
                "reference falls back to torch.matmul",
                file=sys.stderr,
            )

    try:
        rows = []
        for index, case in enumerate(kept):
            rows.append(_run_case(case, device=device, rocblas=rocblas, args=args))
            if (index + 1) % args.progress_every == 0 or index + 1 == len(kept):
                print(f"Completed {index + 1}/{len(kept)} cases", flush=True)
    finally:
        if rocblas is not None:
            rocblas.close()

    report = {
        "title": "Triton GEMM versus direct rocBLAS SGEMM",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "hip_version": str(torch.version.hip),
        "torch_version": str(torch.__version__),
        "triton_version": getattr(triton, "__version__", None),
        "precision": args.precision,
        "pinned_tile_config": list(args.pinned_config or ()),
        "autotune_tile_configs": [list(config) for config in args.tile_configs],
        "group_size_m": args.group_size_m,
        "declared_peak_tflops": args.peak_tflops or None,
        "kinds": kinds,
        "rocblas_library": getattr(rocblas, "library_path", None),
        "memory_budget_bytes": budget,
        "device_free_bytes_before": free_bytes,
        "device_total_bytes": total_bytes,
        "case_count": len(rows),
        "skipped_cases": [
            {
                "kind": case["kind"],
                "shape": [case["batch"], case["m"], case["n"], case["k"]],
            }
            for case in skipped
        ],
        "timing": {
            "warmup": args.warmup,
            "samples": args.samples,
            "repeats_per_sample": args.repeats,
            "device_event_timing": True,
        },
        "results": rows,
        "aggregate": _aggregate(rows),
    }

    print(
        f"\nTriton GEMM vs reference: device={report['device_name']}, "
        f"HIP={report['hip_version']}, triton={report['triton_version']}, "
        f"precision={args.precision}"
    )
    if skipped:
        print(
            f"skipped {len(skipped)} case(s) over the {budget / 2**30:.1f} GiB budget"
        )
    _report_table(rows, sys.stdout)

    aggregate = report["aggregate"]
    print("\nAggregate (ratio = triton time / reference time; <1 means Triton wins):")
    for kind, summary in aggregate["by_kind"].items():
        print(
            f"  {kind:<7} cases={summary['cases']:<3} "
            f"geomean={summary['geomean_triton_over_reference']:.3f}x  "
            f"best={summary['best_ratio']:.3f}x  worst={summary['worst_ratio']:.3f}x  "
            f"peak TF triton={summary['triton_tflops_max']:.2f} "
            f"ref={summary['reference_tflops_max']:.2f}"
        )
    print(
        f"  overall geomean={aggregate['geomean_triton_over_reference']:.3f}x, "
        f"{aggregate['cases_within_10_percent']}/{aggregate['cases']} within 10% of "
        f"the reference, accuracy ok for {aggregate['cases_accuracy_ok']}/{aggregate['cases']}"
    )

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Full report: {output_path}")
    return report


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_tile_configs(raw: str) -> list[tuple[int, int, int, int, int]]:
    """Parse ``BM,BN,BK,warps,stages`` groups separated by ``;``."""
    configs = []
    for chunk in raw.split(";"):
        fields = [field.strip() for field in chunk.split(",") if field.strip()]
        if not fields:
            continue
        if len(fields) != 5:
            raise argparse.ArgumentTypeError(
                f"invalid tile config {chunk!r}; expected BM,BN,BK,warps,stages"
            )
        try:
            values = tuple(int(field) for field in fields)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid integer in tile config {chunk!r}"
            ) from error
        if any(value < 1 for value in values):
            raise argparse.ArgumentTypeError(f"tile config {chunk!r} must be positive")
        configs.append(values)
    if not configs:
        raise argparse.ArgumentTypeError("at least one tile config is required")
    return configs


def parse_case_list(raw: str) -> list[tuple[int, int, int, int]]:
    """Parse ``M,N,K`` or ``B,M,N,K`` groups separated by ``;``."""
    cases = []
    for chunk in raw.split(";"):
        fields = [field.strip() for field in chunk.split(",") if field.strip()]
        if not fields:
            continue
        if len(fields) == 3:
            m, n, k = (int(field) for field in fields)
            cases.append((1, m, n, k))
        elif len(fields) == 4:
            batch, m, n, k = (int(field) for field in fields)
            cases.append((batch, m, n, k))
        else:
            raise argparse.ArgumentTypeError(
                f"invalid case {chunk!r}; expected M,N,K or B,M,N,K"
            )
        if any(value < 1 for value in cases[-1]):
            raise argparse.ArgumentTypeError(f"case {chunk!r} must be positive")
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a Triton tl.dot GEMM against direct rocBLAS SGEMM "
        "across a wide shape sweep."
    )
    parser.add_argument(
        "--kinds",
        default="all",
        help="comma list of square,rect,ktail,dpa4,batch or 'all' (default)",
    )
    parser.add_argument(
        "--cases",
        type=parse_case_list,
        default=None,
        help="restrict the sweep to M,N,K or B,M,N,K groups separated by ';'",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=tuple(_PRECISION_CODES),
        default=DEFAULT_PRECISION,
        help="tl.dot input_precision (default: ieee, what the DPA4 kernels use)",
    )
    parser.add_argument(
        "--tile-configs",
        type=parse_tile_configs,
        default=None,
        help="autotune over these BM,BN,BK,warps,stages groups instead of the built-in grid",
    )
    parser.add_argument(
        "--tile-config",
        type=parse_tile_configs,
        default=None,
        help="pin one tile config (disables autotuning)",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=0.0,
        help="declared fp32 peak of the device; adds the achieved percent of peak",
    )
    parser.add_argument("--group-size-m", type=int, default=8)
    parser.add_argument(
        "--memory-fraction", type=float, default=_DEFAULT_MEMORY_FRACTION
    )
    parser.add_argument(
        "--max-elements-per-tensor",
        type=int,
        default=None,
        help="optional hard cap on elements in any single input",
    )
    parser.add_argument(
        "--rocblas-library", default=None, help="optional librocblas.so path"
    )
    parser.add_argument(
        "--require-rocblas",
        action="store_true",
        help="fail instead of falling back to torch.matmul when direct rocBLAS is missing",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--output-json", default=None)
    return parser


def _parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.memory_fraction < 1.0:
        parser.error("--memory-fraction must be between 0 and 1")
    if args.tile_config is not None and len(args.tile_config) != 1:
        parser.error("--tile-config pins exactly one config")
    if args.tile_configs is not None and args.tile_config is not None:
        parser.error("--tile-configs and --tile-config are mutually exclusive")
    if args.tile_configs is not None and not _TRITON_AVAILABLE:
        parser.error("--tile-configs requires Triton")
    if args.max_elements_per_tensor is not None and args.max_elements_per_tensor < 1:
        parser.error("--max-elements-per-tensor must be positive")
    if args.warmup < 0 or args.samples < 1 or args.repeats < 1:
        parser.error("warmup must be nonnegative; samples and repeats must be positive")
    if args.group_size_m < 1:
        parser.error("--group-size-m must be positive")
    if args.peak_tflops < 0:
        parser.error("--peak-tflops must be nonnegative")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if not _TRITON_AVAILABLE:
        parser.error(f"Triton is required: {_TRITON_IMPORT_ERROR!r}")
    try:
        _selected_kinds(args.kinds)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    args.pinned_config = args.tile_config[0] if args.tile_config else None
    args.custom_tile_grids = args.tile_configs is not None
    args.tile_configs = args.tile_configs or list(_DEFAULT_TILE_CONFIGS)
    return args


def main(argv: list[str] | None = None) -> int:
    # ``--help`` and argument validation come first so the CLI stays usable
    # (and self-describing) on a host without a device.
    args = _parse_args(argv)
    if not _HAS_HIP_DEVICE:
        print(
            "this comparison requires a PyTorch HIP/DCU device; no HIP device visible",
            file=sys.stderr,
        )
        return 2
    try:
        report = run_comparison(args)
    except Exception as error:  # noqa: BLE001 - surface the cause to the caller
        print(
            f"Triton/rocBLAS GEMM comparison failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    return 0 if all(row["accuracy_ok"] for row in report["results"]) else 1


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
class ShapeCoverageTests(unittest.TestCase):
    def test_every_kind_is_nonempty_and_unique(self):
        for kind, builder in _KIND_BUILDERS.items():
            shapes = builder()
            self.assertTrue(shapes, f"{kind} produced no shape")
            self.assertEqual(len(shapes), len(set(shapes)), f"{kind} repeats a shape")

    def test_square_sweep_covers_small_to_large(self):
        dims = [shape[0] for shape in square_shapes()]
        self.assertEqual(dims, sorted(dims))
        self.assertIn(128, dims)
        self.assertIn(8192, dims)

    def test_ktail_sweep_straddles_tile_boundaries(self):
        tails = [shape[2] for shape in ktail_shapes()]
        for k in (1, 31, 33, 4097):
            self.assertIn(k, tails)
        self.assertTrue(any(k % 16 != 0 for k in tails))

    def test_dpa4_sweep_matches_the_profiled_graph(self):
        shapes = dpa4_shapes()
        edges = {shape[0] for shape in shapes}
        channels = {shape[1] for shape in shapes}
        self.assertIn(181838, edges)
        self.assertIn(3000, edges)
        self.assertIn(128, channels)
        for rows, c, k in shapes:
            self.assertEqual(c, k, "the DPA4 GEMMs are square on the weight side")

    def test_case_records_carry_flops(self):
        cases = collect_cases(["square", "batch"])
        square = next(case for case in cases if case["kind"] == "square")
        self.assertEqual(square["batch"], 1)
        self.assertEqual(square["flops"], 2 * square["m"] * square["n"] * square["k"])
        batched = next(case for case in cases if case["kind"] == "batch")
        self.assertGreater(batched["batch"], 1)
        self.assertEqual(
            batched["flops"],
            2 * batched["batch"] * batched["m"] * batched["n"] * batched["k"],
        )

    def test_duplicate_cases_collapse(self):
        cases = collect_cases(["ktail", "ktail"])
        self.assertEqual(len(cases), len(ktail_shapes()))


class MemoryFilterTests(unittest.TestCase):
    def test_budget_skips_oversized_cases_only(self):
        cases = collect_cases(["square"])
        kept, skipped = filter_by_memory(cases, budget_bytes=case_bytes(cases[0]))
        self.assertEqual([case["m"] for case in kept], [cases[0]["m"]])
        self.assertEqual(len(skipped), len(cases) - 1)

    def test_element_cap_is_independent_of_the_byte_budget(self):
        cases = collect_cases(["ktail"])
        kept, skipped = filter_by_memory(
            cases, budget_bytes=1 << 60, max_elements_per_tensor=1024
        )
        self.assertTrue(kept)
        self.assertTrue(skipped)


class ReportingTests(unittest.TestCase):
    def test_tflops_and_geomean(self):
        self.assertAlmostEqual(_tflops(2 * 1024**3, 2.0), 2 * 1024**3 / 2.0e6)
        rows = [
            {"ratio_triton_over_reference": 2.0, "median_device_ms": 2.0},
            {"ratio_triton_over_reference": 8.0, "median_device_ms": 8.0},
        ]
        self.assertAlmostEqual(_geomean_ratio(rows, "median_device_ms"), 4.0)

    def test_aggregate_reports_the_worst_and_best_kind(self):
        rows = [
            {
                "kind": "square",
                "ratio_triton_over_reference": 1.5,
                "accuracy_ok": True,
                "reference_arm": "rocblas",
                "arms": {
                    "triton": {"tflops": 5.0, "median_device_ms": 1.5},
                    "rocblas": {"tflops": 7.5, "median_device_ms": 1.0},
                },
            },
            {
                "kind": "square",
                "ratio_triton_over_reference": 3.0,
                "accuracy_ok": False,
                "reference_arm": "rocblas",
                "arms": {
                    "triton": {"tflops": 2.0, "median_device_ms": 3.0},
                    "rocblas": {"tflops": 6.0, "median_device_ms": 1.0},
                },
            },
        ]
        aggregate = _aggregate(rows)
        self.assertEqual(aggregate["cases"], 2)
        self.assertEqual(aggregate["cases_accuracy_ok"], 1)
        self.assertEqual(aggregate["by_kind"]["square"]["best_ratio"], 1.5)
        self.assertEqual(aggregate["by_kind"]["square"]["worst_ratio"], 3.0)
        self.assertAlmostEqual(
            aggregate["by_kind"]["square"]["reference_tflops_max"], 7.5
        )

    def test_parser_pins_or_widens_the_tile_grid(self):
        args = (
            _parse_args(["--tile-config", "128,128,64,8,3"])
            if _TRITON_AVAILABLE
            else None
        )
        if args is not None:
            self.assertEqual(args.pinned_config, (128, 128, 64, 8, 3))
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--tile-configs", "128,128"])  # too short

    def test_case_parser_accepts_batched_and_plain_groups(self):
        self.assertEqual(
            parse_case_list("64,64,64;8,512,512,128"),
            [(1, 64, 64, 64), (8, 512, 512, 128)],
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_case_list("64,64")


class _KernelStub:
    """Stand-in for a Triton kernel: records launches instead of running."""

    def __init__(self, function, calls, name):
        self.function = function
        self.calls = calls
        self.name = name

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append(
                {"kernel": self.name, "grid": grid, "args": args, "kwargs": kwargs}
            )

        return launch


class _AutotunedStub(_KernelStub):
    """Stand-in for ``triton.autotune``: applies a config when launching."""

    def __init__(self, function, calls, name, configs):
        super().__init__(function, calls, name)
        self.configs = configs
        self.best_config = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            config = self.configs[0]
            merged = dict(kwargs)
            merged.update(config.kwargs)
            merged["num_warps"] = config.num_warps
            merged["num_stages"] = config.num_stages
            self.calls.append(
                {"kernel": self.name, "grid": grid, "args": args, "kwargs": merged}
            )

        return launch


def _load_with_stub_triton():
    """Import this module against a fake Triton that records kernel launches.

    The launch sites are the most error-prone part of a Triton kernel and the
    only part that can be verified without a device: an argument passed in the
    wrong order, or a missing stride, is a silent wrong answer on hardware.
    Loading the module twice -- once for the real import and once against the
    stub -- keeps the device-dependent code path untouched.
    """
    import importlib
    import types

    calls: list[dict[str, Any]] = []

    def jit(function):
        return _KernelStub(function, calls, function.__name__)

    def autotune(*, configs, key):
        def decorate(kernel):
            name = getattr(kernel, "name", getattr(kernel, "__name__", "kernel"))
            return _AutotunedStub(kernel.function, calls, name, configs)

        return decorate

    class Config:
        def __init__(self, kwargs, num_warps=None, num_stages=None):
            self.kwargs = dict(kwargs)
            self.num_warps = num_warps
            self.num_stages = num_stages

    triton_stub = types.ModuleType("triton")
    triton_stub.__version__ = "stub"
    triton_stub.jit = jit
    triton_stub.autotune = autotune
    triton_stub.Config = Config
    triton_stub.cdiv = lambda numerator, denominator: -(-numerator // denominator)
    language_stub = types.ModuleType("triton.language")

    saved = {name: sys.modules.get(name) for name in ("triton", "triton.language")}
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "triton_gemm_compare_stubbed", Path(__file__)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module, calls


class LaunchContractTests(unittest.TestCase):
    """The launch sites must name every kernel parameter, in order."""

    @classmethod
    def setUpClass(cls):
        import inspect

        cls.module, cls.calls = _load_with_stub_triton()
        cls.inspect = inspect

    def setUp(self):
        self.calls.clear()

    def _kernel_parameters(self, name: str) -> list[str]:
        kernel = getattr(self.module, name)
        function = getattr(kernel, "function", kernel)
        return [
            parameter.name
            for parameter in self.inspect.signature(function).parameters.values()
        ]

    def _check_launch(self, call: dict[str, Any], kernel_name: str) -> None:
        """Every kernel parameter must be bound, and none of them twice."""
        parameters = self._kernel_parameters(kernel_name)
        self.assertGreater(len(parameters), 0)
        positional = parameters[: len(call["args"])]
        launch_meta = {"num_warps", "num_stages", "num_ctas", "maxnreg"}
        keyword = [name for name in call["kwargs"] if name not in launch_meta]
        self.assertLessEqual(len(call["args"]), len(parameters))
        for name in keyword:
            self.assertIn(name, parameters)
            # A parameter bound positionally must not also arrive as a keyword.
            self.assertNotIn(name, positional)
        bound = set(positional) | set(keyword)
        missing = [name for name in parameters if name not in bound]
        self.assertEqual(
            missing, [], f"{kernel_name} would be launched without {missing}"
        )

    def test_gemm_launch_binds_every_parameter_in_order(self):
        a = torch.randn(32, 8)
        b = torch.randn(8, 16)
        self.module._gemm_autotuned = self.module._autotuned(
            self.module._gemm_kernel, self.module._DEFAULT_TILE_CONFIGS
        )
        self.module.launch_triton_gemm(a, b, precision="ieee", group_size_m=4)
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertIn("gemm", call["kernel"])
        # Strides arrive in the documented order: A row/col, B row/col, C row/col.
        strides = [value for value in call["args"][6:12]]
        self.assertEqual(strides, [8, 1, 16, 1, 16, 1])
        self.assertEqual(call["args"][3:6], (32, 16, 8))
        self._check_launch(call, "_gemm_kernel")
        self.assertEqual(call["kwargs"]["PRECISION"], _PRECISION_CODES["ieee"])
        self.assertEqual(call["kwargs"]["GROUP_SIZE_M"], 4)
        for block in ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
            self.assertIn(block, call["kwargs"])
        self.assertIn("num_warps", call["kwargs"])

    def test_gemm_pinned_config_skips_the_autotuner(self):
        a = torch.randn(32, 8)
        b = torch.randn(8, 16)
        self.module.launch_triton_gemm(a, b, config=(16, 16, 8, 2, 1))
        call = self.calls[0]
        self.assertEqual(call["grid"], (2,))
        self.assertEqual(call["kwargs"]["BLOCK_M"], 16)
        self.assertEqual(call["kwargs"]["BLOCK_K"], 8)
        self.assertEqual(call["kwargs"]["num_warps"], 2)
        self._check_launch(call, "_gemm_kernel")

    def test_batched_launch_binds_every_parameter_in_order(self):
        a = torch.randn(2, 32, 8)
        b = torch.randn(2, 8, 16)
        self.module.launch_triton_bmm(a, b, config=(16, 16, 8, 2, 1))
        call = self.calls[0]
        self.assertEqual(call["grid"], (2, 2))
        self.assertEqual(call["args"][3:6], (32, 16, 8))
        # batch, row strides of A, then B's batch/row/inner, then C's.
        self.assertEqual(call["args"][6:14], (256, 8, 16, 128, 1, 512, 16, 1))
        self._check_launch(call, "_bmm_kernel")

    def test_requested_precision_reaches_the_kernel(self):
        a = torch.randn(16, 8)
        b = torch.randn(8, 16)
        self.module.launch_triton_gemm(a, b, precision="tf32", config=(16, 16, 8, 2, 1))
        self.assertEqual(self.calls[0]["kwargs"]["PRECISION"], _PRECISION_CODES["tf32"])


class ReferenceAccuracyTests(unittest.TestCase):
    """The comparison must reject a fast-but-wrong kernel."""

    def test_accuracy_metrics_flag_a_wrong_result(self):
        expected = torch.ones(64, 64)
        self.assertTrue(accuracy(expected, expected)["max_abs"] == 0.0)
        wrong = expected + 1.0
        error = accuracy(wrong, expected)
        self.assertAlmostEqual(error["max_abs"], 1.0)
        self.assertAlmostEqual(error["max_rel"], 1.0)

    def test_accuracy_scales_with_the_reference_magnitude(self):
        expected = torch.full((32, 32), 100.0)
        error = accuracy(expected + 0.5, expected)
        self.assertAlmostEqual(error["max_abs"], 0.5)
        self.assertAlmostEqual(error["max_rel"], 0.005, places=6)


@unittest.skipUnless(
    _HAS_HIP_DEVICE and _TRITON_AVAILABLE, "requires a DCU device and Triton"
)
class TritonGemmSmokeTests(unittest.TestCase):
    def test_small_gemm_matches_torch(self):
        torch.manual_seed(0)
        a = torch.randn(257, 129, device="cuda")
        b = torch.randn(129, 511, device="cuda")
        expected = a @ b
        actual = launch_triton_gemm(a, b, precision="ieee", config=(64, 64, 32, 4, 3))
        torch.cuda.synchronize()
        self.assertLess(accuracy(actual, expected)["max_rel"], 1e-4)

    def test_batched_gemm_matches_torch(self):
        torch.manual_seed(0)
        a = torch.randn(4, 128, 64, device="cuda")
        b = torch.randn(4, 64, 192, device="cuda")
        expected = torch.bmm(a, b)
        actual = launch_triton_bmm(a, b, precision="ieee", config=(64, 64, 32, 4, 3))
        torch.cuda.synchronize()
        self.assertLess(accuracy(actual, expected)["max_rel"], 1e-4)


if __name__ == "__main__":
    sys.exit(main())
