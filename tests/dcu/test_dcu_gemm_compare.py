"""Broad accuracy/performance comparison for DUMMA, rocBLAS, and PyTorch.

Run on a supported DTK/DCU host after building the native DUMMA extension::

    source /opt/dtk/env.sh
    PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
      python tests/dcu/test_dcu_gemm_compare.py --suite all \
      --output-json /tmp/dcu-gemm-comparison.json

The first rocBLAS invocation lazily builds a test-local C++ HIP extension with
``hipcc`` and links against rocBLAS. Source the DTK environment first; pass
``--rocblas-library`` when the shared library is outside the linker path.

The default accuracy suite covers tile-boundary and K-tail combinations. The
performance suite covers compute-oriented cases through 4096x4096x4096 and adds
memory-residency cases with large M/N and a moderate K. A test-local C++ HIP extension calls the direct rocBLAS C API
on the same Torch tensors and stream. The PyTorch arm is reported as PyTorch
HIP/ATen because its internal BLAS dispatch is not controlled by this test. This
test does not use or replace the production GEMM API.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock
from typing import Any, Iterable

import torch

_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TEST_DIR.parents[1]
for _path in (str(_TEST_DIR), str(_PROJECT_ROOT / "python")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from kernels import gemm as dumma_gemm  # noqa: E402
from kernels.native import native_available  # noqa: E402
from rocblas_cpp import RocblasSgemm  # noqa: E402

_HAS_HIP_DEVICE = torch.cuda.is_available() and torch.version.hip is not None

_ACCURACY_MN = (1, 15, 16, 17, 31, 32, 33, 63, 64, 65)
_ACCURACY_K = (0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33)
_DUMMA_TILE = 16
_DEFAULT_MEMORY_FRACTION = 0.80
_DEFAULT_VALIDATION_CHUNK_ELEMENTS = 1_048_576
_FALLBACK_MAX_ELEMENTS_PER_TENSOR = 16_777_216
_MEMORY_RESIDENCY_K = 64
_MEMORY_RESIDENCY_INTERMEDIATE_MN = (8192, 16384, 32768)
_DEFAULT_PERFORMANCE_SHAPES = (
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (256, 1024, 128),
    (1024, 256, 512),
    (512, 2048, 256),
    (2048, 512, 1024),
    (257, 511, 129),
    (1023, 2049, 513),
)


def accuracy_shapes() -> list[tuple[int, int, int]]:
    """Cartesian tile-edge/K-tail sweep (shape order is M,N,K)."""
    shapes = [
        (m, n, k)
        for m in _ACCURACY_MN
        for n in _ACCURACY_MN
        for k in _ACCURACY_K
    ]
    shapes.extend(((0, 7, 4), (7, 0, 4)))
    return shapes


def parse_shape_list(raw: str) -> list[tuple[int, int, int]]:
    """Parse semicolon-separated M,N,K triples."""
    shapes = []
    for chunk in raw.split(";"):
        fields = [field.strip() for field in chunk.split(",")]
        if len(fields) != 3:
            raise argparse.ArgumentTypeError(
                f"invalid shape {chunk!r}; expected M,N,K triples separated by ';'"
            )
        try:
            shape = tuple(int(field) for field in fields)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid integer in shape {chunk!r}") from error
        m, n, k = shape
        if m < 0 or n < 0 or k < 0 or (m == 0 and n == 0):
            raise argparse.ArgumentTypeError(
                f"invalid nonnegative GEMM shape {shape}; at least one output dimension must be nonzero"
            )
        shapes.append(shape)
    if not shapes:
        raise argparse.ArgumentTypeError("at least one M,N,K shape is required")
    return shapes


def _error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    chunk_elements: int = _DEFAULT_VALIDATION_CHUNK_ELEMENTS,
) -> dict[str, float]:
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"metric tensors have different shapes: {tuple(actual.shape)} and "
            f"{tuple(expected.shape)}"
        )
    if actual.numel() == 0:
        return {"max_abs": 0.0, "max_rel": 0.0, "rmse": 0.0}
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")

    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    max_abs = torch.zeros((), dtype=torch.float64, device=actual.device)
    max_rel = torch.zeros((), dtype=torch.float64, device=actual.device)
    squared_error_sum = torch.zeros((), dtype=torch.float64, device=actual.device)
    for start in range(0, actual.numel(), chunk_elements):
        stop = min(start + chunk_elements, actual.numel())
        actual_chunk = actual_flat[start:stop].to(torch.float64)
        expected_chunk = expected_flat[start:stop].to(torch.float64)
        difference = actual_chunk - expected_chunk
        absolute = difference.abs()
        expected_magnitude = expected_chunk.abs().clamp_min(1.0e-12)
        max_abs = torch.maximum(max_abs, absolute.max())
        max_rel = torch.maximum(max_rel, (absolute / expected_magnitude).max())
        squared_error_sum = squared_error_sum + difference.square().sum()

    return {
        "max_abs": float(max_abs.item()),
        "max_rel": float(max_rel.item()),
        "rmse": float((squared_error_sum / actual.numel()).sqrt().item()),
    }


def _assert_close_chunked(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    chunk_elements: int = _DEFAULT_VALIDATION_CHUNK_ELEMENTS,
) -> None:
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"{name} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.numel() == 0:
        return
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")

    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    all_close = torch.ones((), dtype=torch.bool, device=actual.device)
    for start in range(0, actual.numel(), chunk_elements):
        stop = min(start + chunk_elements, actual.numel())
        actual_chunk = actual_flat[start:stop].to(expected.dtype)
        expected_chunk = expected_flat[start:stop]
        all_close = all_close & torch.isclose(
            actual_chunk, expected_chunk, rtol=rtol, atol=atol, equal_nan=False
        ).all()
    if all_close.item():
        return

    # Re-run only the failing chunk through PyTorch's assertion helper to retain
    # its familiar diagnostic without building a full-output comparison mask.
    for start in range(0, actual.numel(), chunk_elements):
        stop = min(start + chunk_elements, actual.numel())
        actual_chunk = actual_flat[start:stop].to(expected.dtype)
        expected_chunk = expected_flat[start:stop]
        try:
            torch.testing.assert_close(
                actual_chunk, expected_chunk, rtol=rtol, atol=atol
            )
        except AssertionError as error:
            raise AssertionError(
                f"{name} mismatch in flattened chunk [{start}:{stop}]: {error}"
            ) from error
    raise AssertionError(f"{name} values differ beyond rtol={rtol}, atol={atol}")


def _assert_output(
    name: str,
    output: torch.Tensor,
    shape: tuple[int, int],
    device: torch.device,
    *,
    chunk_elements: int = _DEFAULT_VALIDATION_CHUNK_ELEMENTS,
) -> None:
    if not isinstance(output, torch.Tensor):
        raise AssertionError(f"{name} returned {type(output)!r}, expected Tensor")
    if output.device != device:
        raise AssertionError(f"{name} output is on {output.device}, expected {device}")
    if output.dtype != torch.float32 or tuple(output.shape) != shape:
        raise AssertionError(
            f"{name} output contract mismatch: dtype={output.dtype}, "
            f"shape={tuple(output.shape)}, expected float32/{shape}"
        )
    if not output.is_contiguous():
        raise AssertionError(f"{name} output must be contiguous to support bounded validation")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    finite = torch.ones((), dtype=torch.bool, device=output.device)
    output_flat = output.reshape(-1)
    for start in range(0, output.numel(), chunk_elements):
        stop = min(start + chunk_elements, output.numel())
        finite = finite & torch.isfinite(output_flat[start:stop]).all()
    if not finite.item():
        raise AssertionError(f"{name} returned non-finite values")


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _call_with_wall_time(function, device: torch.device):
    _synchronize(device)
    started = time.perf_counter()
    output = function()
    _synchronize(device)
    return output, (time.perf_counter() - started) * 1000.0


def _measure_device_events(
    function,
    device: torch.device,
    *,
    warmup: int,
    samples: int,
    repeats_per_sample: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    _synchronize(device)
    sample_ms = []
    wall_samples_ms = []
    for _ in range(samples):
        _synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()
        for _ in range(repeats_per_sample):
            function()
        end_event.record()
        end_event.synchronize()
        wall_samples_ms.append((time.perf_counter() - wall_start) * 1000.0 / repeats_per_sample)
        sample_ms.append(start_event.elapsed_time(end_event) / repeats_per_sample)
    return {
        "warmup_calls": warmup,
        "repeats_per_sample": repeats_per_sample,
        "samples": [float(value) for value in sample_ms],
        "median_device_ms": float(statistics.median(sample_ms)),
        "min_device_ms": float(min(sample_ms)),
        "max_device_ms": float(max(sample_ms)),
        "median_synchronized_wall_ms": float(statistics.median(wall_samples_ms)),
        "device_event_timing": True,
    }


def _estimated_case_memory_bytes(
    shape: tuple[int, int, int],
    *,
    fp64_reference_max_flops: int,
    validation_chunk_elements: int,
) -> int:
    """Conservative live-tensor estimate for correctness and validation."""
    m, n, k = shape
    input_and_outputs = 4 * (m * k + k * n + 3 * m * n)
    flops = 2 * m * n * k
    fp64_reference = 0
    if flops <= fp64_reference_max_flops:
        fp64_reference = 8 * (m * k + k * n + m * n)
    chunk_items = min(m * n, validation_chunk_elements)
    # Allow for several simultaneously live float64 validation temporaries.
    validation_scratch = chunk_items * 8 * 8
    return input_and_outputs + fp64_reference + validation_scratch


def _max_elements_in_shape(shape: tuple[int, int, int]) -> int:
    m, n, k = shape
    return max(m * k, k * n, m * n)


def _largest_memory_residency_shape(
    memory_budget_bytes: int,
    *,
    k: int,
    fp64_reference_max_flops: int,
    validation_chunk_elements: int,
    max_elements_per_tensor: int | None,
) -> tuple[int, int, int] | None:
    """Find the largest tile-aligned square-output case fitting the budget."""
    if memory_budget_bytes <= 0 or k <= 0:
        return None
    # Even with K much smaller than M/N, output storage is the dominant term.
    # This upper bound is intentionally loose; binary search checks the complete
    # estimate, including inputs, FP64 references, and validation scratch.
    max_n = math.isqrt(memory_budget_bytes // 12) + _DUMMA_TILE
    max_n = min(max_n, 2**31 - 1)
    if max_elements_per_tensor is not None:
        max_n = min(max_n, math.isqrt(max_elements_per_tensor) + _DUMMA_TILE)
    max_units = max_n // _DUMMA_TILE
    low_units, high_units = 0, max_units
    while low_units < high_units:
        mid_units = (low_units + high_units + 1) // 2
        n = mid_units * _DUMMA_TILE
        shape = (n, n, k)
        fits_cap = (
            max_elements_per_tensor is None
            or _max_elements_in_shape(shape) <= max_elements_per_tensor
        )
        fits_memory = (
            _estimated_case_memory_bytes(
                shape,
                fp64_reference_max_flops=fp64_reference_max_flops,
                validation_chunk_elements=validation_chunk_elements,
            )
            <= memory_budget_bytes
        )
        if fits_cap and fits_memory:
            low_units = mid_units
        else:
            high_units = mid_units - 1
    n = low_units * _DUMMA_TILE
    if n < _DUMMA_TILE:
        return None
    return (n, n, k)


def _memory_residency_shapes(args, memory_budget_bytes: int) -> list[tuple[int, int, int]]:
    candidates = [(n, n, args.stress_k) for n in _MEMORY_RESIDENCY_INTERMEDIATE_MN]
    largest = _largest_memory_residency_shape(
        memory_budget_bytes,
        k=args.stress_k,
        fp64_reference_max_flops=args.fp64_reference_max_flops,
        validation_chunk_elements=args.validation_chunk_elements,
        max_elements_per_tensor=args.max_elements_per_tensor,
    )
    if largest is not None and largest[0] > 4096:
        candidates.append(largest)
    return list(
        dict.fromkeys(
            shape
            for shape in candidates
            if (
                args.max_elements_per_tensor is None
                or _max_elements_in_shape(shape) <= args.max_elements_per_tensor
            )
            and _estimated_case_memory_bytes(
                shape,
                fp64_reference_max_flops=args.fp64_reference_max_flops,
                validation_chunk_elements=args.validation_chunk_elements,
            )
            <= memory_budget_bytes
        )
    )


def _performance_shapes(
    args, memory_budget_bytes: int | None
) -> tuple[list[tuple[int, int, int]], set[tuple[int, int, int]]]:
    if args.performance_shapes:
        return list(args.performance_shapes), set()
    shapes = list(_DEFAULT_PERFORMANCE_SHAPES)
    residency = set()
    if args.suite in ("performance", "all") and memory_budget_bytes is not None:
        stress_shapes = _memory_residency_shapes(args, memory_budget_bytes)
        shapes.extend(stress_shapes)
        residency.update(stress_shapes)
    return list(dict.fromkeys(shapes)), residency


def _memory_sizing(args, device: torch.device) -> tuple[dict[str, Any], int | None, int | None]:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_bytes, total_bytes = int(free_bytes), int(total_bytes)
    except Exception as error:
        effective_cap = min(
            args.max_elements_per_tensor or _FALLBACK_MAX_ELEMENTS_PER_TENSOR,
            _FALLBACK_MAX_ELEMENTS_PER_TENSOR,
        )
        print(
            "WARNING: cannot query DCU free memory; omitting automatic memory-residency "
            f"shapes and using the fallback per-tensor cap ({effective_cap}): {error}",
            file=sys.stderr,
        )
        return (
            {
                "source": "unavailable",
                "query_error": f"{type(error).__name__}: {error}",
                "free_bytes": None,
                "total_bytes": None,
                "target_fraction": args.memory_fraction,
                "budget_bytes": None,
                "effective_max_elements_per_tensor": effective_cap,
            },
            None,
            effective_cap,
        )

    budget_bytes = int(free_bytes * args.memory_fraction)
    effective_cap = args.max_elements_per_tensor
    return (
        {
            "source": "torch.cuda.mem_get_info",
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "target_fraction": args.memory_fraction,
            "budget_bytes": budget_bytes,
            "effective_max_elements_per_tensor": effective_cap,
        },
        budget_bytes,
        effective_cap,
    )


def _preflight_shapes(
    shapes: Iterable[tuple[int, int, int]],
    *,
    args,
    memory_budget_bytes: int | None,
    max_elements_per_tensor: int | None,
) -> dict[tuple[int, int, int], int]:
    estimates = {}
    for shape in shapes:
        max_elements = _max_elements_in_shape(shape)
        if max_elements_per_tensor is not None and max_elements > max_elements_per_tensor:
            raise ValueError(
                f"preflight rejected shape {shape}: a tensor needs {max_elements} elements, "
                f"over the effective per-tensor cap {max_elements_per_tensor}"
            )
        estimate = _estimated_case_memory_bytes(
            shape,
            fp64_reference_max_flops=args.fp64_reference_max_flops,
            validation_chunk_elements=args.validation_chunk_elements,
        )
        estimates[shape] = estimate
        if memory_budget_bytes is not None and estimate > memory_budget_bytes:
            raise ValueError(
                f"preflight rejected shape {shape}: estimated live memory "
                f"{estimate:,} bytes exceeds the DCU budget {memory_budget_bytes:,} bytes"
            )
    return estimates


def _check_hardware(device_index: int) -> tuple[torch.device, Any]:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("this comparison requires HIP-enabled PyTorch and a visible DCU")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError(
            f"device index {device_index} is invalid; "
            f"{torch.cuda.device_count()} DCU device(s) are visible"
        )
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        if not native_available():
            raise RuntimeError(
                "DUMMA extension is unavailable on this device; build with "
                "DPA4_BUILD_HIP=1 DPA4_ENABLE_DUMMA=1 and check the DTK/DUMMA setup"
            )
        properties = torch.cuda.get_device_properties(device)
    return device, properties


def _run_case(
    shape: tuple[int, int, int],
    *,
    index: int,
    device: torch.device,
    rocblas: RocblasSgemm,
    do_performance: bool,
    is_memory_residency: bool,
    estimated_case_bytes: int,
    args,
) -> dict[str, Any]:
    m, n, k = shape
    max_tensor_elements = _max_elements_in_shape(shape)
    if (
        args.effective_max_elements_per_tensor is not None
        and max_tensor_elements > args.effective_max_elements_per_tensor
    ):
        raise ValueError(
            f"shape {shape} needs {max_tensor_elements} elements in one tensor, over "
            f"the effective per-tensor cap="
            f"{args.effective_max_elements_per_tensor}"
        )
    try:
        allocated_before = int(torch.cuda.memory_allocated(device))
        reserved_before = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
    except (AttributeError, RuntimeError):
        allocated_before = None
        reserved_before = None
    torch.manual_seed(args.seed + index)
    a = torch.randn((m, k), device=device, dtype=torch.float32).contiguous()
    b = torch.randn((k, n), device=device, dtype=torch.float32).contiguous()
    if a.device != device or b.device != device or not (a.is_contiguous() and b.is_contiguous()):
        raise AssertionError("GEMM inputs are not contiguous tensors on the selected DCU")

    functions = {
        "dumma": lambda: dumma_gemm(a, b),
        "rocblas_direct": lambda: rocblas(a, b),
        "pytorch_hip_aten": lambda: torch.matmul(a, b),
    }
    # ROCBLAS is not needed when the reduction dimension is zero; the operation
    # is the zero matrix. Report that degenerate arm explicitly, never substitute
    # PyTorch and call it rocBLAS.
    direct_rocblas_status = "measured"
    if k == 0 and m > 0 and n > 0:
        functions.pop("rocblas_direct")
        direct_rocblas_status = "not_called_zero_inner_dimension"
    elif m == 0 or n == 0:
        direct_rocblas_status = "empty_output_no_gemm_call"

    outputs: dict[str, torch.Tensor] = {}
    first_call_wall_ms: dict[str, float] = {}
    for backend, function in functions.items():
        output, wall_ms = _call_with_wall_time(function, device)
        _assert_output(
            backend,
            output,
            (m, n),
            device,
            chunk_elements=args.validation_chunk_elements,
        )
        outputs[backend] = output
        first_call_wall_ms[backend] = float(wall_ms)

    _synchronize(device)
    flops = 2 * m * n * k
    use_fp64 = flops <= args.fp64_reference_max_flops
    if use_fp64:
        reference = torch.matmul(a.to(torch.float64), b.to(torch.float64))
        reference_method = "torch_matmul_float64"
    else:
        reference = outputs["pytorch_hip_aten"]
        reference_method = "torch_matmul_float32_for_large_case"
    _synchronize(device)

    rtol = args.rtol
    atol = args.atol
    errors = {}
    for backend, output in outputs.items():
        try:
            _assert_close_chunked(
                backend,
                output,
                reference,
                rtol=rtol,
                atol=atol,
                chunk_elements=args.validation_chunk_elements,
            )
        except AssertionError as error:
            raise AssertionError(
                f"{backend} accuracy check failed for M,N,K={shape}; "
                f"reference={reference_method}, rtol={rtol}, atol={atol}: {error}"
            ) from error
        errors[backend] = _error_metrics(
            output, reference, chunk_elements=args.validation_chunk_elements
        )

    pairwise = {}
    for left, right in (
        ("dumma", "rocblas_direct"),
        ("dumma", "pytorch_hip_aten"),
        ("rocblas_direct", "pytorch_hip_aten"),
    ):
        if left in outputs and right in outputs:
            pairwise[f"{left}_vs_{right}"] = _error_metrics(
                outputs[left],
                outputs[right],
                chunk_elements=args.validation_chunk_elements,
            )

    # Timed GEMMs need only the inputs. Release the three accuracy outputs and
    # reference first, avoiding a fourth full output at the timing peak.
    del reference, outputs, output
    _synchronize(device)

    performance = None
    if do_performance and m > 0 and n > 0 and k > 0:
        performance = {}
        warmup = args.stress_warmup if is_memory_residency else args.warmup
        samples = args.stress_samples if is_memory_residency else args.samples
        repeats = args.stress_repeats if is_memory_residency else args.repeats_per_sample
        for backend, function in functions.items():
            timing = _measure_device_events(
                function,
                device,
                warmup=warmup,
                samples=samples,
                repeats_per_sample=repeats,
            )
            median_ms = timing["median_device_ms"]
            timing["effective_tflops"] = float(flops / (median_ms * 1.0e9))
            timing["first_call_wall_ms"] = first_call_wall_ms[backend]
            timing["repeats_per_sample"] = repeats
            timing["case_kind"] = (
                "memory_residency_stress" if is_memory_residency else "compute_comparison"
            )
            performance[backend] = timing
        dumma_ms = performance["dumma"]["median_device_ms"]
        for timing in performance.values():
            timing["speedup_vs_dumma"] = float(dumma_ms / timing["median_device_ms"])

    _synchronize(device)
    try:
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    except (AttributeError, RuntimeError):
        peak_allocated_bytes = None
        peak_reserved_bytes = None
    peak_allocated_delta = (
        max(0, peak_allocated_bytes - allocated_before)
        if peak_allocated_bytes is not None and allocated_before is not None
        else None
    )
    peak_reserved_delta = (
        max(0, peak_reserved_bytes - reserved_before)
        if peak_reserved_bytes is not None and reserved_before is not None
        else None
    )
    return {
        "shape_m_n_k": {"m": m, "n": n, "k": k},
        "case_kind": "memory_residency_stress" if is_memory_residency else "compute_comparison",
        "estimated_case_bytes": estimated_case_bytes,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_allocated_delta_bytes": peak_allocated_delta,
        "peak_reserved_delta_bytes": peak_reserved_delta,
        "dtype": "float32",
        "input_device": str(device),
        "reference_method": reference_method,
        "tolerance": {"rtol": rtol, "atol": atol},
        "rocblas_status": direct_rocblas_status,
        "accuracy": errors,
        "pairwise_accuracy": pairwise,
        "performance": performance,
    }


def run_comparison(args) -> dict[str, Any]:
    device, properties = _check_hardware(args.device_index)
    supported = {"gfx926", "gfx928", "gfx936", "gfx938"}
    architecture = str(getattr(properties, "gcnArchName", "unknown")).split(":", 1)[0]
    if architecture not in supported:
        raise RuntimeError(
            f"unsupported DCU architecture {architecture!r}; expected one of {sorted(supported)}"
        )

    accuracy = accuracy_shapes()
    accuracy_set = set(accuracy) if args.suite in ("accuracy", "all") else set()

    with torch.cuda.device(device), torch.inference_mode(), RocblasSgemm(args.rocblas_library) as rocblas:
        memory_sizing, memory_budget_bytes, effective_cap = _memory_sizing(args, device)
        args.effective_max_elements_per_tensor = effective_cap
        performance, residency_shapes = _performance_shapes(args, memory_budget_bytes)
        performance_set = set(performance) if args.suite in ("performance", "all") else set()
        if memory_sizing["budget_bytes"] is None:
            print("Memory-based residency sizing is unavailable; using fallback limits.")
        else:
            stress_text = ", ".join(
                f"{m}x{n}x{k}" for m, n, k in sorted(residency_shapes)
            ) or "none"
            print(
                "Memory sizing: "
                f"free={memory_sizing['free_bytes']:,} B / "
                f"total={memory_sizing['total_bytes']:,} B, "
                f"budget={memory_sizing['budget_bytes']:,} B "
                f"({memory_sizing['target_fraction']:.0%}); "
                f"residency shapes={stress_text}"
            )
        if args.suite == "accuracy":
            shapes = list(accuracy)
        elif args.suite == "performance":
            shapes = list(performance)
        else:
            shapes = list(dict.fromkeys([*accuracy, *performance]))
        case_memory_estimates = _preflight_shapes(
            shapes,
            args=args,
            memory_budget_bytes=memory_budget_bytes,
            max_elements_per_tensor=effective_cap,
        )

        results = []
        for index, shape in enumerate(shapes):
            results.append(
                _run_case(
                    shape,
                    index=index,
                    device=device,
                    rocblas=rocblas,
                    do_performance=shape in performance_set,
                    is_memory_residency=shape in residency_shapes,
                    estimated_case_bytes=case_memory_estimates[shape],
                    args=args,
                )
            )
            if (index + 1) % args.progress_every == 0 or index + 1 == len(shapes):
                print(f"Completed {index + 1}/{len(shapes)} shape cases", flush=True)

    report = {
        "title": "DUMMA vs direct rocBLAS vs PyTorch HIP/ATen",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "architecture": architecture,
        "hip_version": str(torch.version.hip),
        "torch_version": str(torch.__version__),
        "rocblas_library": rocblas.library_path,
        "pytorch_backend_note": (
            "PyTorch HIP/ATen may internally dispatch to rocBLAS or rocBLASLt; "
            "this arm measures the PyTorch API, not a second direct rocBLAS call."
        ),
        "boundary_accuracy_shape_count": len(accuracy_set),
        "performance_shape_count": len(performance_set),
        "memory_residency_shape_count": len(residency_shapes),
        "memory_sizing": memory_sizing,
        "memory_telemetry_note": (
            "PyTorch allocator peaks may exclude direct rocBLAS internal workspace; reserved "
            "peaks include any cache already reserved before a case, with per-case deltas "
            "reported separately. Device free/total values are pre-suite snapshots."
        ),
        "total_shape_cases": len(results),
        "suite": args.suite,
        "results": results,
    }

    performance_rows = [row for row in results if row["performance"] is not None]
    print(
        f"\nGEMM comparison: device={report['device_name']} ({architecture}), "
        f"HIP={report['hip_version']}, rocBLAS={report['rocblas_library']}"
    )
    print("case kind                 M,N,K       backend             median ms   TFLOP/s  speedup vs DUMMA")
    for row in performance_rows:
        shape = row["shape_m_n_k"]
        shape_text = f"{shape['m']},{shape['n']},{shape['k']}"
        for backend, timing in row["performance"].items():
            print(
                f"{row['case_kind']:<25} {shape_text:<11} {backend:<19} "
                f"{timing['median_device_ms']:>9.4f} "
                f"{timing['effective_tflops']:>9.4f} "
                f"{timing['speedup_vs_dumma']:>9.4f}x"
            )
    summary = _aggregate_errors(results)
    print("\nAccuracy summary by reference (max absolute / max relative / RMSE):")
    for reference_name, backend_rows in summary.items():
        print(f"  reference={reference_name}")
        for backend, metrics in backend_rows.items():
            print(
                f"    {backend:<19} {metrics['max_abs']:.6g} / "
                f"{metrics['max_rel']:.6g} / {metrics['rmse']:.6g}"
            )
    pairwise = _aggregate_pairwise_errors(results)
    print("\nPairwise backend accuracy (max absolute / max relative / RMSE):")
    for pair, metrics in pairwise.items():
        print(
            f"  {pair:<48} {metrics['max_abs']:.6g} / "
            f"{metrics['max_rel']:.6g} / {metrics['rmse']:.6g}"
        )
    report["aggregate_accuracy_by_reference"] = summary
    report["aggregate_pairwise_accuracy"] = pairwise

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Full report: {output_path}")
    return report


def _aggregate_metric_rows(results, field: str):
    aggregate = {}
    squared_sums = {}
    counts = {}
    for row in results:
        for label, metrics in row[field].items():
            current = aggregate.setdefault(label, {"max_abs": 0.0, "max_rel": 0.0})
            current["max_abs"] = max(current["max_abs"], metrics["max_abs"])
            current["max_rel"] = max(current["max_rel"], metrics["max_rel"])
            size = math.prod(row["shape_m_n_k"][axis] for axis in ("m", "n"))
            squared_sums[label] = squared_sums.get(label, 0.0) + metrics["rmse"] ** 2 * size
            counts[label] = counts.get(label, 0) + size
    for label, current in aggregate.items():
        current["rmse"] = math.sqrt(squared_sums[label] / max(counts[label], 1))
    return aggregate


def _aggregate_errors(results: Iterable[dict[str, Any]]):
    grouped = {}
    for row in results:
        reference = row["reference_method"]
        grouped.setdefault(reference, []).append(row)
    return {
        reference: _aggregate_metric_rows(rows, "accuracy")
        for reference, rows in grouped.items()
    }


def _aggregate_pairwise_errors(results: Iterable[dict[str, Any]]):
    pairwise_rows = []
    for row in results:
        pairwise_rows.append(
            {
                "shape_m_n_k": row["shape_m_n_k"],
                "pairwise": row["pairwise_accuracy"],
            }
        )
    return _aggregate_metric_rows(pairwise_rows, "pairwise") if pairwise_rows else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare DUMMA GEMM, direct rocBLAS SGEMM, and PyTorch HIP/ATen on one DCU."
    )
    parser.add_argument("--suite", choices=("accuracy", "performance", "all"), default="all")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--rocblas-library", default=None, help="optional librocblas.so path")
    parser.add_argument(
        "--performance-shapes",
        type=parse_shape_list,
        default=None,
        help="semicolon-separated M,N,K triples",
    )
    parser.add_argument(
        "--max-elements-per-tensor",
        type=int,
        default=None,
        help="optional hard cap on elements in any single tensor",
    )
    parser.add_argument("--memory-fraction", type=float, default=_DEFAULT_MEMORY_FRACTION)
    parser.add_argument("--stress-k", type=int, default=_MEMORY_RESIDENCY_K)
    parser.add_argument(
        "--validation-chunk-elements",
        type=int,
        default=_DEFAULT_VALIDATION_CHUNK_ELEMENTS,
    )
    parser.add_argument("--fp64-reference-max-flops", type=int, default=5_000_000_000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repeats-per-sample", type=int, default=5)
    parser.add_argument("--stress-warmup", type=int, default=0)
    parser.add_argument("--stress-samples", type=int, default=1)
    parser.add_argument("--stress-repeats", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    parser.add_argument("--atol", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--output-json", default=None)
    return parser


def _parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_elements_per_tensor is not None and args.max_elements_per_tensor < 1:
        parser.error("--max-elements-per-tensor must be positive")
    if not 0.0 < args.memory_fraction < 1.0:
        parser.error("--memory-fraction must be greater than 0 and less than 1")
    if args.stress_k < 1 or args.stress_k % 4 != 0:
        parser.error("--stress-k must be a positive multiple of 4")
    if args.validation_chunk_elements < 1:
        parser.error("--validation-chunk-elements must be positive")
    if args.fp64_reference_max_flops < 0:
        parser.error("--fp64-reference-max-flops must be nonnegative")
    if args.warmup < 0 or args.samples < 1 or args.repeats_per_sample < 1:
        parser.error("warmup must be nonnegative; samples and repeats must be positive")
    if args.stress_warmup < 0 or args.stress_samples < 1 or args.stress_repeats < 1:
        parser.error("stress warmup must be nonnegative; stress samples and repeats must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.rtol < 0 or args.atol < 0:
        parser.error("--rtol and --atol must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_comparison(args)
    except Exception as error:
        print(f"GEMM comparison failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


@unittest.skipUnless(_HAS_HIP_DEVICE, "requires a PyTorch HIP/DCU device")
class DcuGemmComparisonSmokeTests(unittest.TestCase):
    def test_three_backend_tail_shapes(self):
        args = _parse_args(
            [
                "--suite",
                "accuracy",
                "--progress-every",
                "1000",
            ]
        )
        # Keep unittest discovery quick; the CLI's default accuracy suite is the
        # broad M/N tile-edge × K-tail sweep.
        args.performance_shapes = parse_shape_list("17,23,7;33,31,17;0,7,4;7,0,4")
        args.suite = "performance"
        report = run_comparison(args)
        self.assertGreaterEqual(len(report["results"]), 1)


class ShapeUtilityTests(unittest.TestCase):
    def test_accuracy_shape_sweep_covers_tile_edges_and_k_tails(self):
        shapes = accuracy_shapes()
        self.assertGreater(len(shapes), 1000)
        self.assertIn((16, 16, 4), shapes)
        self.assertIn((17, 33, 17), shapes)
        self.assertIn((0, 7, 4), shapes)

    def test_custom_shape_parser(self):
        self.assertEqual(parse_shape_list("16,32,4;17,23,7"), [(16, 32, 4), (17, 23, 7)])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_shape_list("16,32")

    def test_chunked_error_metrics_match_full_tensor_calculation(self):
        actual = torch.tensor([1.0, 1.25, -0.5, 3.0, 0.0, 4.5, -2.0])
        expected = torch.tensor([1.0, 1.0, -0.25, 3.5, 0.1, 4.0, -2.5])
        difference = actual.double() - expected.double()
        absolute = difference.abs()
        expected_magnitude = expected.double().abs().clamp_min(1.0e-12)
        expected_metrics = {
            "max_abs": absolute.max().item(),
            "max_rel": (absolute / expected_magnitude).max().item(),
            "rmse": difference.square().mean().sqrt().item(),
        }
        chunked_metrics = _error_metrics(actual, expected, chunk_elements=3)
        for key, value in expected_metrics.items():
            self.assertAlmostEqual(chunked_metrics[key], value, places=12)

    def test_chunked_assert_close_detects_error_after_boundary(self):
        actual = torch.ones(8)
        expected = torch.ones(8)
        expected[4] = 2.0
        with self.assertRaisesRegex(AssertionError, "chunk \[3:6\]"):
            _assert_close_chunked(
                "test", actual, expected, rtol=0.0, atol=0.0, chunk_elements=3
            )

    def test_chunked_output_finiteness_checks_later_chunk(self):
        output = torch.ones(_DEFAULT_VALIDATION_CHUNK_ELEMENTS + 1)
        output[-1] = float("nan")
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            _assert_output("test", output, tuple(output.shape), torch.device("cpu"))

    def test_memory_estimate_and_largest_shape_respect_budget_and_cap(self):
        shape = (8192, 8192, 64)
        budget = _estimated_case_memory_bytes(
            shape,
            fp64_reference_max_flops=0,
            validation_chunk_elements=1024,
        )
        self.assertEqual(
            budget,
            4 * (8192 * 64 + 64 * 8192 + 3 * 8192 * 8192) + 1024 * 64,
        )
        selected = _largest_memory_residency_shape(
            budget,
            k=64,
            fp64_reference_max_flops=0,
            validation_chunk_elements=1024,
            max_elements_per_tensor=8192 * 8192,
        )
        self.assertEqual(selected, shape)
        self.assertIsNone(
            _largest_memory_residency_shape(
                0,
                k=64,
                fp64_reference_max_flops=0,
                validation_chunk_elements=1024,
                max_elements_per_tensor=None,
            )
        )

    def test_memory_sizing_adds_stress_shapes_and_falls_back_safely(self):
        args = _parse_args([])
        free_bytes = 64 * 1024**3
        total_bytes = 64 * 1024**3
        with mock.patch("torch.cuda.mem_get_info", return_value=(free_bytes, total_bytes)):
            sizing, budget, cap = _memory_sizing(args, torch.device("cpu"))
        self.assertEqual(budget, int(free_bytes * args.memory_fraction))
        self.assertIsNone(cap)
        shapes, stress = _performance_shapes(args, budget)
        self.assertIn((8192, 8192, args.stress_k), stress)
        self.assertIn((16384, 16384, args.stress_k), stress)
        self.assertIn((32768, 32768, args.stress_k), stress)
        self.assertGreater(max(shape[0] for shape in stress), 32768)
        self.assertEqual(sizing["total_bytes"], total_bytes)

        with mock.patch("torch.cuda.mem_get_info", side_effect=RuntimeError("unavailable")):
            with redirect_stderr(StringIO()):
                fallback_sizing, fallback_budget, fallback_cap = _memory_sizing(
                    args, torch.device("cpu")
                )
        self.assertIsNone(fallback_budget)
        self.assertEqual(fallback_cap, _FALLBACK_MAX_ELEMENTS_PER_TENSOR)
        self.assertEqual(fallback_sizing["source"], "unavailable")

    def test_preflight_rejects_shape_over_memory_budget(self):
        args = _parse_args(["--fp64-reference-max-flops", "0"])
        shape = (128, 128, 64)
        estimate = _estimated_case_memory_bytes(
            shape,
            fp64_reference_max_flops=args.fp64_reference_max_flops,
            validation_chunk_elements=args.validation_chunk_elements,
        )
        with self.assertRaisesRegex(ValueError, "preflight rejected shape"):
            _preflight_shapes(
                [shape],
                args=args,
                memory_budget_bytes=estimate - 1,
                max_elements_per_tensor=None,
            )


if __name__ == "__main__":
    raise SystemExit(main())
