"""Broad accuracy/performance comparison for DUMMA, rocBLAS, and PyTorch.

Run on a supported DTK/DCU host after building the native DUMMA extension::

    source /opt/dtk/env.sh
    PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
      python tests/dcu/test_dcu_gemm_compare.py --suite all \
      --output-json /tmp/dcu-gemm-comparison.json

The default accuracy suite covers tile-boundary and K-tail combinations. The
performance suite covers square, rectangular, tail, and larger matrices through
4096x4096x4096. Direct rocBLAS is called through its C API; the PyTorch arm is
reported as PyTorch HIP/ATen because its internal BLAS dispatch is not controlled
by this test. This test does not use or replace the production GEMM API.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Iterable

import torch

_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TEST_DIR.parents[1]
for _path in (str(_TEST_DIR), str(_PROJECT_ROOT / "python")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from kernels import gemm as dumma_gemm  # noqa: E402
from kernels.native import native_available  # noqa: E402
from rocblas_ctypes import RocblasSgemm  # noqa: E402

_HAS_HIP_DEVICE = torch.cuda.is_available() and torch.version.hip is not None

_ACCURACY_MN = (1, 15, 16, 17, 31, 32, 33, 63, 64, 65)
_ACCURACY_K = (0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33)
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


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.numel() == 0:
        return {"max_abs": 0.0, "max_rel": 0.0, "rmse": 0.0}
    difference = actual.to(torch.float64) - expected.to(torch.float64)
    absolute = difference.abs()
    expected_magnitude = expected.to(torch.float64).abs().clamp_min(1.0e-12)
    return {
        "max_abs": float(absolute.max().item()),
        "max_rel": float((absolute / expected_magnitude).max().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
    }


def _assert_output(
    name: str,
    output: torch.Tensor,
    shape: tuple[int, int],
    device: torch.device,
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
    if not torch.isfinite(output).all().item():
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
        "samples": [float(value) for value in sample_ms],
        "median_device_ms": float(statistics.median(sample_ms)),
        "min_device_ms": float(min(sample_ms)),
        "max_device_ms": float(max(sample_ms)),
        "median_synchronized_wall_ms": float(statistics.median(wall_samples_ms)),
        "device_event_timing": True,
    }


def _performance_shapes(args) -> list[tuple[int, int, int]]:
    if args.performance_shapes:
        return list(args.performance_shapes)
    return list(_DEFAULT_PERFORMANCE_SHAPES)


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
    args,
) -> dict[str, Any]:
    m, n, k = shape
    max_tensor_elements = max(m * k, k * n, m * n)
    if max_tensor_elements > args.max_elements_per_tensor:
        raise ValueError(
            f"shape {shape} needs {max_tensor_elements} elements in one tensor, over "
            f"--max-elements-per-tensor={args.max_elements_per_tensor}"
        )
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
        _assert_output(backend, output, (m, n), device)
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
            torch.testing.assert_close(
                output.to(reference.dtype), reference, rtol=rtol, atol=atol
            )
        except AssertionError as error:
            raise AssertionError(
                f"{backend} accuracy check failed for M,N,K={shape}; "
                f"reference={reference_method}, rtol={rtol}, atol={atol}: {error}"
            ) from error
        errors[backend] = _error_metrics(output, reference)

    pairwise = {}
    for left, right in (
        ("dumma", "rocblas_direct"),
        ("dumma", "pytorch_hip_aten"),
        ("rocblas_direct", "pytorch_hip_aten"),
    ):
        if left in outputs and right in outputs:
            pairwise[f"{left}_vs_{right}"] = _error_metrics(outputs[left], outputs[right])

    performance = None
    if do_performance and m > 0 and n > 0 and k > 0:
        performance = {}
        for backend, function in functions.items():
            timing = _measure_device_events(
                function,
                device,
                warmup=args.warmup,
                samples=args.samples,
                repeats_per_sample=args.repeats_per_sample,
            )
            median_ms = timing["median_device_ms"]
            timing["effective_tflops"] = float(flops / (median_ms * 1.0e9))
            timing["first_call_wall_ms"] = first_call_wall_ms[backend]
            performance[backend] = timing
        dumma_ms = performance["dumma"]["median_device_ms"]
        for timing in performance.values():
            timing["speedup_vs_dumma"] = float(dumma_ms / timing["median_device_ms"])

    _synchronize(device)
    return {
        "shape_m_n_k": {"m": m, "n": n, "k": k},
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
    performance = _performance_shapes(args)
    accuracy_set = set(accuracy) if args.suite in ("accuracy", "all") else set()
    performance_set = set(performance) if args.suite in ("performance", "all") else set()
    if args.suite == "accuracy":
        shapes = list(accuracy)
    elif args.suite == "performance":
        shapes = list(performance)
    else:
        shapes = list(dict.fromkeys([*accuracy, *performance]))

    with torch.cuda.device(device), torch.inference_mode(), RocblasSgemm(args.rocblas_library) as rocblas:
        results = []
        for index, shape in enumerate(shapes):
            results.append(
                _run_case(
                    shape,
                    index=index,
                    device=device,
                    rocblas=rocblas,
                    do_performance=shape in performance_set,
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
        "total_shape_cases": len(results),
        "suite": args.suite,
        "results": results,
    }

    performance_rows = [row for row in results if row["performance"] is not None]
    print(
        f"\nGEMM comparison: device={report['device_name']} ({architecture}), "
        f"HIP={report['hip_version']}, rocBLAS={report['rocblas_library']}"
    )
    print("M,N,K       backend             median ms   TFLOP/s  speedup vs DUMMA")
    for row in performance_rows:
        shape = row["shape_m_n_k"]
        shape_text = f"{shape['m']},{shape['n']},{shape['k']}"
        for backend, timing in row["performance"].items():
            print(
                f"{shape_text:<11} {backend:<19} "
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
    parser.add_argument("--max-elements-per-tensor", type=int, default=16_777_216)
    parser.add_argument("--fp64-reference-max-flops", type=int, default=5_000_000_000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repeats-per-sample", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    parser.add_argument("--atol", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--output-json", default=None)
    return parser


def _parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_elements_per_tensor < 1:
        parser.error("--max-elements-per-tensor must be positive")
    if args.fp64_reference_max_flops < 0:
        parser.error("--fp64-reference-max-flops must be nonnegative")
    if args.warmup < 0 or args.samples < 1 or args.repeats_per_sample < 1:
        parser.error("warmup must be nonnegative; samples and repeats must be positive")
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


if __name__ == "__main__":
    raise SystemExit(main())
