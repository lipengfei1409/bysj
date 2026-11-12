"""TileLang dense FlashAttention forward parity and opt-in benchmark.

Run the quick correctness tests with unittest discovery (configuration defaults
are used)::

    python -m unittest discover -s tests/dcu \
        -p 'test_tilelang_flash_attention.py' -v

Run the fixed-axis benchmark directly; add ``--coverage`` separately for the
broader heads/head_dim correctness matrix::

    python tests/dcu/test_tilelang_flash_attention.py \
        --benchmark --bench-warmup 1 --bench-repeats 2 \
        --max-qkv-input-bytes 8589934592 \
        --report-path /tmp/fa-report.json --csv-report-path /tmp/fa-report.csv

The benchmark fixes heads=8 and head_dim=128, varying only batch (1/4/16) and
sequence length (1k through 16k). Broader head/head_dim coverage remains in the
correctness matrix. Cases that error or exceed the memory budget are recorded as
skipped and excluded from metrics. The budget reserves device memory for
outputs, workspaces, and other processes. The test
covers the dense ``flash_attn_func`` forward contract only: zero dropout, no GQA,
windowing, ALiBi, softcap, or returned attention probabilities. These tests do
not exercise backward.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import statistics
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tilelang_ops import flash_attn_func


@dataclass(frozen=True)
class FlashAttentionConfig:
    benchmark: bool = False
    coverage: bool = False
    bench_repeats: int | None = None
    bench_warmup: int | None = None
    report_path: Path | None = None
    csv_report_path: Path | None = None
    max_qkv_input_bytes: int | None = None


_CONFIG = FlashAttentionConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TileLang FlashAttention correctness, coverage, and benchmark tests."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="benchmark sequence length and batch only (heads=8, head_dim=128)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="run broad forward correctness coverage across heads and head_dim",
    )
    parser.add_argument(
        "--bench-repeats", type=int, default=None,
        help="override adaptive repeats per timing sample (minimum 1)",
    )
    parser.add_argument(
        "--bench-warmup", type=int, default=None,
        help="override adaptive warmup calls (minimum 0)",
    )
    parser.add_argument(
        "--report-path", type=Path, default=None,
        help="JSON report destination; defaults to the system temporary directory",
    )
    parser.add_argument(
        "--csv-report-path", type=Path, default=None,
        help="benchmark CSV destination; defaults beside the JSON report",
    )
    parser.add_argument(
        "--max-qkv-input-bytes", type=int, default=None,
        help="override the estimated Q/K/V input-memory budget (minimum 1 byte)",
    )
    return parser


def _config_from_args(args) -> FlashAttentionConfig:
    return FlashAttentionConfig(
        benchmark=bool(args.benchmark),
        coverage=bool(args.coverage),
        bench_repeats=(
            None if args.bench_repeats is None else max(1, int(args.bench_repeats))
        ),
        bench_warmup=(
            None if args.bench_warmup is None else max(0, int(args.bench_warmup))
        ),
        report_path=args.report_path,
        csv_report_path=args.csv_report_path,
        max_qkv_input_bytes=(
            None
            if args.max_qkv_input_bytes is None
            else max(1, int(args.max_qkv_input_bytes))
        ),
    )


class FlashAttentionParserTests(unittest.TestCase):
    def test_cli_defaults_preserve_short_test_mode(self):
        config = _config_from_args(build_parser().parse_args([]))
        self.assertFalse(config.benchmark)
        self.assertFalse(config.coverage)
        self.assertIsNone(config.bench_repeats)
        self.assertIsNone(config.bench_warmup)
        self.assertIsNone(config.report_path)
        self.assertIsNone(config.csv_report_path)
        self.assertIsNone(config.max_qkv_input_bytes)

    def test_cli_options_populate_all_config_fields(self):
        args = build_parser().parse_args(
            [
                "--benchmark",
                "--coverage",
                "--bench-repeats", "2",
                "--bench-warmup", "1",
                "--report-path", "~/flash-attn-report.json",
                "--csv-report-path", "~/flash-attn-report.csv",
                "--max-qkv-input-bytes", "4096",
            ]
        )
        config = _config_from_args(args)
        self.assertTrue(config.benchmark)
        self.assertTrue(config.coverage)
        self.assertEqual(config.bench_repeats, 2)
        self.assertEqual(config.bench_warmup, 1)
        self.assertEqual(config.report_path, Path("~/flash-attn-report.json"))
        self.assertEqual(config.csv_report_path, Path("~/flash-attn-report.csv"))
        self.assertEqual(config.max_qkv_input_bytes, 4096)

    def test_numeric_overrides_keep_existing_clamping(self):
        args = build_parser().parse_args(
            ["--bench-repeats", "0", "--bench-warmup", "-2", "--max-qkv-input-bytes", "0"]
        )
        config = _config_from_args(args)
        self.assertEqual(config.bench_repeats, 1)
        self.assertEqual(config.bench_warmup, 0)
        self.assertEqual(config.max_qkv_input_bytes, 1)

    def test_benchmark_help_documents_fixed_heads_and_dimension(self):
        help_text = build_parser().format_help()
        self.assertIn("benchmark sequence length and batch only", help_text)
        self.assertIn("heads=8,", help_text)
        self.assertIn("head_dim=128)", help_text)


class FlashAttentionBenchmarkMatrixTests(unittest.TestCase):
    def test_benchmark_matrix_varies_only_sequence_length_and_batch(self):
        cases = _requested_benchmark_cases()
        axes = _requested_benchmark_axes()

        self.assertEqual(len(cases), 15)
        self.assertEqual(len(set(cases)), 15)
        self.assertEqual(axes["heads"], 8)
        self.assertEqual(axes["head_dim"], 128)
        self.assertEqual(axes["batch"], [1, 4, 16])
        self.assertEqual(axes["sequence_lengths"], [1024, 2048, 4096, 8192, 16384])
        self.assertEqual(axes["dtype"], "torch.float16")
        self.assertFalse(axes["causal"])
        self.assertEqual({case[2] for case in cases}, {1, 4, 16})
        self.assertEqual({case[3] for case in cases}, {8})
        self.assertEqual({case[4] for case in cases}, {128})
        self.assertTrue(all(case[0] == case[1] for case in cases))

    def test_broad_correctness_matrix_remains_unchanged(self):
        cases = _requested_stress_cases()
        self.assertEqual(len(cases), 90)
        self.assertEqual({case[3] for case in cases}, {8, 16, 32})
        self.assertEqual({case[4] for case in cases}, {64, 128})
        self.assertTrue(set(_requested_benchmark_cases()).issubset(set(cases)))
        self.assertEqual(len(_FORWARD_COVERAGE_CASES), 12)
        self.assertEqual(len(_BF16_COVERAGE_CASES), 4)

    def test_benchmark_runner_only_times_fixed_matrix(self):
        class Runner:
            def __init__(self):
                self.cases = []

            def _benchmark_case(self, reference, case, dtype, seed):
                self.cases.append(case)
                row = {**_case_dict(case, dtype), "timing": "synthetic"}
                return {}, row

        runner = Runner()
        report = {
            "benchmark_matrix": {"successful_cases": [], "skipped_cases": []},
            "benchmark_cases": [],
            "skipped_cases": [],
        }
        config = FlashAttentionConfig(benchmark=True)
        with (
            patch.object(sys.modules[__name__], "_CONFIG", config),
            patch.object(sys.modules[__name__], "_qkv_input_budget", return_value=(10**12, "test")),
            patch.object(sys.modules[__name__], "_write_report"),
            patch("builtins.print"),
        ):
            FlashAttentionExtendedTests._requested_benchmark_sweep(
                runner, reference=None, report=report, device=None
            )

        self.assertEqual(runner.cases, list(_requested_benchmark_cases()))
        self.assertEqual(len(report["benchmark_cases"]), 15)
        self.assertEqual(report["benchmark_matrix"]["requested_case_count"], 15)
        self.assertEqual(report["benchmark_matrix"]["successful_case_count"], 15)

    def test_correctness_runner_keeps_the_full_matrix_without_timing(self):
        class Runner:
            def __init__(self):
                self.cases = []

            def _correctness_case(self, reference, case, dtype, seed):
                self.cases.append(case)
                return {**_case_dict(case, dtype), "correct": True}

        runner = Runner()
        report = {
            "requested_matrix": {"successful_cases": [], "skipped_cases": []},
            "skipped_cases": [],
        }
        config = FlashAttentionConfig(coverage=True)
        with (
            patch.object(sys.modules[__name__], "_CONFIG", config),
            patch.object(sys.modules[__name__], "_qkv_input_budget", return_value=(10**12, "test")),
            patch.object(sys.modules[__name__], "_write_report"),
            patch("builtins.print"),
        ):
            FlashAttentionExtendedTests._requested_stress_sweep(
                runner, reference=None, report=report, device=None
            )

        self.assertEqual(runner.cases, list(_requested_stress_cases()))
        self.assertEqual(report["requested_matrix"]["requested_case_count"], 90)
        self.assertEqual(report["requested_matrix"]["successful_case_count"], 90)


class FlashAttentionCsvReportTests(unittest.TestCase):
    @staticmethod
    def _timing(median_ms, p95_ms):
        return {
            "first_invocation_wall_ms": median_ms * 2,
            "warmup_steps": 1,
            "repeat_steps": 3,
            "device_time": {"median_ms": median_ms, "p95_ms": p95_ms},
            "synchronized_wall_time": {
                "median_ms": median_ms + 0.1,
                "p95_ms": p95_ms + 0.1,
            },
        }

    def test_csv_flattens_backends_and_only_includes_benchmark_skips(self):
        measured_case = {
            **_case_dict((2048, 2048, 4, 8, 128, False), torch.float16),
            "tilelang": self._timing(1.0, 1.2),
            "flash_attn": self._timing(0.8, 0.9),
            "speedup_tilelang_vs_flash_attn": {
                "device_time_median": 0.8,
                "synchronized_wall_time_median": 0.8,
            },
            "accuracy": {
                "tilelang_vs_flash_attn": {
                    "max_abs": 0.02,
                    "rmse": 0.003,
                    "relative_l2": 0.004,
                },
                "tilelang_vs_pytorch_reference": {
                    "max_abs": 0.01,
                    "rmse": 0.002,
                    "relative_l2": 0.003,
                },
                "flash_attn_vs_pytorch_reference": {
                    "max_abs": 0.015,
                    "rmse": 0.0025,
                    "relative_l2": 0.0035,
                },
            },
            "torch_sdpa": {
                "status": "unsupported_or_failed",
                "reason": "flash backend unavailable",
            },
        }
        benchmark_skip = {
            **_case_dict((1024, 1024, 1, 8, 128, False), torch.float16),
            "status": "skipped",
            "group": "requested_benchmark_matrix",
            "reason": "out of memory",
            "counted_in_metrics": False,
        }
        correctness_skips = [
            {
                **_case_dict((4096, 4096, 1, 32, 128, False), torch.float16),
                "status": "skipped",
                "group": "requested_stress_matrix",
                "reason": "correctness-only skip",
                "counted_in_metrics": False,
            },
            {
                **_case_dict((8192, 8192, 1, 1, 64, False), torch.float16),
                "status": "skipped",
                "group": "supported_long_sweep_correctness",
                "reason": "supplemental correctness skip",
                "counted_in_metrics": False,
            },
        ]
        report = {
            "benchmark_cases": [measured_case],
            "skipped_cases": correctness_skips,
            "benchmark_matrix": {"skipped_cases": [benchmark_skip]},
            "requested_matrix": {"skipped_cases": correctness_skips[:1]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "benchmark.csv"
            _write_csv_report(report, csv_path)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 6)
        measured = {row["backend"]: row for row in rows if row["q_len"] == "2048"}
        self.assertEqual(set(measured), {"tilelang", "flash_attn", "torch_sdpa"})
        self.assertEqual(measured["tilelang"]["device_time_median_ms"], "1.0")
        self.assertEqual(
            measured["tilelang"]["tilelang_vs_flash_attn_device_speedup"], "0.8"
        )
        self.assertEqual(measured["tilelang"]["accuracy_rmse"], "0.003")
        self.assertEqual(measured["torch_sdpa"]["status"], "unsupported_or_failed")
        self.assertEqual(measured["torch_sdpa"]["reason"], "flash backend unavailable")
        skipped_rows = [row for row in rows if row["status"] == "skipped"]
        self.assertEqual(len(skipped_rows), 3)
        self.assertTrue(all(row["q_len"] == "1024" for row in skipped_rows))
        self.assertTrue(all(row["heads"] == "8" for row in skipped_rows))
        self.assertTrue(all(row["head_dim"] == "128" for row in skipped_rows))
        self.assertTrue(all(row["reason"] == "out of memory" for row in skipped_rows))

    def test_json_report_also_writes_csv_beside_it_by_default(self):
        report = {
            "benchmark_cases": [],
            "skipped_cases": [],
            "requested_matrix": {"skipped_cases": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            json_path = Path(temporary) / "run.json"
            csv_path = Path(temporary) / "run.csv"
            config = FlashAttentionConfig(benchmark=True, report_path=json_path)
            with patch.object(sys.modules[__name__], "_CONFIG", config):
                _write_report(report)
                self.assertEqual(_csv_report_path(), csv_path)
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            with csv_path.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.reader(stream)), list(_CSV_FIELDNAMES))

    def test_explicit_csv_report_path_is_respected(self):
        report = {"benchmark_cases": [], "skipped_cases": []}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "run.json"
            custom = root / "custom.csv"
            config = FlashAttentionConfig(
                benchmark=True,
                report_path=json_path,
                csv_report_path=custom,
            )
            with patch.object(sys.modules[__name__], "_CONFIG", config):
                self.assertEqual(_csv_report_path(), custom)
                _write_report(report)
            self.assertTrue(json_path.is_file())
            self.assertTrue(custom.is_file())
            self.assertFalse((root / "run.csv").exists())

    def test_csv_and_json_paths_must_differ(self):
        with tempfile.TemporaryDirectory() as temporary:
            same_path = Path(temporary) / "same.json"
            config = FlashAttentionConfig(
                benchmark=True,
                report_path=same_path,
                csv_report_path=same_path,
            )
            with patch.object(sys.modules[__name__], "_CONFIG", config):
                with self.assertRaisesRegex(ValueError, "must be different"):
                    _write_report({"benchmark_cases": []})


_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)
_LONG_SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384)
_REQUESTED_HEAD_COUNTS = (8, 16, 32)
_REQUESTED_BATCH_SIZES = (1, 4, 16)
_REQUESTED_HEAD_DIMS = (64, 128)
_BENCHMARK_HEADS = 8
_BENCHMARK_HEAD_DIM = 128
_DEFAULT_MAX_QKV_INPUT_BYTES = 16 * 1024 * 1024 * 1024
_QKV_TOTAL_MEMORY_FRACTION = 0.25
_QKV_FREE_MEMORY_FRACTION = 0.50
# Supplemental unequal-length and causal cases remain correctness checks only.
_SUPPLEMENTAL_CORRECTNESS_CASES = (
    (512, 512, 1, 8, 64, False),
    (512, 512, 1, 8, 64, True),
    (2048, 4096, 1, 4, 64, True),
)


def _requested_stress_cases():
    return tuple(
        (sequence_length, sequence_length, batch, heads, head_dim, False)
        for sequence_length in _LONG_SEQUENCE_LENGTHS
        for batch in _REQUESTED_BATCH_SIZES
        for heads in _REQUESTED_HEAD_COUNTS
        for head_dim in _REQUESTED_HEAD_DIMS
    )


def _requested_benchmark_cases():
    return tuple(
        (
            sequence_length,
            sequence_length,
            batch,
            _BENCHMARK_HEADS,
            _BENCHMARK_HEAD_DIM,
            False,
        )
        for sequence_length in _LONG_SEQUENCE_LENGTHS
        for batch in _REQUESTED_BATCH_SIZES
    )


def _requested_benchmark_axes():
    return {
        "heads": _BENCHMARK_HEADS,
        "batch": list(_REQUESTED_BATCH_SIZES),
        "head_dim": _BENCHMARK_HEAD_DIM,
        "sequence_lengths": list(_LONG_SEQUENCE_LENGTHS),
        "dtype": "torch.float16",
        "causal": False,
    }


def _estimated_qkv_input_bytes(case):
    q_len, kv_len, batch, heads, head_dim, _causal = case
    return batch * heads * head_dim * (q_len + 2 * kv_len) * 2


# The explicit reference materializes scores. At 8 million FP32 elements this
# is already a 32 MiB tensor before softmax and other intermediate storage.
_MAX_REFERENCE_SCORE_ELEMENTS = 8_000_000
_TORCH_SDPA_MATH_MAX_SEQUENCE = 2048

# (q_len, kv_len, batch, heads, head_dim, causal). The curated—not Cartesian—
# sweep reaches all supported head dimensions, batch/head variation, 32-token
# boundaries, equal/unequal lengths, and both bottom-right causal edge cases.
_FORWARD_COVERAGE_CASES = (
    (1, 1, 1, 1, 16, False),
    (31, 33, 1, 2, 32, False),
    (32, 32, 2, 4, 64, True),
    (33, 65, 1, 8, 128, False),
    (63, 65, 2, 1, 16, True),
    (64, 64, 1, 4, 32, False),
    (65, 97, 2, 8, 64, True),
    (127, 129, 1, 4, 128, False),
    (128, 128, 1, 1, 16, True),
    (129, 65, 2, 4, 32, True),
    (257, 512, 1, 2, 64, False),
    (512, 257, 1, 8, 128, True),
)
_BF16_COVERAGE_CASES = (
    (32, 33, 1, 2, 16, False),
    (65, 65, 2, 4, 32, True),
    (129, 257, 1, 4, 64, False),
    (257, 129, 1, 2, 128, True),
)


def pytorch_flash_attn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Explicit stable PyTorch reference with FlashAttention causal alignment."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "q, k, and v must have shape (batch, sequence, heads, head_dim)"
        )
    if q.shape[0] != k.shape[0] or k.shape != v.shape or q.shape[2:] != k.shape[2:]:
        raise ValueError(
            "reference requires matching batch, heads, and head dimensions"
        )
    scale = (
        1.0 / math.sqrt(q.shape[-1])
        if softmax_scale is None
        else float(softmax_scale)
    )
    q_acc = q.float().permute(0, 2, 1, 3)
    k_acc = k.float().permute(0, 2, 1, 3)
    v_acc = v.float().permute(0, 2, 1, 3)
    scores = torch.matmul(q_acc, k_acc.transpose(-1, -2)) * scale
    q_len = q.shape[1]
    kv_len = k.shape[1]
    if causal:
        q_pos = torch.arange(q_len, device=q.device)[:, None]
        kv_pos = torch.arange(kv_len, device=q.device)[None, :]
        allowed = kv_pos <= q_pos + (kv_len - q_len)
        scores = scores.masked_fill(~allowed[None, None, :, :], -torch.inf)
        valid_rows = allowed.any(dim=-1)[None, None, :, None]
        scores = torch.where(valid_rows, scores, torch.zeros_like(scores))
    else:
        valid_rows = None
    probs = torch.softmax(scores, dim=-1)
    if valid_rows is not None:
        probs = torch.where(valid_rows, probs, torch.zeros_like(probs))
    output = torch.matmul(probs, v_acc).permute(0, 2, 1, 3)
    return output.to(dtype=q.dtype)


def _open_source_flash_attn():
    try:
        from flash_attn import flash_attn_func as reference
    except Exception as error:  # optional target-specific package
        return None, f"flash_attn import failed: {type(error).__name__}: {error}"
    return reference, None


def _make_device_inputs(q_len, kv_len, dtype, *, batch=1, heads=2, head_dim=64):
    generator_device = torch.device("cuda")
    q = torch.randn(
        (batch, q_len, heads, head_dim), device=generator_device, dtype=dtype
    )
    k = torch.randn(
        (batch, kv_len, heads, head_dim), device=generator_device, dtype=dtype
    )
    v = torch.randn_like(k)
    return q, k, v


def _case_dict(case, dtype):
    q_len, kv_len, batch, heads, head_dim, causal = case
    return {
        "q_len": q_len,
        "kv_len": kv_len,
        "batch": batch,
        "heads": heads,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "causal": causal,
    }


def _tolerance(dtype):
    value = 8.0e-2 if dtype == torch.bfloat16 else 3.0e-2
    return {"rtol": value, "atol": value}


def _error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference32 = reference.detach().float()
    actual32 = actual.detach().float()
    difference = actual32 - reference32
    epsilon = 1.0e-12
    return {
        "max_abs": float(difference.abs().max().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "relative_l2": float(
            difference.norm().item() / max(reference32.norm().item(), epsilon)
        ),
    }


def _assert_output(test_case, name, output, expected_shape):
    test_case.assertEqual(tuple(output.shape), tuple(expected_shape), f"{name} shape")
    test_case.assertTrue(
        torch.isfinite(output).all().item(), f"{name} returned non-finite values"
    )


def _bf16_supported() -> bool:
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _synchronize(device):
    torch.cuda.synchronize(device)


def _percentile(values, quantile):
    """Nearest-rank percentile (p95 is the ceil(0.95*n)-th sample)."""
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _summarize_samples(samples):
    return {
        "samples_ms": [float(value) for value in samples],
        "median_ms": float(statistics.median(samples)),
        "p95_ms": float(_percentile(samples, 0.95)),
        "p95_convention": "nearest-rank",
    }


def _benchmark_steps(q_len, kv_len, *, batch=1, heads=1, head_dim=64):
    work = batch * heads * q_len * kv_len * head_dim
    override = _CONFIG.bench_repeats
    if override is not None:
        repeats = override
    elif work <= 1_000_000_000:
        repeats = 10
    elif work <= 10_000_000_000:
        repeats = 7
    elif work <= 50_000_000_000:
        repeats = 5
    elif work <= 200_000_000_000:
        repeats = 3
    elif work <= 1_000_000_000_000:
        repeats = 2
    else:
        repeats = 1

    warmup_override = _CONFIG.bench_warmup
    if warmup_override is not None:
        warmup = warmup_override
    elif work <= 1_000_000_000:
        warmup = 3
    elif work <= 10_000_000_000:
        warmup = 2
    elif work <= 200_000_000_000:
        warmup = 1
    else:
        warmup = 0
    return warmup, repeats


def _measure_forward(function, device, warmup_steps, repeat_steps):
    """Return output and synchronized first-call, event, and wall-time metrics.

    The first invocation is measured inside this process and may reuse JIT or
    disk caches from earlier tests; it is deliberately not labeled a cold-JIT
    measurement. Steady-state wall samples synchronize each call. Device-event
    samples isolate elapsed device time. Both are reported for transparency.
    """
    _synchronize(device)
    start = time.perf_counter()
    first_output = function()
    _synchronize(device)
    first_wall_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(warmup_steps):
        function()
    _synchronize(device)

    device_samples = []
    wall_samples = []
    for _ in range(repeat_steps):
        _synchronize(device)
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        wall_start = time.perf_counter()
        function()
        event_end.record()
        event_end.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
        device_samples.append(event_start.elapsed_time(event_end))

    return first_output, {
        "first_invocation_wall_ms": float(first_wall_ms),
        "warmup_steps": int(warmup_steps),
        "repeat_steps": int(repeat_steps),
        "device_time": _summarize_samples(device_samples),
        "synchronized_wall_time": _summarize_samples(wall_samples),
    }


def _make_sdpa_forward(q, k, v, *, scale, causal):
    """Build an SDPA call with bottom-right causality and safe long-seq dispatch."""
    attention_mask = None
    is_causal = causal
    if causal and q.shape[1] != k.shape[1]:
        q_pos = torch.arange(q.shape[1], device=q.device)[:, None]
        kv_pos = torch.arange(k.shape[1], device=q.device)[None, :]
        attention_mask = (kv_pos <= q_pos + (k.shape[1] - q.shape[1]))[None, None]
        is_causal = False

    def run_sdpa():
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attention_mask,
            scale=scale,
            is_causal=is_causal,
        )
        return output.transpose(1, 2)

    if max(q.shape[1], k.shape[1]) <= _TORCH_SDPA_MATH_MAX_SEQUENCE:
        return run_sdpa

    def run_fused_sdpa_only():
        # Never allow the quadratic-memory math fallback for very long sequences.
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            flash_backend = SDPBackend.FLASH_ATTENTION
        except (ImportError, AttributeError) as error:
            raise NotImplementedError(
                "this PyTorch version lacks the SDPA flash-backend selector "
                "needed to safely benchmark long sequences"
            ) from error
        with sdpa_kernel(backends=[flash_backend]):
            return run_sdpa()

    return run_fused_sdpa_only


def _version(distribution, module=None):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def _runtime_metadata(device, flash_module):
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_arch": getattr(
            properties,
            "gcnArchName",
            getattr(properties, "gcn_arch_name", "unknown"),
        ),
        "torch_version": str(torch.__version__),
        "torch_hip_version": torch.version.hip,
        "tilelang_version": _version("tilelang"),
        "flash_attn_version": _version("flash-attn", flash_module),
    }


def _report_path():
    configured = _CONFIG.report_path
    if configured is not None:
        return configured.expanduser()
    return Path(tempfile.gettempdir()) / "tilelang_flash_attention_forward_report.json"


_CSV_CASE_KEY_FIELDS = (
    "q_len",
    "kv_len",
    "batch",
    "heads",
    "head_dim",
    "dtype",
    "causal",
)

_CSV_FIELDNAMES = (
    "case_group",
    "device",
    "device_name",
    "device_arch",
    "torch_version",
    "torch_hip_version",
    "tilelang_version",
    "flash_attn_version",
    "q_len",
    "kv_len",
    "batch",
    "heads",
    "head_dim",
    "dtype",
    "causal",
    "backend",
    "status",
    "first_invocation_wall_ms",
    "warmup_steps",
    "repeat_steps",
    "device_time_median_ms",
    "device_time_p95_ms",
    "synchronized_wall_time_median_ms",
    "synchronized_wall_time_p95_ms",
    "tilelang_vs_flash_attn_device_speedup",
    "tilelang_vs_flash_attn_wall_speedup",
    "tilelang_vs_torch_sdpa_device_speedup",
    "tilelang_vs_torch_sdpa_wall_speedup",
    "accuracy_comparison",
    "accuracy_max_abs",
    "accuracy_rmse",
    "accuracy_relative_l2",
    "tilelang_vs_pytorch_reference_max_abs",
    "tilelang_vs_pytorch_reference_rmse",
    "tilelang_vs_pytorch_reference_relative_l2",
    "reason",
)


def _csv_report_path():
    configured = _CONFIG.csv_report_path
    if configured is not None:
        return configured.expanduser()
    return _report_path().with_suffix(".csv")


def _csv_case_fields(case):
    return {name: case.get(name, "") for name in _CSV_CASE_KEY_FIELDS}


def _csv_accuracy_fields(metrics, comparison):
    metrics = metrics or {}
    return {
        "accuracy_comparison": comparison if metrics else "",
        "accuracy_max_abs": metrics.get("max_abs", ""),
        "accuracy_rmse": metrics.get("rmse", ""),
        "accuracy_relative_l2": metrics.get("relative_l2", ""),
    }


def _csv_performance_row(
    case,
    backend,
    timing,
    *,
    status,
    reason="",
    accuracy=None,
    accuracy_name="",
    runtime=None,
):
    timing = timing if isinstance(timing, dict) else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    device_time = timing.get("device_time", {})
    wall_time = timing.get("synchronized_wall_time", {})
    flash_speedup = case.get("speedup_tilelang_vs_flash_attn", {})
    sdpa = case.get("torch_sdpa", {})
    sdpa_speedup = sdpa.get("tilelang_vs_sdpa_speedup", {})
    row = {
        "case_group": case.get("case_group", "benchmark"),
        "device": runtime.get("device", ""),
        "device_name": runtime.get("device_name", ""),
        "device_arch": runtime.get("device_arch", ""),
        "torch_version": runtime.get("torch_version", ""),
        "torch_hip_version": runtime.get("torch_hip_version", ""),
        "tilelang_version": runtime.get("tilelang_version", ""),
        "flash_attn_version": runtime.get("flash_attn_version", ""),
        **_csv_case_fields(case),
        "backend": backend,
        "status": status,
        "first_invocation_wall_ms": timing.get("first_invocation_wall_ms", ""),
        "warmup_steps": timing.get("warmup_steps", ""),
        "repeat_steps": timing.get("repeat_steps", ""),
        "device_time_median_ms": device_time.get("median_ms", ""),
        "device_time_p95_ms": device_time.get("p95_ms", ""),
        "synchronized_wall_time_median_ms": wall_time.get("median_ms", ""),
        "synchronized_wall_time_p95_ms": wall_time.get("p95_ms", ""),
        "tilelang_vs_flash_attn_device_speedup": flash_speedup.get(
            "device_time_median", ""
        ),
        "tilelang_vs_flash_attn_wall_speedup": flash_speedup.get(
            "synchronized_wall_time_median", ""
        ),
        "tilelang_vs_torch_sdpa_device_speedup": sdpa_speedup.get(
            "device_time_median", ""
        ),
        "tilelang_vs_torch_sdpa_wall_speedup": sdpa_speedup.get(
            "synchronized_wall_time_median", ""
        ),
        "reason": reason,
    }
    row.update(_csv_accuracy_fields(accuracy, accuracy_name))
    row.update(
        {
            "tilelang_vs_pytorch_reference_max_abs": "",
            "tilelang_vs_pytorch_reference_rmse": "",
            "tilelang_vs_pytorch_reference_relative_l2": "",
        }
    )
    return row


def _benchmark_csv_rows(report):
    rows = []
    runtime = report.get("runtime", {})
    for case in report.get("benchmark_cases", []):
        accuracy = case.get("accuracy", {})
        backend_specs = (
            (
                "tilelang",
                case.get("tilelang"),
                "measured",
                "",
                accuracy.get("tilelang_vs_flash_attn"),
                "TileLang vs flash-attn",
            ),
            (
                "flash_attn",
                case.get("flash_attn"),
                "measured",
                "",
                accuracy.get("flash_attn_vs_pytorch_reference"),
                "flash-attn vs PyTorch reference",
            ),
            (
                "torch_sdpa",
                case.get("torch_sdpa", {}),
                case.get("torch_sdpa", {}).get("status", "not_run"),
                case.get("torch_sdpa", {}).get("reason", ""),
                case.get("torch_sdpa", {}).get("tilelang_vs_sdpa_accuracy"),
                "TileLang vs PyTorch SDPA",
            ),
        )
        for backend, timing, status, reason, metrics, accuracy_name in backend_specs:
            row = _csv_performance_row(
                case,
                backend,
                timing,
                status=status,
                reason=reason,
                accuracy=metrics,
                accuracy_name=accuracy_name,
                runtime=runtime,
            )
            pytorch_metrics = accuracy.get("tilelang_vs_pytorch_reference", {})
            if backend == "tilelang" and pytorch_metrics:
                row["tilelang_vs_pytorch_reference_max_abs"] = pytorch_metrics.get(
                    "max_abs", ""
                )
                row["tilelang_vs_pytorch_reference_rmse"] = pytorch_metrics.get(
                    "rmse", ""
                )
                row["tilelang_vs_pytorch_reference_relative_l2"] = pytorch_metrics.get(
                    "relative_l2", ""
                )
            rows.append(row)

    seen_skips = set()
    # Correctness-only skips do not belong in the benchmark CSV. The dedicated
    # matrix contains only cases that would have been timed by --benchmark.
    skipped_rows = report.get("benchmark_matrix", {}).get("skipped_cases", [])
    for skipped in skipped_rows:
        group = skipped.get("group", "requested_stress_matrix")
        case_fields = _csv_case_fields(skipped)
        identity = (
            group,
            *(case_fields.get(name, "") for name in _CSV_CASE_KEY_FIELDS),
            skipped.get("reason", ""),
        )
        if identity in seen_skips:
            continue
        seen_skips.add(identity)
        for backend in ("tilelang", "flash_attn", "torch_sdpa"):
            row = _csv_performance_row(
                skipped,
                backend,
                {},
                status="skipped",
                reason=skipped.get("reason", ""),
                runtime=runtime,
            )
            row["case_group"] = group
            rows.append(row)
    return rows


def _write_csv_report(report, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else _csv_report_path()
    path = path.expanduser()
    json_path = _report_path().expanduser()
    if path.resolve() == json_path.resolve():
        raise ValueError("CSV and JSON report paths must be different")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=_CSV_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(_benchmark_csv_rows(report))
    return path


def _write_report(report):
    path = _report_path()
    csv_path = _csv_report_path() if _CONFIG.benchmark else None
    if csv_path is not None and csv_path.resolve() == path.resolve():
        raise ValueError("CSV and JSON report paths must be different")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if csv_path is not None:
        _write_csv_report(report, csv_path)
    return path


def _qkv_input_budget(device):
    """Keep Q/K/V inputs within a conservative share of available device RAM."""
    override = _CONFIG.max_qkv_input_bytes
    if override is not None:
        return override, {"source": "CLI override"}
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception as error:
        return _DEFAULT_MAX_QKV_INPUT_BYTES, {
            "source": "fallback default; mem_get_info failed",
            "reason": f"{type(error).__name__}: {error}",
            "default_bytes": _DEFAULT_MAX_QKV_INPUT_BYTES,
        }
    budget = max(
        1,
        min(
            int(total_bytes * _QKV_TOTAL_MEMORY_FRACTION),
            int(free_bytes * _QKV_FREE_MEMORY_FRACTION),
        ),
    )
    return budget, {
        "source": "minimum of 25% total memory and 50% currently free memory",
        "free_bytes_at_sweep_start": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _has_reasonable_reference_size(case):
    q_len, kv_len, batch, heads, _, _ = case
    return batch * heads * q_len * kv_len <= _MAX_REFERENCE_SCORE_ELEMENTS


class FlashAttentionReferenceTests(unittest.TestCase):
    def test_matches_torch_sdpa_forward(self):
        for causal in (False, True):
            with self.subTest(causal=causal):
                torch.manual_seed(21)
                q = torch.randn((2, 5, 2, 7), dtype=torch.float64)
                k = torch.randn((2, 5, 2, 7), dtype=torch.float64)
                v = torch.randn((2, 5, 2, 7), dtype=torch.float64)
                actual = pytorch_flash_attn_reference(q, k, v, causal=causal)
                expected = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=causal,
                ).transpose(1, 2)
                torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=2.0e-6)

    def test_bottom_right_causal_alignment_and_empty_rows(self):
        q = torch.zeros((1, 4, 1, 1), dtype=torch.float64)
        k = torch.zeros((1, 2, 1, 1), dtype=torch.float64)
        v = torch.tensor([1.0, 3.0], dtype=torch.float64).reshape(1, 2, 1, 1)
        output = pytorch_flash_attn_reference(q, k, v, causal=True)
        expected = torch.tensor([0.0, 0.0, 1.0, 2.0], dtype=torch.float64).reshape(
            1, 4, 1, 1
        )
        torch.testing.assert_close(output, expected)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_sdpa_helper_uses_bottom_right_causal_alignment(self):
        for q_len, kv_len in ((2, 4), (4, 2)):
            with self.subTest(q_len=q_len, kv_len=kv_len):
                torch.manual_seed(46 + q_len + kv_len)
                q = torch.randn((1, q_len, 1, 16), dtype=torch.float32)
                k = torch.randn((1, kv_len, 1, 16), dtype=torch.float32)
                v = torch.randn((1, kv_len, 1, 16), dtype=torch.float32)
                scale = 1.0 / math.sqrt(q.shape[-1])
                actual = _make_sdpa_forward(
                    q, k, v, scale=scale, causal=True
                )()
                expected = pytorch_flash_attn_reference(
                    q, k, v, softmax_scale=scale, causal=True
                )
                torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=2.0e-6)

    def test_p95_uses_nearest_rank(self):
        self.assertEqual(_percentile(list(range(1, 101)), 0.95), 95.0)

    def test_rejects_unsupported_options_before_device_dispatch(self):
        q = torch.empty((1, 2, 1, 64), dtype=torch.float16)
        with self.assertRaisesRegex(NotImplementedError, "dropout_p=0"):
            flash_attn_func(q, q, q, dropout_p=0.1)
        with self.assertRaisesRegex(NotImplementedError, "windowed"):
            flash_attn_func(q, q, q, window_size=(8, 8))
        with self.assertRaisesRegex(RuntimeError, "HIP/DCU"):
            flash_attn_func(q, q, q)


@unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
class FlashAttentionDeviceTests(unittest.TestCase):
    def test_forward_matches_pytorch_reference(self):
        cases = ((65, 65, False), (67, 83, True))
        for q_len, kv_len, causal in cases:
            with self.subTest(q_len=q_len, kv_len=kv_len, causal=causal):
                try:
                    q, k, v = _make_device_inputs(q_len, kv_len, torch.float16)
                    with torch.inference_mode():
                        output_tl = flash_attn_func(q, k, v, causal=causal)
                        output_ref = pytorch_flash_attn_reference(
                            q, k, v, causal=causal
                        )
                    _assert_output(self, "TileLang", output_tl, q.shape)
                    torch.testing.assert_close(
                        output_tl, output_ref, rtol=3.0e-2, atol=3.0e-2
                    )
                except Exception as error:
                    print(
                        "FlashAttention device case skipped; no data counted: "
                        f"q_len={q_len}, kv_len={kv_len}, causal={causal}, "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )

    def test_bfloat16_forward_when_supported(self):
        if not _bf16_supported():
            self.skipTest("the active DCU does not report bfloat16 support")
        try:
            q, k, v = _make_device_inputs(65, 67, torch.bfloat16)
            with torch.inference_mode():
                output_tl = flash_attn_func(q, k, v, causal=True)
                output_ref = pytorch_flash_attn_reference(q, k, v, causal=True)
            _assert_output(self, "TileLang BF16", output_tl, q.shape)
            torch.testing.assert_close(
                output_tl, output_ref, rtol=8.0e-2, atol=8.0e-2
            )
        except Exception as error:
            print(
                "FlashAttention BF16 case skipped; no data counted: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    def test_optional_open_source_flash_attn_forward_parity(self):
        reference, reason = _open_source_flash_attn()
        if reference is None:
            self.skipTest(reason)
        q, k, v = _make_device_inputs(65, 65, torch.float16)
        try:
            with torch.inference_mode():
                output_tl = flash_attn_func(q, k, v, causal=True)
                output_ref = reference(q, k, v, dropout_p=0.0, causal=True)
                torch.cuda.synchronize(q.device)
        except Exception as error:
            self.skipTest(
                "this TileLang/flash_attn case is unusable on the active target; "
                f"no comparison data counted: {type(error).__name__}: {error}"
            )
        try:
            _assert_output(self, "TileLang", output_tl, q.shape)
            _assert_output(self, "flash_attn", output_ref, q.shape)
            torch.testing.assert_close(
                output_tl, output_ref, rtol=3.0e-2, atol=3.0e-2
            )
        except Exception as error:
            self.skipTest(
                "TileLang/flash_attn parity failed; no comparison data counted: "
                f"{type(error).__name__}: {error}"
            )


class FlashAttentionExtendedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (_CONFIG.benchmark or _CONFIG.coverage):
            raise unittest.SkipTest("pass --benchmark and/or --coverage to enable extended tests")
        super().setUpClass()

    def _require_runtime_and_reference(self):
        if not _HAS_TILELANG_HIP:
            self.skipTest(
                "extended FlashAttention tests require TileLang and a HIP/DCU device"
            )
        reference, reason = _open_source_flash_attn()
        if reference is None:
            self.skipTest(
                "open-source flash_attn is unavailable; no comparison data counted: "
                f"{reason}"
            )
        return reference

    def _assert_pair(self, label, q, output_tl, output_ref, dtype):
        _assert_output(self, f"TileLang {label}", output_tl, q.shape)
        _assert_output(self, f"flash_attn {label}", output_ref, q.shape)
        tolerance = _tolerance(dtype)
        torch.testing.assert_close(output_tl, output_ref, **tolerance)
        return _error_metrics(output_ref, output_tl)

    def _correctness_case(self, reference, case, dtype, seed):
        q_len, kv_len, batch, heads, head_dim, causal = case
        torch.manual_seed(seed)
        q, k, v = _make_device_inputs(
            q_len,
            kv_len,
            dtype,
            batch=batch,
            heads=heads,
            head_dim=head_dim,
        )
        scale = 1.0 / math.sqrt(head_dim)
        with torch.inference_mode():
            output_tl = flash_attn_func(
                q, k, v, softmax_scale=scale, causal=causal
            )
            output_fa = reference(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=causal,
            )
            torch.cuda.synchronize(q.device)
            metrics = self._assert_pair(
                "vs flash_attn", q, output_tl, output_fa, dtype
            )
            row = {
                **_case_dict(case, dtype),
                "comparison": "TileLang vs open-source flash_attn",
                "tilelang_vs_flash_attn": metrics,
                "tolerance": _tolerance(dtype),
                "counted_in_metrics": True,
            }
            if _has_reasonable_reference_size(case):
                output_torch = pytorch_flash_attn_reference(
                    q, k, v, softmax_scale=scale, causal=causal
                )
                torch_metrics = self._assert_pair(
                    "vs explicit PyTorch reference", q, output_tl, output_torch, dtype
                )
                fa_metrics = _error_metrics(output_torch, output_fa)
                row["tilelang_vs_pytorch_reference"] = torch_metrics
                row["flash_attn_vs_pytorch_reference"] = fa_metrics
            else:
                row["pytorch_reference"] = (
                    "not run: explicit reference score matrix exceeds configured "
                    "memory budget"
                )
        return row

    def _benchmark_case(self, reference, case, dtype, seed):
        q_len, kv_len, batch, heads, head_dim, causal = case
        torch.manual_seed(seed)
        q, k, v = _make_device_inputs(
            q_len,
            kv_len,
            dtype,
            batch=batch,
            heads=heads,
            head_dim=head_dim,
        )
        device = q.device
        scale = 1.0 / math.sqrt(head_dim)
        warmup, repeats = _benchmark_steps(
            q_len,
            kv_len,
            batch=batch,
            heads=heads,
            head_dim=head_dim,
        )

        def tilelang_forward():
            return flash_attn_func(
                q, k, v, softmax_scale=scale, causal=causal
            )

        def flash_forward():
            return reference(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=causal,
            )

        with torch.inference_mode():
            output_tl, tilelang_timing = _measure_forward(
                tilelang_forward, device, warmup, repeats
            )
            output_fa, flash_timing = _measure_forward(
                flash_forward, device, warmup, repeats
            )
            accuracy = self._assert_pair(
                "benchmark vs flash_attn", q, output_tl, output_fa, dtype
            )
            accuracy_row = {
                **_case_dict(case, dtype),
                "comparison": "TileLang vs open-source flash_attn",
                "tilelang_vs_flash_attn": accuracy,
                "tolerance": _tolerance(dtype),
                "counted_in_metrics": True,
            }
            if _has_reasonable_reference_size(case):
                output_torch_ref = pytorch_flash_attn_reference(
                    q, k, v, softmax_scale=scale, causal=causal
                )
                accuracy_row["tilelang_vs_pytorch_reference"] = self._assert_pair(
                    "benchmark vs explicit PyTorch reference",
                    q,
                    output_tl,
                    output_torch_ref,
                    dtype,
                )
                accuracy_row["flash_attn_vs_pytorch_reference"] = _error_metrics(
                    output_torch_ref, output_fa
                )
            else:
                accuracy_row["pytorch_reference"] = (
                    "not run: explicit reference score matrix exceeds configured "
                    "memory budget"
                )

            benchmark_row = {
                **_case_dict(case, dtype),
                "counted_in_metrics": True,
                "tilelang": tilelang_timing,
                "flash_attn": flash_timing,
                "speedup_tilelang_vs_flash_attn": {
                    "device_time_median": (
                        flash_timing["device_time"]["median_ms"]
                        / tilelang_timing["device_time"]["median_ms"]
                    ),
                    "synchronized_wall_time_median": (
                        flash_timing["synchronized_wall_time"]["median_ms"]
                        / tilelang_timing["synchronized_wall_time"]["median_ms"]
                    ),
                },
                "torch_sdpa": {"status": "not_run"},
                "accuracy": {
                    key: accuracy_row[key]
                    for key in (
                        "tilelang_vs_flash_attn",
                        "tilelang_vs_pytorch_reference",
                        "flash_attn_vs_pytorch_reference",
                    )
                    if key in accuracy_row
                },
            }

            sdpa_forward = _make_sdpa_forward(q, k, v, scale=scale, causal=causal)
            try:
                output_sdpa, sdpa_timing = _measure_forward(
                    sdpa_forward, device, warmup, repeats
                )
            except (RuntimeError, NotImplementedError) as error:
                benchmark_row["torch_sdpa"] = {
                    "status": "unsupported_or_failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            else:
                _assert_output(self, "PyTorch SDPA", output_sdpa, q.shape)
                torch.testing.assert_close(
                    output_tl, output_sdpa, **_tolerance(dtype)
                )
                benchmark_row["torch_sdpa"] = {
                    "status": "measured",
                    **sdpa_timing,
                    "tilelang_vs_sdpa_accuracy": _error_metrics(
                        output_tl, output_sdpa
                    ),
                    "tilelang_vs_sdpa_speedup": {
                        "device_time_median": (
                            sdpa_timing["device_time"]["median_ms"]
                            / tilelang_timing["device_time"]["median_ms"]
                        ),
                        "synchronized_wall_time_median": (
                            sdpa_timing["synchronized_wall_time"]["median_ms"]
                            / tilelang_timing["synchronized_wall_time"]["median_ms"]
                        ),
                    },
                }

        return accuracy_row, benchmark_row

    def _record_skipped(self, report, case, dtype, error, *, group):
        message = f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error)
        row = {
            **_case_dict(case, dtype),
            "status": "skipped",
            "group": group,
            "reason": message[:1000],
            "counted_in_metrics": False,
        }
        report["skipped_cases"].append(row)
        _write_report(report)
        print("FlashAttention case skipped: " + json.dumps(row, sort_keys=True), flush=True)

    def _requested_stress_sweep(self, reference, report, device):
        """Run the original broad matrix for accuracy coverage only."""
        max_input_bytes, memory_policy = _qkv_input_budget(device)
        sweep = report["requested_matrix"]
        sweep["max_qkv_input_bytes"] = max_input_bytes
        sweep["memory_budget_policy"] = memory_policy

        seed = 20264000
        for case in _requested_stress_cases():
            input_bytes = _estimated_qkv_input_bytes(case)
            if input_bytes > max_input_bytes:
                row = {
                    **_case_dict(case, torch.float16),
                    "status": "skipped",
                    "group": "requested_stress_matrix",
                    "reason": (
                        f"estimated Q/K/V input size {input_bytes} bytes exceeds "
                        f"available-memory budget {max_input_bytes} bytes"
                    ),
                    "estimated_qkv_input_bytes": input_bytes,
                    "counted_in_metrics": False,
                }
                sweep["skipped_cases"].append(row)
                report["skipped_cases"].append(row)
                _write_report(report)
                seed += 1
                continue

            try:
                accuracy_row = self._correctness_case(
                    reference,
                    case,
                    torch.float16,
                    seed=seed,
                )
            except Exception as error:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                row = {
                    **_case_dict(case, torch.float16),
                    "status": "skipped",
                    "group": "requested_stress_matrix",
                    "reason": f"{type(error).__name__}: {error}"[:1000],
                    "estimated_qkv_input_bytes": input_bytes,
                    "counted_in_metrics": False,
                }
                sweep["skipped_cases"].append(row)
                report["skipped_cases"].append(row)
                print(
                    "FlashAttention correctness case skipped: "
                    + json.dumps(row, sort_keys=True),
                    flush=True,
                )
            else:
                accuracy_row["estimated_qkv_input_bytes"] = input_bytes
                accuracy_row["counted_in_metrics"] = True
                sweep["successful_cases"].append(accuracy_row)
                print(
                    "FlashAttention correctness matrix: "
                    + json.dumps(accuracy_row, sort_keys=True),
                    flush=True,
                )
            _write_report(report)
            seed += 1

        sweep["requested_case_count"] = len(_requested_stress_cases())
        sweep["successful_case_count"] = len(sweep["successful_cases"])
        sweep["skipped_case_count"] = len(sweep["skipped_cases"])
        skip_reasons = {}
        for row in sweep["skipped_cases"]:
            reason = row["reason"]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        sweep["skip_reason_counts"] = skip_reasons
        _write_report(report)
        print(
            "Requested correctness matrix totals: "
            f"requested={sweep['requested_case_count']}, "
            f"successful={sweep['successful_case_count']}, "
            f"skipped={sweep['skipped_case_count']} (skips excluded from metrics)",
            flush=True,
        )

    def _requested_benchmark_sweep(self, reference, report, device):
        """Benchmark only the fixed-head/fixed-dimension 15-case matrix."""
        max_input_bytes, memory_policy = _qkv_input_budget(device)
        matrix = report["benchmark_matrix"]
        matrix["max_qkv_input_bytes"] = max_input_bytes
        matrix["memory_budget_policy"] = memory_policy

        for index, case in enumerate(_requested_benchmark_cases()):
            input_bytes = _estimated_qkv_input_bytes(case)
            if input_bytes > max_input_bytes:
                row = {
                    **_case_dict(case, torch.float16),
                    "status": "skipped",
                    "group": "requested_benchmark_matrix",
                    "reason": (
                        f"estimated Q/K/V input size {input_bytes} bytes exceeds "
                        f"available-memory budget {max_input_bytes} bytes"
                    ),
                    "estimated_qkv_input_bytes": input_bytes,
                    "counted_in_metrics": False,
                }
                matrix["skipped_cases"].append(row)
                report["skipped_cases"].append(row)
                _write_report(report)
                continue

            try:
                _accuracy_row, benchmark_row = self._benchmark_case(
                    reference,
                    case,
                    torch.float16,
                    seed=20264000 + index,
                )
            except Exception as error:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                row = {
                    **_case_dict(case, torch.float16),
                    "status": "skipped",
                    "group": "requested_benchmark_matrix",
                    "reason": f"{type(error).__name__}: {error}"[:1000],
                    "estimated_qkv_input_bytes": input_bytes,
                    "counted_in_metrics": False,
                }
                matrix["skipped_cases"].append(row)
                report["skipped_cases"].append(row)
                print(
                    "FlashAttention benchmark case skipped: "
                    + json.dumps(row, sort_keys=True),
                    flush=True,
                )
                _write_report(report)
                continue

            benchmark_row["case_group"] = "requested_benchmark_matrix"
            benchmark_row["estimated_qkv_input_bytes"] = input_bytes
            benchmark_row["counted_in_metrics"] = True
            report["benchmark_cases"].append(benchmark_row)
            matrix["successful_cases"].append(benchmark_row)
            _write_report(report)
            print(
                "FlashAttention benchmark: "
                + json.dumps(benchmark_row, sort_keys=True),
                flush=True,
            )

        matrix["requested_case_count"] = len(_requested_benchmark_cases())
        matrix["successful_case_count"] = len(matrix["successful_cases"])
        matrix["skipped_case_count"] = len(matrix["skipped_cases"])
        skip_reasons = {}
        for row in matrix["skipped_cases"]:
            reason = row["reason"]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        matrix["skip_reason_counts"] = skip_reasons
        _write_report(report)
        print(
            "Requested benchmark matrix totals: "
            f"requested={matrix['requested_case_count']}, "
            f"successful={matrix['successful_case_count']}, "
            f"skipped={matrix['skipped_case_count']} "
            "(correctness-only cases are not benchmarked)",
            flush=True,
        )

    def test_opt_in_forward_coverage_and_long_sequence_benchmark(self):
        reference = self._require_runtime_and_reference()
        device = torch.device("cuda")
        try:
            flash_module = __import__("flash_attn")
        except Exception:
            flash_module = None
        report = {
            "scope": "forward-only dense attention; no backward is tested",
            "runtime": _runtime_metadata(device, flash_module),
            "long_sequence_lengths": list(_LONG_SEQUENCE_LENGTHS),
            "reference_score_element_limit": _MAX_REFERENCE_SCORE_ELEMENTS,
            "torch_sdpa_math_max_sequence": _TORCH_SDPA_MATH_MAX_SEQUENCE,
            "long_sequence_sdpa_policy": (
                "above the math-fallback limit, attempt flash-only SDPA and "
                "report unsupported rather than risk quadratic-memory fallback"
            ),
            "benchmark_notes": {
                "shape_policy": (
                    "heads=8 and head_dim=128 are fixed; only batch and equal "
                    "query/key sequence length vary"
                ),
                "first_invocation": (
                    "synchronized wall time inside this process; may reuse JIT or "
                    "disk caches and is not guaranteed to be a cold compile"
                ),
                "device_time": "CUDA/HIP event elapsed time",
                "synchronized_wall_time": (
                    "host wall time for one call with device synchronization"
                ),
                "p95": "nearest-rank percentile",
                "long_repeat_policy": (
                    "adaptive by estimated workload; override with --bench-repeats "
                    "or --bench-warmup"
                ),
            },
            "coverage_cases": [],
            "long_sequence_cases": [],
            "benchmark_cases": [],
            "supplemental_correctness_cases": [],
            "skipped_cases": [],
            "requested_matrix": {
                "purpose": "broad forward correctness matrix; never timed",
                "axes": {
                    "heads": list(_REQUESTED_HEAD_COUNTS),
                    "batch": list(_REQUESTED_BATCH_SIZES),
                    "head_dim": list(_REQUESTED_HEAD_DIMS),
                    "sequence_lengths": list(_LONG_SEQUENCE_LENGTHS),
                    "dtype": "torch.float16",
                    "causal": False,
                },
                "successful_cases": [],
                "skipped_cases": [],
                "requested_case_count": 0,
                "successful_case_count": 0,
                "skipped_case_count": 0,
            },
            "benchmark_matrix": {
                "enabled": _CONFIG.benchmark,
                "axes": _requested_benchmark_axes(),
                "successful_cases": [],
                "skipped_cases": [],
                "requested_case_count": (
                    len(_requested_benchmark_cases()) if _CONFIG.benchmark else 0
                ),
                "successful_case_count": 0,
                "skipped_case_count": 0,
            },
            "configuration": {
                "coverage_enabled": _CONFIG.coverage,
                "benchmark_enabled": _CONFIG.benchmark,
                "csv_report_path": (
                    str(_csv_report_path()) if _CONFIG.benchmark else None
                ),
                "bf16_supported": _bf16_supported(),
            },
        }
        report_path = _write_report(report)

        if _CONFIG.coverage:
            coverage = [(case, torch.float16) for case in _FORWARD_COVERAGE_CASES]
            if _bf16_supported():
                coverage.extend(
                    (case, torch.bfloat16) for case in _BF16_COVERAGE_CASES
                )
            for index, (case, dtype) in enumerate(coverage):
                try:
                    row = self._correctness_case(
                        reference, case, dtype, seed=20261109 + index
                    )
                except Exception as error:
                    self._record_skipped(
                        report, case, dtype, error, group="forward_coverage"
                    )
                    continue
                report["coverage_cases"].append(row)
                _write_report(report)
                print(
                    "FlashAttention coverage: "
                    + json.dumps(row, sort_keys=True),
                    flush=True,
                )

        if _CONFIG.coverage:
            for index, sequence_length in enumerate(_LONG_SEQUENCE_LENGTHS):
                case = (sequence_length, sequence_length, 1, 1, 64, False)
                try:
                    accuracy_row = self._correctness_case(
                        reference,
                        case,
                        torch.float16,
                        seed=20262000 + index,
                    )
                except Exception as error:
                    self._record_skipped(
                        report,
                        case,
                        torch.float16,
                        error,
                        group="supported_long_sweep_correctness",
                    )
                    continue
                report["long_sequence_cases"].append(accuracy_row)
                _write_report(report)
                print(
                    "FlashAttention long-sequence correctness: "
                    + json.dumps(accuracy_row, sort_keys=True),
                    flush=True,
                )

            for index, case in enumerate(_SUPPLEMENTAL_CORRECTNESS_CASES):
                try:
                    accuracy_row = self._correctness_case(
                        reference,
                        case,
                        torch.float16,
                        seed=20263000 + index,
                    )
                except Exception as error:
                    self._record_skipped(
                        report,
                        case,
                        torch.float16,
                        error,
                        group="supplemental_correctness",
                    )
                    continue
                report["supplemental_correctness_cases"].append(accuracy_row)
                _write_report(report)
                print(
                    "FlashAttention supplemental correctness: "
                    + json.dumps(accuracy_row, sort_keys=True),
                    flush=True,
                )

            self._requested_stress_sweep(reference, report, device)

        if _CONFIG.benchmark:
            self._requested_benchmark_sweep(reference, report, device)

        report_path = _write_report(report)
        print(f"FlashAttention JSON report: {report_path}", flush=True)
        if _CONFIG.benchmark:
            print(f"FlashAttention CSV report: {_csv_report_path()}", flush=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    global _CONFIG
    _CONFIG = _config_from_args(args)
    program = unittest.main(
        module=sys.modules[__name__],
        argv=[sys.argv[0]],
        exit=False,
        verbosity=2,
    )
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
