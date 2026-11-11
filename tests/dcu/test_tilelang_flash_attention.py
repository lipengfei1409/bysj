"""TileLang dense FlashAttention forward parity and opt-in benchmark.

Run the quick correctness tests with unittest discovery (configuration defaults
are used)::

    python -m unittest discover -s tests/dcu \
        -p 'test_tilelang_flash_attention.py' -v

Pass benchmark/coverage settings as CLI options when running this file directly::

    python tests/dcu/test_tilelang_flash_attention.py \
        --coverage --benchmark --bench-warmup 1 --bench-repeats 2 \
        --max-qkv-input-bytes 8589934592 --report-path /tmp/fa-report.json

The requested stress axes are heads 8/16/32, batch 1/4/16, and head_dim
64/128 crossed with sequence lengths 1k through 16k. Cases that error or exceed
the memory budget are recorded as skipped and excluded from metrics. The budget
reserves device memory for outputs, workspaces, and other processes. The test
covers the dense ``flash_attn_func`` forward contract only: zero dropout, no GQA,
windowing, ALiBi, softcap, or returned attention probabilities. These tests do
not exercise backward.
"""

from __future__ import annotations

import argparse
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
    max_qkv_input_bytes: int | None = None


_CONFIG = FlashAttentionConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TileLang FlashAttention correctness, coverage, and benchmark tests."
    )
    parser.add_argument("--benchmark", action="store_true", help="run long-sequence benchmarks")
    parser.add_argument("--coverage", action="store_true", help="run forward precision coverage")
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
        self.assertIsNone(config.max_qkv_input_bytes)

    def test_cli_options_populate_all_config_fields(self):
        args = build_parser().parse_args(
            [
                "--benchmark",
                "--coverage",
                "--bench-repeats", "2",
                "--bench-warmup", "1",
                "--report-path", "~/flash-attn-report.json",
                "--max-qkv-input-bytes", "4096",
            ]
        )
        config = _config_from_args(args)
        self.assertTrue(config.benchmark)
        self.assertTrue(config.coverage)
        self.assertEqual(config.bench_repeats, 2)
        self.assertEqual(config.bench_warmup, 1)
        self.assertEqual(config.report_path, Path("~/flash-attn-report.json"))
        self.assertEqual(config.max_qkv_input_bytes, 4096)

    def test_numeric_overrides_keep_existing_clamping(self):
        args = build_parser().parse_args(
            ["--bench-repeats", "0", "--bench-warmup", "-2", "--max-qkv-input-bytes", "0"]
        )
        config = _config_from_args(args)
        self.assertEqual(config.bench_repeats, 1)
        self.assertEqual(config.bench_warmup, 0)
        self.assertEqual(config.max_qkv_input_bytes, 1)


_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)
_LONG_SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384)
_REQUESTED_HEAD_COUNTS = (8, 16, 32)
_REQUESTED_BATCH_SIZES = (1, 4, 16)
_REQUESTED_HEAD_DIMS = (64, 128)
_DEFAULT_MAX_QKV_INPUT_BYTES = 16 * 1024 * 1024 * 1024
_QKV_TOTAL_MEMORY_FRACTION = 0.25
_QKV_FREE_MEMORY_FRACTION = 0.50
# Keep a few representative non-square and multi-head measurements in addition
# to the requested long, equal-length sweep.
_EXTRA_BENCHMARK_CASES = (
    (512, 512, 1, 8, 64, False),
    (512, 512, 1, 8, 64, True),
    (2048, 4096, 1, 4, 64, True),
)
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


def _write_report(report):
    path = _report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
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
        max_input_bytes, memory_policy = _qkv_input_budget(device)
        sweep = report["requested_matrix"]
        sweep["max_qkv_input_bytes"] = max_input_bytes
        sweep["memory_budget_policy"] = memory_policy

        seed = 20264000
        for sequence_length in _LONG_SEQUENCE_LENGTHS:
            for batch in _REQUESTED_BATCH_SIZES:
                for heads in _REQUESTED_HEAD_COUNTS:
                    for head_dim in _REQUESTED_HEAD_DIMS:
                        case = (
                            sequence_length,
                            sequence_length,
                            batch,
                            heads,
                            head_dim,
                            False,
                        )
                        input_bytes = (
                            batch
                            * heads
                            * head_dim
                            * (sequence_length + 2 * sequence_length)
                            * torch.empty((), dtype=torch.float16).element_size()
                        )
                        reason = None
                        if input_bytes > max_input_bytes:
                            reason = (
                                f"estimated Q/K/V input size {input_bytes} bytes "
                                f"exceeds available-memory budget {max_input_bytes} bytes"
                            )

                        if reason is not None:
                            row = {
                                **_case_dict(case, torch.float16),
                                "status": "skipped",
                                "group": "requested_stress_matrix",
                                "reason": reason[:1000],
                                "estimated_qkv_input_bytes": input_bytes,
                                "counted_in_metrics": False,
                            }
                            sweep["skipped_cases"].append(row)
                            continue

                        try:
                            if _CONFIG.benchmark:
                                accuracy_row, benchmark_row = self._benchmark_case(
                                    reference,
                                    case,
                                    torch.float16,
                                    seed=seed,
                                )
                                sweep["successful_cases"].append(accuracy_row)
                                report["benchmark_cases"].append(benchmark_row)
                            else:
                                accuracy_row = self._correctness_case(
                                    reference,
                                    case,
                                    torch.float16,
                                    seed=seed,
                                )
                                sweep["successful_cases"].append(accuracy_row)
                        except Exception as error:
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            row = {
                                **_case_dict(case, torch.float16),
                                "status": "skipped",
                                "group": "requested_stress_matrix",
                                "reason": (
                                    f"{type(error).__name__}: {error}"
                                )[:1000],
                                "estimated_qkv_input_bytes": input_bytes,
                                "counted_in_metrics": False,
                            }
                            sweep["skipped_cases"].append(row)
                            report["skipped_cases"].append(row)
                            print(
                                "FlashAttention case skipped: "
                                + json.dumps(row, sort_keys=True),
                                flush=True,
                            )
                        else:
                            row = sweep["successful_cases"][-1]
                            row["estimated_qkv_input_bytes"] = input_bytes
                            row["counted_in_metrics"] = True
                            _write_report(report)
                            print(
                                "FlashAttention requested sweep: "
                                + json.dumps(row, sort_keys=True),
                                flush=True,
                            )
                        seed += 1

        sweep["requested_case_count"] = (
            len(_LONG_SEQUENCE_LENGTHS)
            * len(_REQUESTED_BATCH_SIZES)
            * len(_REQUESTED_HEAD_COUNTS)
            * len(_REQUESTED_HEAD_DIMS)
        )
        sweep["successful_case_count"] = len(sweep["successful_cases"])
        sweep["skipped_case_count"] = len(sweep["skipped_cases"])
        skip_reasons = {}
        for row in sweep["skipped_cases"]:
            reason = row["reason"]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        sweep["skip_reason_counts"] = skip_reasons
        _write_report(report)
        print(
            "Requested stress matrix totals: "
            f"requested={sweep['requested_case_count']}, "
            f"successful={sweep['successful_case_count']}, "
            f"skipped={sweep['skipped_case_count']} (skips excluded from metrics)",
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
            "auxiliary_benchmark_accuracy_cases": [],
            "skipped_cases": [],
            "requested_matrix": {
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
            "configuration": {
                "coverage_enabled": _CONFIG.coverage,
                "benchmark_enabled": _CONFIG.benchmark,
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

        for index, sequence_length in enumerate(_LONG_SEQUENCE_LENGTHS):
            case = (sequence_length, sequence_length, 1, 1, 64, False)
            try:
                if _CONFIG.benchmark:
                    accuracy_row, benchmark_row = self._benchmark_case(
                        reference,
                        case,
                        torch.float16,
                        seed=20262000 + index,
                    )
                else:
                    accuracy_row = self._correctness_case(
                        reference,
                        case,
                        torch.float16,
                        seed=20262000 + index,
                    )
            except Exception as error:
                self._record_skipped(
                    report, case, torch.float16, error, group="supported_long_sweep"
                )
                continue
            report["long_sequence_cases"].append(accuracy_row)
            if _CONFIG.benchmark:
                report["benchmark_cases"].append(benchmark_row)
            _write_report(report)
            print(
                "FlashAttention long sequence: "
                + json.dumps(accuracy_row, sort_keys=True),
                flush=True,
            )
            if _CONFIG.benchmark:
                print(
                    "FlashAttention benchmark: "
                    + json.dumps(benchmark_row, sort_keys=True),
                    flush=True,
                )

        if _CONFIG.benchmark:
            for index, case in enumerate(_EXTRA_BENCHMARK_CASES):
                try:
                    accuracy_row, benchmark_row = self._benchmark_case(
                        reference,
                        case,
                        torch.float16,
                        seed=20263000 + index,
                    )
                except Exception as error:
                    self._record_skipped(
                        report, case, torch.float16, error, group="auxiliary_benchmark"
                    )
                    continue
                report["auxiliary_benchmark_accuracy_cases"].append(accuracy_row)
                report["benchmark_cases"].append(benchmark_row)
                _write_report(report)
                print(
                    "FlashAttention benchmark: "
                    + json.dumps(benchmark_row, sort_keys=True),
                    flush=True,
                )

        self._requested_stress_sweep(reference, report, device)
        report_path = _write_report(report)
        print(f"FlashAttention JSON report: {report_path}", flush=True)


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
