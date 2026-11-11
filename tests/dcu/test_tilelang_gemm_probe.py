"""TileLang dense GEMM device/efficiency probe for HIP/DCU.

Run from the repository root on a HIP/DCU host::

    python tests/dcu/test_tilelang_gemm_probe.py --m 2048 --n 2048 --k 2048

Or run the smoke test through unittest discovery::

    python -m unittest discover -s tests/dcu -p 'test_tilelang_gemm_probe.py' -v

The probe uses TileLang's explicit ``target="hip"`` JIT backend, reports the
selected DCU, validates against same-device ``torch.matmul``, and times warmed
kernel launches with HIP-backed PyTorch CUDA events. It is test-only and does
not call the production DUMMA GEMM extension.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
import time
import unittest
from importlib import metadata
from typing import Any

import torch

from tilelang_ops.dense_gemm import tilelang_dense_gemm

_HAS_HIP_DEVICE = torch.cuda.is_available() and torch.version.hip is not None


def _version(distribution: str, module: Any) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _measure(
    function,
    device: torch.device,
    *,
    warmup: int,
    samples: int,
    repeats_per_sample: int,
):
    """Separate first-call wall time from synchronized device-event samples."""
    _synchronize(device)
    started = time.perf_counter()
    first_output = function()
    _synchronize(device)
    first_call_wall_ms = (time.perf_counter() - started) * 1000.0

    for _ in range(warmup):
        function()
    _synchronize(device)

    sample_ms = []
    for _ in range(samples):
        _synchronize(device)
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        for _ in range(repeats_per_sample):
            function()
        event_end.record()
        event_end.synchronize()
        sample_ms.append(event_start.elapsed_time(event_end) / repeats_per_sample)

    return first_output, {
        "first_call_wall_ms": first_call_wall_ms,
        "warmup_calls": warmup,
        "samples": [float(value) for value in sample_ms],
        "repeats_per_sample": repeats_per_sample,
        "median_device_ms": float(statistics.median(sample_ms)),
        "min_device_ms": float(min(sample_ms)),
        "max_device_ms": float(max(sample_ms)),
        "device_event_timing": True,
    }


def run_probe(
    *,
    m: int,
    n: int,
    k: int,
    dtype_name: str = "float16",
    device_index: int = 0,
    warmup: int = 3,
    samples: int = 3,
    repeats_per_sample: int = 10,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("this probe requires HIP-enabled PyTorch and a visible DCU")
    if m < 1 or n < 1 or k < 1:
        raise ValueError("M, N, and K must be positive")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError(
            f"device index {device_index} is invalid; "
            f"{torch.cuda.device_count()} DCU device(s) are visible"
        )
    if warmup < 0 or samples < 1 or repeats_per_sample < 1:
        raise ValueError("warmup must be nonnegative; samples and repeats must be positive")
    if dtype_name not in ("float16", "bfloat16"):
        raise ValueError("--dtype must be float16 or bfloat16")

    try:
        tilelang_module = importlib.import_module("tilelang")
    except Exception as error:
        raise RuntimeError(
            "HIP is available but TileLang could not be imported; "
            "the DCU probe cannot pass without TileLang"
        ) from error

    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    torch_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    torch.manual_seed(2026)
    with torch.cuda.device(device), torch.inference_mode():
        a = torch.randn((m, k), device=device, dtype=torch_dtype).contiguous()
        b = torch.randn((k, n), device=device, dtype=torch_dtype).contiguous()
        for name, tensor in (("A", a), ("B", b)):
            if tensor.device != device or not tensor.is_contiguous():
                raise AssertionError(f"{name} is not a contiguous tensor on {device}")

        output, timing = _measure(
            lambda: tilelang_dense_gemm(a, b),
            device,
            warmup=warmup,
            samples=samples,
            repeats_per_sample=repeats_per_sample,
        )
        _synchronize(device)
        reference = torch.matmul(a, b)
        torch.testing.assert_close(
            output,
            reference,
            rtol=4.0e-2 if dtype_name == "bfloat16" else 2.0e-2,
            atol=1.0e-1 if dtype_name == "bfloat16" else 5.0e-2,
        )
        if output.device != device or output.shape != (m, n):
            raise AssertionError(
                f"TileLang output device/shape mismatch: {output.device}, {tuple(output.shape)}"
            )
        if timing["median_device_ms"] <= 0.0:
            raise AssertionError("GPU event timing was not positive")

        median_ms = timing["median_device_ms"]
        tflops = (2.0 * m * n * k) / (median_ms * 1.0e9)
        error = (output.float() - reference.float()).abs()
        report = {
            "backend": "TileLang HIP",
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "architecture": str(getattr(properties, "gcnArchName", "unknown")),
            "hip_version": str(torch.version.hip),
            "torch_version": str(torch.__version__),
            "tilelang_version": _version("tilelang", tilelang_module),
            "shape": {"m": m, "n": n, "k": k},
            "dtype": dtype_name,
            "input_device": str(a.device),
            "output_device": str(output.device),
            "max_abs_error": float(error.max().item()),
            "timing": timing,
            "effective_tflops": float(tflops),
        }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile, validate, and time a TileLang dense GEMM on a HIP/DCU device."
    )
    parser.add_argument("--m", type=int, default=2048, help="rows of A and C")
    parser.add_argument("--n", type=int, default=2048, help="columns of B and C")
    parser.add_argument("--k", type=int, default=2048, help="inner GEMM dimension")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repeats-per-sample", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_probe(
            m=args.m,
            n=args.n,
            k=args.k,
            dtype_name=args.dtype,
            device_index=args.device_index,
            warmup=args.warmup,
            samples=args.samples,
            repeats_per_sample=args.repeats_per_sample,
        )
    except Exception as error:
        print(f"TileLang GEMM probe failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


@unittest.skipUnless(_HAS_HIP_DEVICE, "requires a PyTorch HIP/DCU device")
class TileLangDenseGemmProbeTests(unittest.TestCase):
    def test_dense_gemm_runs_on_dcu(self):
        report = run_probe(
            m=128,
            n=96,
            k=80,
            dtype_name="float16",
            warmup=1,
            samples=2,
            repeats_per_sample=2,
        )
        self.assertEqual(report["output_device"], report["device"])
        self.assertGreater(report["timing"]["median_device_ms"], 0.0)
        self.assertTrue(math.isfinite(report["effective_tflops"]))


if __name__ == "__main__":
    raise SystemExit(main())
