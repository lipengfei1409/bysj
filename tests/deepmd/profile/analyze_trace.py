"""Attribute a DeepMD profiling trace to kernel families, shapes and call stacks.

The profile CLI already writes a 40-row hotspot table, but a DCU tuning run
needs three things it does not answer:

1. *Which share of the device time is Triton at all.*  If no Triton kernel
   appears, the run measured the dense reference path and every tuning
   conclusion drawn from it is void.
2. *What each kernel would have to move / compute to hit the roofline.*  The
   "equivalent MB" column is the gate for "is this kernel worth rewriting":
   a memory-bound kernel moving hundreds of MB per call is a fusion target,
   not a tile-config target.
3. *Where the interface casts come from.*  ``aten::to`` / ``_to_copy`` /
   ``copy_`` can dominate the CPU timeline; the profiler records the Python
   call stack for them when ``with_stack`` is on, which turns "casts are
   expensive" into a list of file:line edit points.

Usage::

    python tests/deepmd/profile/analyze_trace.py \
        /tmp/wk/np_lv2c0/dpa4_*/dpa4_inference_trace.json --json /tmp/wk/lv2.json

Duration caveat: summed event durations overlap and are not the inference
latency; use them for attribution only.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

_DEVICE_CATEGORIES = ("kernel", "gpu_kernel", "hip_kernel", "cuda_kernel")
_CAST_OPS = (
    "aten::to",
    "aten::_to_copy",
    "aten::copy_",
    "aten::clone",
    "aten::contiguous",
    "aten::type_as",
)

# SeZM/DPA1 Triton modules whose JIT functions can appear in a trace.  The
# kernel names are read from the installed DeepMD instead of being guessed
# from the trace: ATen has its own ``*_kernel`` symbols (``sinc_vectorized4_kernel``
# and friends), so a name-shape heuristic alone over-reports Triton.
_TRITON_MODULES = (
    "sezm.flash_atten",
    "sezm.force_assembly",
    "sezm.gated_activation",
    "sezm.grid_pair",
    "sezm.radial_mix",
    "sezm.segment_softmax",
    "sezm.so2_block_gemm",
    "sezm.so2_rotation",
    "sezm.so2_stack_fp16x3",
    "sezm.so2_value_path",
    "sezm.wigner_monomials",
    "env_mat",
    "dpa1.activation",
    "dpa1.edge_conv",
    "dpa1.gemm_fp16x3",
    "dpa1.se_conv",
)


def _import_optional(module_name: str) -> object | None:
    """Import a DeepMD Triton module, or return ``None`` when unavailable.

    A profiling host may have no DeepMD, no Triton, or a DeepMD built without
    these kernels; all three are valid states for a trace analyzer.
    """
    try:
        return importlib.import_module(f"deepmd.pt_expt.kernels.triton.{module_name}")
    except Exception:  # noqa: BLE001 - availability is exactly what is uncertain
        return None


def _known_triton_kernels() -> tuple[set[str], str]:
    """Return the Triton kernel names the installed DeepMD can launch."""
    names: set[str] = set()
    imported = 0
    for module_name in _TRITON_MODULES:
        module = _import_optional(module_name)
        if module is None:
            continue
        imported += 1
        for value in vars(module).values():
            function = getattr(value, "fn", None)
            kernel_name = getattr(function, "__name__", None)
            if kernel_name:
                names.add(kernel_name)
    if not names:
        return set(), f"none (imported {imported} modules, no JIT function found)"
    return names, f"deepmd ({len(names)} kernels from {imported} modules)"


def _classify(name: str, known: set[str]) -> str:
    """Bucket a kernel name into the family a tuning decision is made for."""
    if name.startswith("Cijk_") or "Tensile" in name or "rocblas" in name.lower():
        return "gemm"
    if name in known:
        return "triton"
    if "at::" in name or "rocprim" in name or "::" in name or name.startswith("void "):
        return "aten"
    # Fallback for a Triton build the registry could not import: SeZM kernels are
    # module-private ``_*_kernel`` functions, which excludes ATen's globals.
    if name.endswith("_kernel") and name.startswith("_"):
        return "triton"
    return "other"


def _load_events(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise TypeError(f"{path}: traceEvents is not a list")
    return [event for event in events if isinstance(event, dict)]


def _shorten(value: object, limit: int = 110) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _cast_signature(event: dict) -> str:
    """Return an editable description of one cast/copy call.

    Torch's Kineto emits Python frames as separate ``python_function`` events
    on another thread id, so a ``cpu_op`` rarely carries an inline stack.  The
    recorded input dtypes and dimensions are what actually localises the cast:
    an fp64 ``[1, 1]`` conversion repeated once per layer is an accumulation
    cast, an fp64 ``[N, C]`` one is a tensor that should have stayed fp32.
    """
    args = event.get("args") or {}
    stack = args.get("Call stack") or args.get("call_stack") or ""
    if stack:
        lines = [line.strip() for line in str(stack).splitlines() if line.strip()]
        frames = [line for line in lines if ".py" in line] or lines
        return " <- ".join(frames[:3])
    return (
        f"input_dtype={_shorten(args.get('Input type', '?'), 40)} "
        f"input_dims={_shorten(args.get('Input Dims', '?'), 70)}"
    )


def analyze(
    trace_path: Path,
    *,
    top: int = 25,
    bandwidth_gbps: float = 1000.0,
    peak_tflops: float = 13.7,
    use_registry: bool = True,
) -> dict:
    known, detection = _known_triton_kernels() if use_registry else (set(), "disabled")
    events = _load_events(trace_path)
    device: dict[str, dict] = {}
    cpu: dict[str, dict] = {}
    casts: Counter = Counter()
    cast_calls: Counter = Counter()
    cast_samples: dict[str, list[float]] = {}
    cast_op: dict[str, str] = {}

    for event in events:
        if event.get("ph") != "X":
            continue
        duration = float(event.get("dur", 0) or 0)
        if duration <= 0:
            continue
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", ""))
        if category in _DEVICE_CATEGORIES:
            row = device.setdefault(name, {"name": name, "calls": 0, "total_us": 0.0})
            row["calls"] += 1
            row["total_us"] += duration
        elif "cpu_op" in category:
            row = cpu.setdefault(name, {"name": name, "calls": 0, "total_us": 0.0})
            row["calls"] += 1
            row["total_us"] += duration
            if name in _CAST_OPS:
                signature = _cast_signature(event)
                casts[signature] += duration
                cast_calls[signature] += 1
                cast_op.setdefault(signature, name)
                cast_samples.setdefault(signature, []).append(duration)

    if not device:
        raise RuntimeError(f"{trace_path}: no device kernel events recorded")

    rows = []
    for row in device.values():
        family = _classify(row["name"], known)
        mean_us = row["total_us"] / row["calls"] if row["calls"] else 0.0
        enriched = dict(row, family=family, mean_us=mean_us)
        if family == "gemm":
            # Work a compute-bound kernel of this duration would complete.
            enriched["equiv_gflop"] = mean_us / 1e3 * peak_tflops
        else:
            # Bytes a memory-bound kernel of this duration would move.
            enriched["equiv_mb"] = mean_us / 1e3 * bandwidth_gbps
        rows.append(enriched)
    rows.sort(key=lambda row: row["total_us"], reverse=True)

    total_device_us = sum(row["total_us"] for row in rows)
    families: dict[str, dict] = {}
    for row in rows:
        bucket = families.setdefault(
            row["family"], {"calls": 0, "total_us": 0.0, "kernels": 0}
        )
        bucket["calls"] += row["calls"]
        bucket["total_us"] += row["total_us"]
        bucket["kernels"] += 1
    for bucket in families.values():
        bucket["share"] = (
            bucket["total_us"] / total_device_us if total_device_us else 0.0
        )

    triton_rows = [row for row in rows if row["family"] == "triton"]
    cpu_rows = sorted(cpu.values(), key=lambda row: row["total_us"], reverse=True)

    return {
        "trace_file": str(trace_path),
        "device_kernel_calls": sum(row["calls"] for row in rows),
        "device_kernel_count": len(rows),
        "total_device_us": total_device_us,
        "families": families,
        "top_kernels": rows[:top],
        "triton_kernels": triton_rows,
        "triton_kernel_calls": sum(row["calls"] for row in triton_rows),
        "top_cpu_ops": cpu_rows[:top],
        "cast_report": [
            {
                "op": cast_op[signature],
                "calls": cast_calls[signature],
                "duration_us": duration,
                "max_us": max(cast_samples[signature]),
                "signature": signature,
            }
            for signature, duration in casts.most_common(20)
        ],
        "triton_detection": detection,
        "assumptions": {
            "bandwidth_gbps": bandwidth_gbps,
            "peak_tflops": peak_tflops,
            "note": "Summed durations overlap; they are not the inference latency.",
        },
    }


def _print_report(report: dict, top: int) -> None:
    print(f"trace: {report['trace_file']}")
    print(
        f"device kernels: {report['device_kernel_calls']} calls in "
        f"{report['device_kernel_count']} distinct kernels, "
        f"{report['total_device_us'] / 1e3:.1f} ms summed"
    )
    print("\n-- family share --")
    print(
        f"{'family':8s} {'kernels':>8s} {'calls':>8s} {'total ms':>10s} {'share':>7s}"
    )
    for family, bucket in sorted(
        report["families"].items(), key=lambda item: -item[1]["total_us"]
    ):
        print(
            f"{family:8s} {bucket['kernels']:8d} {bucket['calls']:8d} "
            f"{bucket['total_us'] / 1e3:10.1f} {bucket['share'] * 100:6.1f}%"
        )

    print(f"\n-- top {top} kernels --")
    header = (
        f"{'ms':>9s} {'share':>6s} {'calls':>7s} {'mean us':>9s} {'bound':>10s}  name"
    )
    print(header)
    for row in report["top_kernels"]:
        if row["family"] == "gemm":
            bound = f"{row.get('equiv_gflop', 0):.1f} GF"
        else:
            bound = f"{row.get('equiv_mb', 0):.0f} MB"
        share = row["total_us"] / report["total_device_us"] * 100
        print(
            f"{row['total_us'] / 1e3:9.1f} {share:5.1f}% {row['calls']:7d} "
            f"{row['mean_us']:9.1f} {bound:>10s}  {row['name'][:96]}"
        )

    print(f"\n-- triton inventory (source: {report['triton_detection']}) --")
    if report["triton_kernels"]:
        print(
            f"{report['triton_kernel_calls']} calls across "
            f"{len(report['triton_kernels'])} Triton kernels"
        )
        for row in report["triton_kernels"]:
            print(
                f"  {row['total_us'] / 1e3:8.2f} ms  {row['calls']:6d}x  "
                f"{row['mean_us']:8.1f} us  {row['name'][:80]}"
            )
    else:
        print(
            "NO Triton kernel in this trace: the run measured the dense reference\n"
            "path. Check DP_TRITON_INFER / TRITON_AVAILABLE before drawing any\n"
            "tuning conclusion from these numbers."
        )

    if report["top_cpu_ops"]:
        print(f"\n-- top {top} CPU ops (overlapping) --")
        for row in report["top_cpu_ops"][:top]:
            print(
                f"  {row['total_us'] / 1e3:9.1f} ms  {row['calls']:6d}x  {row['name'][:80]}"
            )

    if report["cast_report"]:
        print("\n-- cast / copy attribution (dtype + dims of the converted input) --")
        for row in report["cast_report"]:
            print(
                f"  {row['duration_us'] / 1e3:9.1f} ms  {row['calls']:6d}x  "
                f"max {row['max_us'] / 1e3:8.1f} ms  {row['op']}"
            )
            print(f"      {row['signature'][:200]}")
        print(
            "  caution: a CPU op's duration includes waiting for the device, so a\n"
            "  single call far longer than the others is a synchronisation point\n"
            "  (the pipeline draining), not conversion work."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace", type=Path, help="Chrome trace written by the profile CLI"
    )
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=1000.0,
        help="effective HBM bandwidth for the equivalent-MB estimate",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=13.7,
        help="measured peak for the equivalent-GFLOP estimate",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="classify Triton kernels by name shape only, without importing DeepMD",
    )
    args = parser.parse_args(argv)

    if not args.trace.is_file():
        print(f"trace does not exist: {args.trace}", file=sys.stderr)
        return 2
    report = analyze(
        args.trace,
        top=args.top,
        bandwidth_gbps=args.bandwidth_gbps,
        peak_tflops=args.peak_tflops,
        use_registry=not args.no_registry,
    )
    _print_report(report, args.top)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
