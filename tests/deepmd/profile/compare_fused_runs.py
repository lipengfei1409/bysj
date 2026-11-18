"""Compare two 3000-atom DPA4 profile summaries: baseline vs fused dispatch.

``test_dpa4_inference_profile.py dpa4 --fused <seams>`` writes a summary with a
``fused`` block; this tool puts two of them side by side and reports where the
device time moved. It only reads summaries -- it never runs inference -- so it
works anywhere Python runs.

Usage::

    python tests/deepmd/profile/compare_fused_runs.py \\
        --baseline /tmp/a/dpa4_*/dpa4_inference_profile_summary.json \\
        --patched  /tmp/b/dpa4_*/dpa4_inference_profile_summary.json \\
        --json /tmp/fused_compare.json

The device total is the ``dpa4/inference.single_energy_force`` annotation from
``top_profiler_events``; per-kernel deltas come from ``trace_analysis`` when the
run was profiled. Numeric parity (energy/force) needs the DCU-side parity test
``tests/csrc/accuracy/test_fused_deepmd_parity.py``; summaries only record
shapes and dtypes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _device_ms(summary: dict) -> float | None:
    events = summary.get("top_profiler_events") or []
    for event in events:
        if str(event.get("name", "")).startswith("dpa4/"):
            return float(event.get("self_device_us", 0.0)) / 1000.0
    analysis = summary.get("trace_analysis") or {}
    kernels = analysis.get("top_device_kernels") or []
    if kernels:
        return sum(float(kernel.get("duration_us", 0.0)) for kernel in kernels) / 1000.0
    return None


def _kernels(summary: dict) -> dict[str, float]:
    analysis = summary.get("trace_analysis") or {}
    table = {}
    for kernel in analysis.get("top_device_kernels") or []:
        table[str(kernel.get("name", ""))] = float(kernel.get("duration_us", 0.0)) / 1000.0
    return table


def _short(name: str, width: int = 72) -> str:
    text = name.replace("void ", "").replace("at::native::", "")
    return text if len(text) <= width else text[: width - 1] + "…"


def compare(baseline: dict, patched: dict, *, top: int = 8) -> dict:
    """Return the comparison report; every number comes from the two summaries."""
    base_ms, patched_ms = _device_ms(baseline), _device_ms(patched)
    base_kernels, patched_kernels = _kernels(baseline), _kernels(patched)
    shared = sorted(
        set(base_kernels) | set(patched_kernels),
        key=lambda name: -(abs(patched_kernels.get(name, 0.0) - base_kernels.get(name, 0.0))),
    )
    deltas = [
        {
            "kernel": name,
            "baseline_ms": base_kernels.get(name, 0.0),
            "patched_ms": patched_kernels.get(name, 0.0),
            "delta_ms": patched_kernels.get(name, 0.0) - base_kernels.get(name, 0.0),
        }
        for name in shared[:top]
    ]
    speedup = None
    if base_ms and patched_ms:
        speedup = base_ms / patched_ms
    fused = patched.get("fused") or {}
    dispatched = (fused.get("softmax_hits") or 0) + (fused.get("envelope_hits") or 0)
    bound = bool(fused.get("bound_softmax") or fused.get("bound_envelope"))
    config_fields = (
        "model_file",
        "backend",
        "device_name",
        "atom_count",
        "input_source",
        "triton",
        "warmup_calls",
        "profiled",
    )
    return {
        "baseline_ms": base_ms,
        "patched_ms": patched_ms,
        "speedup": speedup,
        "baseline_launches": (baseline.get("trace_analysis") or {}).get("device_kernel_calls"),
        "patched_launches": (patched.get("trace_analysis") or {}).get("device_kernel_calls"),
        "fused": fused,
        "fused_bound": bound,
        "fused_dispatched": dispatched,
        "command_differs": baseline.get("command") != patched.get("command"),
        "baseline_command": baseline.get("command"),
        "patched_command": patched.get("command"),
        "kernel_deltas": deltas,
        "config_matches": {
            field: baseline.get(field) == patched.get(field) for field in config_fields
        },
    }


def render(report: dict) -> str:
    lines = ["# fused dispatch comparison", ""]
    base_ms, patched_ms = report["baseline_ms"], report["patched_ms"]
    if base_ms is None or patched_ms is None:
        lines.append("device time unavailable in one of the summaries (was --no-profile used?)")
    else:
        lines.append(f"device total : {base_ms:9.1f} ms -> {patched_ms:9.1f} ms"
                     f"   ({report['speedup']:.3f}x)")
    base_launches, patched_launches = report["baseline_launches"], report["patched_launches"]
    if base_launches is not None and patched_launches is not None:
        lines.append(f"launches     : {base_launches:9d} -> {patched_launches:9d}"
                     f"   ({patched_launches - base_launches:+d})")
    fused = report["fused"]
    if fused:
        lines.append(
            "fused seams  : "
            f"softmax={fused.get('softmax_hits')} (fallback {fused.get('softmax_fallbacks')}), "
            f"envelope={fused.get('envelope_hits')} (fallback {fused.get('envelope_fallbacks')})"
        )
        if report.get("fused_bound") and not report.get("fused_dispatched"):
            lines.append(
                "fused seams  : bound but never dispatched (hits 0) -- the counters are "
                "read after inference, so the seam really was not called"
            )
        if fused.get("last_error"):
            lines.append(f"last error   : {fused['last_error']}")
    else:
        lines.append("fused seams  : none recorded (run the patched pass with --fused on)")
    mismatched = [name for name, same in report["config_matches"].items() if not same]
    if mismatched:
        lines.append(f"config       : NOT comparable, differs in {', '.join(mismatched)}")
    if report.get("command_differs"):
        lines.append(
            "command      : the two runs used different invocations; compare the "
            "--json output before trusting the delta"
        )
    if report["kernel_deltas"]:
        lines += ["", "per-kernel delta (top by magnitude)", ""]
        lines.append(f"{'baseline':>10} {'patched':>10} {'delta':>10}  kernel")
        for row in report["kernel_deltas"]:
            lines.append(
                f"{row['baseline_ms']:10.2f} {row['patched_ms']:10.2f} "
                f"{row['delta_ms']:+10.2f}  {_short(row['kernel'])}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--patched", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args(argv)

    baseline = json.loads(args.baseline.read_text())
    patched = json.loads(args.patched.read_text())
    report = compare(baseline, patched, top=args.top)
    print(render(report), flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\ncomparison JSON: {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
