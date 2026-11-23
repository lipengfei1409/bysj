"""Report the device-time share of every operator issued inside one module scope.

`analyze_trace.py` answers "which kernel family dominates"; this script answers
the narrower question a fusion plan needs: **of the time the model itself
spends on the device, which operators own how much**, split into forward and
backward, and at which exact shape.

Scope handling matters because a profiled DCU trace contains work that is not
the model: neighbour-list builds, host-side casts, the DeepPot wrapper.  The
selector is a Kineto module event (`nn.Module: ModelWrapper_0` by default);
a kernel counts only when its *launch* timestamp falls inside that interval.
Launch time, not device time, is the correct key: the whole graph is enqueued
within the module call and the device drains it afterwards, so filtering on
device timestamps would drop most of the backward.

Usage::

    python tests/deepmd/profile/op_time_share.py dpa4_inference_trace.json \
        --module 'nn.Module: ModelWrapper_0' --top 25 --json /tmp/share.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCAN_PATH = Path(__file__).with_name("scan_fusion_patterns.py")
DEFAULT_MODULE = "nn.Module: ModelWrapper_0"


def _load_scan_module():
    """Import ``scan_fusion_patterns`` by path.

    The profile directory is not a package (the CLI is run as a file), so the
    join helpers are loaded by path instead of a normal import.
    """
    spec = importlib.util.spec_from_file_location("_dpa4_scan", _SCAN_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError(f"cannot load {_SCAN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_interval(events: Sequence[Dict[str, Any]], module: str) -> Tuple[float, float]:
    """Return the ``(start, end)`` of the named module scope, or raise."""
    for event in events:
        if event.get("cat") == "python_function" and event.get("name") == module:
            start = float(event.get("ts") or 0.0)
            return start, start + float(event.get("dur") or 0.0)
    available = sorted(
        {
            str(event.get("name"))
            for event in events
            if event.get("cat") == "python_function"
            and str(event.get("name") or "").startswith("nn.Module:")
        }
    )
    raise KeyError(f"{module!r} not in trace; module scopes present: {available[:12]}")


def _short_op(op: str) -> str:
    return op.split("::")[-1]


def analyze(
    trace_path: Path,
    *,
    module: str = DEFAULT_MODULE,
    top: int = 25,
) -> Dict[str, Any]:
    scan = _load_scan_module()
    events = scan._load_events(trace_path)
    launches, join = scan._build_launches(events)
    start, end = _module_interval(events, module)

    inside = [row for row in launches if start <= row.cpu_ts <= end]
    forward = scan._region_intervals(events, scan.DEFAULT_FORWARD_REGION_SUFFIX)
    backward = scan._region_intervals(events, scan.DEFAULT_BACKWARD_REGION_SUFFIX)

    def region(row) -> str:
        if scan._in_intervals(row.cpu_ts, backward):
            return "backward"
        if scan._in_intervals(row.cpu_ts, forward):
            return "forward"
        return "other"

    total_us = sum(row.gpu_us for row in launches)

    ops: Dict[str, Dict[str, Any]] = {}
    shapes: Dict[str, Dict[str, Any]] = {}
    families: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "us": 0.0, "ops": set()}
    )
    for row in inside:
        where = region(row)
        op = _short_op(row.op)
        bucket = ops.setdefault(
            op,
            {"op": op, "family": row.family, "calls": 0, "us": 0.0,
             "forward_us": 0.0, "backward_us": 0.0, "other_us": 0.0,
             "shapes": defaultdict(float)},
        )
        bucket["calls"] += 1
        bucket["us"] += row.gpu_us
        bucket[f"{where}_us"] += row.gpu_us
        bucket["shapes"][row.signature] += row.gpu_us
        sig = shapes.setdefault(
            row.signature,
            {"signature": row.signature, "op": op, "family": row.family,
             "calls": 0, "us": 0.0, "forward_us": 0.0, "backward_us": 0.0,
             "shape_class": row.shape_class, "numel": row.numel},
        )
        sig["calls"] += 1
        sig["us"] += row.gpu_us
        if where != "other":
            sig[f"{where}_us"] += row.gpu_us
        fam = families[row.family]
        fam["calls"] += 1
        fam["us"] += row.gpu_us
        fam["ops"].add(op)

    scoped_us = sum(bucket["us"] for bucket in ops.values())

    def finish(bucket: Dict[str, Any], key: str) -> Dict[str, Any]:
        for name in ("us", "forward_us", "backward_us", "other_us"):
            if name in bucket:
                bucket[f"{name.replace('_us', '')}_ms" if name != "us" else "ms"] = (
                    bucket.pop(name) / 1e3
                )
        bucket["share"] = bucket["ms"] / (scoped_us / 1e3) if scoped_us else 0.0
        bucket["mean_us"] = bucket["ms"] * 1e3 / bucket["calls"] if bucket["calls"] else 0.0
        if key == "ops":
            bucket["top_shape"] = max(
                bucket.pop("shapes").items(), key=lambda item: item[1]
            )[0]
        return bucket

    op_rows = sorted(
        (finish(dict(bucket, shapes=dict(bucket["shapes"])), "ops") for bucket in ops.values()),
        key=lambda row: row["ms"],
        reverse=True,
    )
    shape_rows = sorted(
        (finish(dict(bucket), "shapes") for bucket in shapes.values()),
        key=lambda row: row["ms"],
        reverse=True,
    )
    family_rows = []
    for name, bucket in sorted(families.items(), key=lambda item: -item[1]["us"]):
        family_rows.append(
            {
                "family": name,
                "calls": bucket["calls"],
                "ms": bucket["us"] / 1e3,
                "share": bucket["us"] / scoped_us if scoped_us else 0.0,
                "distinct_ops": len(bucket["ops"]),
            }
        )

    def region_ms(rows) -> Tuple[float, float]:
        return (
            sum(row.gpu_us for row in rows if region(row) == "forward") / 1e3,
            sum(row.gpu_us for row in rows if region(row) == "backward") / 1e3,
        )

    fwd_ms, bwd_ms = region_ms(inside)
    return {
        "trace_file": str(trace_path),
        "module": module,
        "module_window_ms": {"start": 0.0, "duration": (end - start) / 1e3},
        "coverage": {
            "kernels_in_trace": len(launches),
            "kernels_inside_module": len(inside),
            "trace_total_ms": total_us / 1e3,
            "module_total_ms": scoped_us / 1e3,
            "share_of_trace": scoped_us / total_us if total_us else 0.0,
            "join": join,
        },
        "regions": {
            "forward_ms": fwd_ms,
            "backward_ms": bwd_ms,
            "other_ms": scoped_us / 1e3 - fwd_ms - bwd_ms,
            "forward_calls": sum(1 for row in inside if region(row) == "forward"),
            "backward_calls": sum(1 for row in inside if region(row) == "backward"),
        },
        "families": family_rows,
        # The full lists are kept: a caller asking "where did operator X go"
        # must not be defeated by the print-time truncation below.
        "ops": op_rows,
        "op_ms_total": sum(row["ms"] for row in op_rows),
        "shapes": shape_rows,
    }


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _print(report: Dict[str, Any], top: int) -> None:
    cov = report["coverage"]
    print(f"trace: {report['trace_file']}")
    print(
        f"scope: {report['module']}  "
        f"(window {report['module_window_ms']['duration']:.1f} ms)\n"
    )
    print(
        f"kernels inside scope: {cov['kernels_inside_module']} / "
        f"{cov['kernels_in_trace']}  ->  {cov['module_total_ms']:.1f} ms "
        f"of {cov['trace_total_ms']:.1f} ms ({_pct(cov['share_of_trace'])})"
    )
    join = cov["join"]
    if join["missing_cpu_op"] or join["missing_kernel"] or join["missing_ids"]:
        print(f"  WARNING unjoined launches: {join}")

    regions = report["regions"]
    print(
        f"forward:  {regions['forward_calls']:5d} calls  {regions['forward_ms']:8.1f} ms  "
        f"{_pct(regions['forward_ms'] / cov['module_total_ms'])}"
    )
    print(
        f"backward: {regions['backward_calls']:5d} calls  {regions['backward_ms']:8.1f} ms  "
        f"{_pct(regions['backward_ms'] / cov['module_total_ms'])}"
    )
    if regions["other_ms"] > 0.05:
        print(f"outside both regions: {regions['other_ms']:8.1f} ms")

    print("\n-- family roll-up --")
    print(f"{'family':12s} {'ops':>4s} {'calls':>7s} {'ms':>9s} {'share':>7s}")
    for row in report["families"]:
        print(
            f"{row['family']:12s} {row['distinct_ops']:4d} {row['calls']:7d} "
            f"{row['ms']:9.1f} {_pct(row['share'])}"
        )

    print(f"\n-- top {top} operators (of {len(report['ops'])}) --")
    print(
        f"{'ms':>8s} {'share':>6s} {'fwd ms':>8s} {'bwd ms':>8s} {'calls':>6s} "
        f"{'mean us':>8s}  {'family':10s} op  [top shape]"
    )
    for row in report["ops"][:top]:
        print(
            f"{row['ms']:8.1f} {_pct(row['share'])} {row['forward_ms']:8.1f} "
            f"{row['backward_ms']:8.1f} {row['calls']:6d} {row['mean_us']:8.1f}  "
            f"{row['family']:10s} {row['op']}  [{row['top_shape'][:90]}]"
        )

    print(f"\n-- top {top} operator+shape sites (of {len(report['shapes'])}) --")
    for row in report["shapes"][:top]:
        print(
            f"{row['ms']:8.1f} {_pct(row['share'])} {row['forward_ms']:8.1f} "
            f"{row['backward_ms']:8.1f} {row['calls']:6d}  {row['shape_class']:6s} "
            f"{row['signature'][:120]}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.trace.is_file():
        print(f"trace does not exist: {args.trace}", file=sys.stderr)
        return 2
    report = analyze(args.trace, module=args.module, top=args.top)
    _print(report, args.top)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
