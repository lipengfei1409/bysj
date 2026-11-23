"""Audit whether a trace has *chain-fusable* elementwise runs to fuse.

A "fuse the elementwise chain" plan assumes the graph contains runs of
back-to-back elementwise ops on the same tensor, so one kernel can read once and
write once instead of k times.  That assumption is measurable, and it is easy to
get wrong from a family-level total: the elementwise family can be a large share
of device time and still consist of isolated single ops separated by GEMMs,
gathers and scatters, which no elementwise-chain kernel can merge.

This script reports, for one trace:

* every maximal run of consecutive launches that are elementwise and share the
  same ``Input Dims`` signature (same shape is the cheap proxy for "same
  tensor"), with the run length distribution;
* the same-shape total split by length, so the reader can see how much lives in
  runs long enough to be worth a kernel;
* whether any ``fill_`` is followed by an accumulate on the same tensor, which
  is the other pattern an elementwise pass can remove (write on first touch
  instead of zero-then-accumulate).

Usage::

    python tests/deepmd/profile/chain_runs.py dpa4_inference_trace.json --top 12
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: Ops that a single elementwise kernel could evaluate in one pass.
ELEMENTWISE = frozenset(
    {
        "mul", "add", "add_", "sub", "mul_", "div", "neg", "exp", "sqrt",
        "rsqrt", "pow", "clamp", "where", "reciprocal", "tanh", "log", "square",
        "sigmoid", "sigmoid_backward", "silu", "silu_backward", "leaky_relu",
        "gelu", "copy_", "fill_", "cat", "index_select", "gather",
    }
)
ACCUMULATE = frozenset(
    {"add_", "index_add_", "scatter_add_", "_index_put_impl_", "copy_", "mul_"}
)


def _scan_module():
    spec = importlib.util.spec_from_file_location(
        "_dpa4_scan_chain", _HERE / "scan_fusion_patterns.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError("cannot load scan_fusion_patterns.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _short(op: str) -> str:
    return op.split("::")[-1]


def analyse(trace: Path, *, top: int = 12) -> dict:
    scan = _scan_module()
    events = scan._load_events(trace)
    launches, join = scan._build_launches(events)
    forward = scan._region_intervals(events, scan.DEFAULT_FORWARD_REGION_SUFFIX)
    backward = scan._region_intervals(events, scan.DEFAULT_BACKWARD_REGION_SUFFIX)

    def region(row) -> str:
        if scan._in_intervals(row.cpu_ts, backward):
            return "backward"
        if scan._in_intervals(row.cpu_ts, forward):
            return "forward"
        return "other"

    runs: list[list] = []
    current: list = []
    for row in launches:
        op = _short(row.op)
        fusable = op in ELEMENTWISE
        if fusable and (not current or str(row.dims) == str(current[-1].dims)):
            current.append(row)
        else:
            if len(current) > 1:
                runs.append(current)
            current = [row] if fusable else []
    if len(current) > 1:
        runs.append(current)

    def ms(run: list) -> float:
        return sum(row.gpu_us for row in run) / 1e3

    by_length: Counter = Counter()
    count_by_length: Counter = Counter()
    for run in runs:
        by_length[len(run)] += ms(run)
        count_by_length[len(run)] += 1

    # fill_ followed within two launches by an accumulate on the same signature
    fill_pairs = []
    for index, row in enumerate(launches):
        if _short(row.op) != "fill_":
            continue
        for offset in range(index + 1, min(index + 3, len(launches))):
            other = launches[offset]
            if str(other.dims) == str(row.dims) and _short(other.op) in ACCUMULATE:
                fill_pairs.append((row, other))
                break

    total_fill = sum(row.gpu_us for row in launches if _short(row.op) == "fill_") / 1e3
    paired_fill = sum(fill.gpu_us + acc.gpu_us for fill, acc in fill_pairs) / 1e3

    ordered = sorted(runs, key=ms, reverse=True)[:top]
    return {
        "trace": str(trace),
        "launches": len(launches),
        "device_ms": sum(row.gpu_us for row in launches) / 1e3,
        "join": join,
        "runs": {
            "count": len(runs),
            "total_ms": sum(ms(run) for run in runs),
            "by_length_ms": {length: by_length[length] for length in sorted(by_length)},
            "count_by_length": {
                length: count_by_length[length] for length in sorted(count_by_length)
            },
            "at_least_3_ms": sum(ms(run) for run in runs if len(run) >= 3),
            "at_least_4_ms": sum(ms(run) for run in runs if len(run) >= 4),
        },
        "top_runs": [
            {
                "ms": ms(run),
                "length": len(run),
                "region": region(run[0]),
                "ops": [_short(row.op) for row in run],
                "dims": str(run[0].dims),
            }
            for run in ordered
        ],
        "fill": {
            "total_ms": total_fill,
            "paired_ms": paired_fill,
            "pairs": len(fill_pairs),
        },
    }


def _print(report: dict) -> None:
    runs = report["runs"]
    print(f"trace: {report['trace']}")
    print(
        f"{report['launches']} launches, {report['device_ms']:.1f} ms device\n"
        f"same-shape elementwise runs: {runs['count']}, {runs['total_ms']:.1f} ms"
    )
    for length, value in runs["by_length_ms"].items():
        print(
            f"  len {length:2d}: {value:8.1f} ms  ({runs['count_by_length'][length]} runs)"
        )
    print(
        f"\nfusable material: len>=3 {runs['at_least_3_ms']:.1f} ms, "
        f"len>=4 {runs['at_least_4_ms']:.1f} ms"
    )
    print(
        f"fill_ total {report['fill']['total_ms']:.1f} ms; "
        f"fill->accumulate pairs {report['fill']['pairs']} "
        f"({report['fill']['paired_ms']:.1f} ms)"
    )
    print("\n-- top runs --")
    for row in report["top_runs"]:
        print(
            f"{row['ms']:8.1f} ms  len {row['length']:2d}  {row['region']:8s}  "
            f"{' -> '.join(row['ops'])[:60]}  [{row['dims'][:50]}]"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.trace.is_file():
        print(f"trace does not exist: {args.trace}", file=sys.stderr)
        return 2
    report = analyse(args.trace, top=args.top)
    _print(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
