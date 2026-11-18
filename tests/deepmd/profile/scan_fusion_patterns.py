"""Scan a Kineto/ROCm trace for adjacent ATen launch patterns (fusion candidates).

The trace records device kernels (``cat == "kernel"``), the HIP launch API calls
that produced them (``cat == "cuda_runtime"``) and the CPU ATen operators those
launches belong to (``cat == "cpu_op"``).  Kineto links them with two ids:

* ``correlation``: ``cuda_runtime`` launch -> ``kernel`` duration,
* ``External id``: ``cuda_runtime`` launch -> the ``cpu_op`` that enqueued it.

Sorting the joined launches by their CPU timestamp reconstructs the exact order
in which the model enqueued GPU work.  Consecutive entries in that order are
what this tool calls *adjacent*, and an n-gram of consecutive entries is a
candidate fusion chain.

Signature
---------
An entry's signature is ``op_name + "|" + str(Input Dims)``; the full signature
adds the recorded input dtypes (``"|" + str(Input type)``).  Patterns are counted
on the collapsed signature, while ``stability`` (the share of occurrences whose
full signature matches the most common one) guards against merging chains that
only look identical because the dtype was dropped.

Traffic estimate
----------------
For every pattern entry the tool estimates addressable traffic as
``3 * numel * 4`` bytes.  ``numel`` is the element count of the *largest* tensor
shape found in ``Input Dims`` (the dominant read/write allocation); a launch
with no shape information is counted as unknown and excluded from the estimate.
The roofline uses the measured 630 GB/s effective streaming bandwidth of these
ATen kernels on the profiled DCU::

    roofline_ms = traffic_bytes / 630e9 * 1000
    saving_ms   = total_ms - roofline_ms

``total_ms`` sums device kernel durations.  Kineto sums overlap across launches
and the n-gram windows themselves overlap, so these totals attribute device time
and are *not* the inference latency.  bmm/GEMM chains are compute bound, so their
bandwidth roofline is not a meaningful target; their saving is reported only
because the formula above is applied uniformly.

Usage::

    python3 myproject/tests/deepmd/profile/scan_fusion_patterns.py \
        --trace dpa4_inference_trace.json \
        --top 25 --max-len 5 \
        --json myproject/docs/fusion-patterns.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

LAUNCH_API_NAMES: Tuple[str, ...] = (
    "hipLaunchKernel",
    "hipExtModuleLaunchKernel",
    "hipModuleLaunchKernel",
)

DEFAULT_FORWARD_REGION_SUFFIX = "sezm.py(1470): forward_with_edges"
DEFAULT_BACKWARD_REGION_SUFFIX = "transform_output.py(214): edge_energy_deriv"

# Shape sentinels documented for dpa4_inference_trace.json: 3000 atoms and
# 181838 edges (a handful of gathers see a 181836-edge index tensor).
NODE_MAJOR_SIZE = 3000
EDGE_MAJOR_SIZES: Tuple[int, ...] = (181838, 181836)

EFFECTIVE_BANDWIDTH_BYTES_PER_S = 630e9
BYTES_PER_ELEMENT = 4
STREAMS_PER_ENTRY = 3

REFERENCE_TRACE_NAME = "dpa4_inference_trace.json"
REFERENCE_SANITY: Dict[str, float] = {
    "kernel_events": 2614,
    "kernel_sum_ms": 1218.58,
    "forward_launches": 943,
    "forward_ms": 436.2,
    "backward_launches": 1573,
    "backward_ms": 780.4,
}
MS_TOLERANCE = 0.05

_FAMILY_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "memcpy",
        (
            "copy_",
            "_copy_",
            "to",
            "_to_copy",
            "clone",
            "contiguous",
            "type_as",
            "memcpy",
        ),
    ),
    (
        "gemm",
        ("bmm", "mm", "addmm", "baddbmm", "matmul", "linear", "einsum"),
    ),
    (
        "index",
        (
            "gather",
            "index",
            "index_select",
            "index_add",
            "index_add_",
            "index_copy",
            "index_copy_",
            "index_put_",
            "_index_put_impl_",
            "scatter",
            "scatter_add",
            "scatter_add_",
            "scatter_reduce",
            "nonzero",
            "masked_select",
            "take",
            "embedding",
            "unique",
        ),
    ),
    (
        "reduce",
        (
            "sum",
            "mean",
            "max",
            "min",
            "amax",
            "amin",
            "any",
            "all",
            "prod",
            "norm",
            "argmax",
            "argmin",
            "logsumexp",
            "std",
            "var",
            "_softmax",
            "_softmax_backward_data",
            "_log_softmax",
        ),
    ),
    (
        "cat",
        (
            "cat",
            "concat",
            "stack",
            "hstack",
            "vstack",
            "split",
            "split_with_sizes",
            "chunk",
            "narrow",
        ),
    ),
    (
        "init",
        (
            "fill_",
            "zero_",
            "arange",
            "empty",
            "ones",
            "zeros",
            "rand",
            "randn",
            "uniform_",
            "normal_",
            "full",
        ),
    ),
    (
        "elementwise",
        (
            "add",
            "add_",
            "sub",
            "sub_",
            "mul",
            "mul_",
            "div",
            "div_",
            "true_divide",
            "neg",
            "neg_",
            "pow",
            "pow_",
            "exp",
            "log",
            "sqrt",
            "rsqrt",
            "reciprocal",
            "sigmoid",
            "sigmoid_backward",
            "silu",
            "silu_backward",
            "relu",
            "relu_",
            "tanh",
            "tanh_backward",
            "softplus",
            "abs",
            "clamp",
            "clamp_min",
            "clamp_max",
            "remainder",
            "fmod",
            "floor",
            "ceil",
            "round",
            "sin",
            "cos",
            "sinc",
            "erf",
            "eq",
            "ne",
            "gt",
            "ge",
            "le",
            "lt",
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "bitwise_not",
            "logical_and",
            "logical_or",
            "logical_not",
            "where",
            "masked_fill",
            "masked_fill_",
        ),
    ),
)


def _build_family_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for family, names in _FAMILY_RULES:
        for name in names:
            mapping[name] = family
    return mapping


_FAMILY_BY_BASE_OP = _build_family_map()


class Launch(NamedTuple):
    """One GPU kernel launch joined to its ATen op and shape information."""

    cpu_ts: float
    gpu_us: float
    op: str
    dims: Any
    dtypes: Any
    signature: str
    full_signature: str
    numel: Optional[int]
    family: str
    shape_class: str


def _load_events(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise TypeError(f"{path}: traceEvents is not a list")
    return [event for event in events if isinstance(event, dict)]


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tensor_shapes(dims: Any) -> List[List[int]]:
    """Return the statically-known integer shapes inside an ``Input Dims`` value.

    ``Input Dims`` is usually a list of shapes, but ``aten::cat`` / ``stack``
    record a ``TensorList`` as a list of shapes nested one level deeper, so the
    value is walked recursively.  Empty lists are non-tensor arguments (dtype,
    bias, ``None``) and are skipped.
    """
    if isinstance(dims, str):
        try:
            dims = json.loads(dims)
        except json.JSONDecodeError:
            return []
    shapes: List[List[int]] = []

    def visit(value: Any) -> None:
        if not isinstance(value, list) or not value:
            return
        if all(isinstance(dim, int) and not isinstance(dim, bool) for dim in value):
            shapes.append([int(dim) for dim in value])
            return
        for item in value:
            visit(item)

    visit(dims)
    return shapes


def _largest_numel(dims: Any) -> Optional[int]:
    """Element count of the largest tensor shape, or ``None`` if statically unknown."""
    numels: List[int] = []
    for shape in _tensor_shapes(dims):
        numel = 1
        for dim in shape:
            numel *= dim
        numels.append(numel)
    return max(numels) if numels else None


def _shape_class(dims: Any) -> str:
    """Classify a launch as edge-major, node-major, mixed or other."""
    shapes = _tensor_shapes(dims)
    has_edge = False
    has_node = False
    for shape in shapes:
        if shape[0] in EDGE_MAJOR_SIZES:
            has_edge = True
        if len(shape) >= 2 and shape[0] == 1 and shape[1] in EDGE_MAJOR_SIZES:
            has_edge = True
        if shape[0] == NODE_MAJOR_SIZE:
            has_node = True
    if has_edge and has_node:
        return "mixed"
    if has_edge:
        return "edge"
    if has_node:
        return "node"
    return "other"


def _family(op: str) -> str:
    base = op.split("::")[-1]
    return _FAMILY_BY_BASE_OP.get(base, "other")


def _signature(op: str, dims: Any) -> str:
    return f"{op}|{dims}"


def _raw_tensor_bytes(row: Launch) -> Optional[int]:
    if row.numel is None:
        return None
    return STREAMS_PER_ENTRY * row.numel * BYTES_PER_ELEMENT


def _collect_join(
    events: Iterable[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, float], int]:
    cpu_by_external: Dict[int, Dict[str, Any]] = {}
    kernel_us_by_correlation: Dict[int, float] = {}
    kernel_events = 0
    for event in events:
        category = event.get("cat")
        args = event.get("args") or {}
        if category == "cpu_op":
            external = _as_int(args.get("External id"))
            if external is not None:
                cpu_by_external[external] = event
        elif category == "kernel":
            kernel_events += 1
            correlation = _as_int(args.get("correlation"))
            if correlation is not None:
                kernel_us_by_correlation[correlation] = float(event.get("dur") or 0.0)
    return cpu_by_external, kernel_us_by_correlation, kernel_events


def _region_intervals(
    events: Iterable[Dict[str, Any]], suffix: str
) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    for event in events:
        if event.get("cat") != "python_function":
            continue
        name = str(event.get("name") or "")
        if not name.endswith(suffix):
            continue
        start = float(event.get("ts") or 0.0)
        duration = float(event.get("dur") or 0.0)
        intervals.append((start, start + duration))
    return intervals


def _in_intervals(timestamp: float, intervals: Sequence[Tuple[float, float]]) -> bool:
    return any(start <= timestamp <= end for start, end in intervals)


def _build_launches(
    events: Iterable[Dict[str, Any]],
) -> Tuple[List[Launch], Dict[str, Any]]:
    cpu_by_external, kernel_us_by_correlation, kernel_events = _collect_join(events)

    launches: List[Launch] = []
    api_events = 0
    missing_cpu_op = 0
    missing_kernel = 0
    missing_ids = 0
    for event in events:
        if event.get("cat") != "cuda_runtime":
            continue
        if event.get("name") not in LAUNCH_API_NAMES:
            continue
        api_events += 1
        args = event.get("args") or {}
        external = _as_int(args.get("External id"))
        correlation = _as_int(args.get("correlation"))
        if external is None or correlation is None:
            missing_ids += 1
            continue
        cpu_event = cpu_by_external.get(external)
        kernel_us = kernel_us_by_correlation.get(correlation)
        if cpu_event is None:
            missing_cpu_op += 1
            continue
        if kernel_us is None:
            missing_kernel += 1
            continue
        cpu_args = cpu_event.get("args") or {}
        dims = cpu_args.get("Input Dims")
        dtypes = cpu_args.get("Input type")
        op = str(cpu_event.get("name") or "<unnamed>")
        launches.append(
            Launch(
                cpu_ts=float(event.get("ts") or 0.0),
                gpu_us=kernel_us,
                op=op,
                dims=dims,
                dtypes=dtypes,
                signature=_signature(op, dims),
                full_signature=_signature(op, f"{dims}|{dtypes}"),
                numel=_largest_numel(dims),
                family=_family(op),
                shape_class=_shape_class(dims),
            )
        )

    launches.sort(key=lambda row: row.cpu_ts)
    join: Dict[str, Any] = {
        "kernel_events": kernel_events,
        "launch_api_events": api_events,
        "joined_launches": len(launches),
        "missing_cpu_op": missing_cpu_op,
        "missing_kernel": missing_kernel,
        "missing_ids": missing_ids,
    }
    return launches, join


def _split_regions(
    launches: Sequence[Launch],
    events: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    forward = _region_intervals(events, DEFAULT_FORWARD_REGION_SUFFIX)
    backward = _region_intervals(events, DEFAULT_BACKWARD_REGION_SUFFIX)

    def bucket(intervals: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
        selected = [
            row
            for row in launches
            if _in_intervals(row.cpu_ts, intervals)
        ]
        return {
            "regions": len(intervals),
            "launches": len(selected),
            "ms": sum(row.gpu_us for row in selected) / 1000.0,
        }

    forward_bucket = bucket(forward)
    backward_bucket = bucket(backward)
    forward_keys = {
        id(row)
        for row in launches
        if _in_intervals(row.cpu_ts, forward)
    }
    backward_keys = {
        id(row)
        for row in launches
        if _in_intervals(row.cpu_ts, backward)
    }
    outside = [
        row
        for row in launches
        if id(row) not in forward_keys and id(row) not in backward_keys
    ]
    return {
        "forward": {
            **forward_bucket,
            "suffix": DEFAULT_FORWARD_REGION_SUFFIX,
        },
        "backward": {
            **backward_bucket,
            "suffix": DEFAULT_BACKWARD_REGION_SUFFIX,
        },
        "outside": {
            "launches": len(outside),
            "ms": sum(row.gpu_us for row in outside) / 1000.0,
        },
    }


def _sanity(
    launches: Sequence[Launch],
    events: Sequence[Dict[str, Any]],
    join: Dict[str, Any],
) -> Dict[str, Any]:
    split = _split_regions(launches, events)
    sentinels = {"3000": 0, "181838": 0, "181836": 0}
    for row in launches:
        found = {
            str(dim)
            for shape in _tensor_shapes(row.dims)
            for dim in shape
        }
        for sentinel in sentinels:
            if sentinel in found:
                sentinels[sentinel] += 1
    return {
        "kernel_events": join["kernel_events"],
        "kernel_sum_ms": sum(
            float(event.get("dur") or 0.0)
            for event in events
            if event.get("cat") == "kernel"
        )
        / 1000.0,
        "launch_events": len(launches),
        "launch_sum_ms": sum(row.gpu_us for row in launches) / 1000.0,
        "forward": split["forward"],
        "backward": split["backward"],
        "outside": split["outside"],
        "shape_sentinels": sentinels,
    }


def _reference_checks(sanity: Dict[str, Any]) -> List[Dict[str, Any]]:
    computed = {
        "kernel_events": sanity["kernel_events"],
        "kernel_sum_ms": sanity["kernel_sum_ms"],
        "forward_launches": sanity["forward"]["launches"],
        "forward_ms": sanity["forward"]["ms"],
        "backward_launches": sanity["backward"]["launches"],
        "backward_ms": sanity["backward"]["ms"],
    }
    checks: List[Dict[str, Any]] = []
    for key, expected in REFERENCE_SANITY.items():
        actual = computed[key]
        delta = actual - expected
        tolerance = MS_TOLERANCE if key.endswith("_ms") else 0.0
        checks.append(
            {
                "key": key,
                "computed": actual,
                "reference": expected,
                "delta": delta,
                "tolerance": tolerance,
                "ok": abs(delta) <= tolerance,
            }
        )
    return checks


def _family_rows(launches: Sequence[Launch]) -> List[Dict[str, Any]]:
    total_us = sum(row.gpu_us for row in launches)
    buckets: Dict[str, Dict[str, Any]] = {}
    for row in launches:
        bucket = buckets.setdefault(
            row.family, {"family": row.family, "launches": 0, "total_ms": 0.0}
        )
        bucket["launches"] += 1
        bucket["total_ms"] += row.gpu_us / 1000.0
    rows = list(buckets.values())
    for row in rows:
        row["share_pct"] = (
            row["total_ms"] / (total_us / 1000.0) * 100.0 if total_us else 0.0
        )
    rows.sort(key=lambda row: (-row["total_ms"], row["family"]))
    return rows


def _signature_rows(launches: Sequence[Launch]) -> List[Dict[str, Any]]:
    total_ms = sum(row.gpu_us for row in launches) / 1000.0
    buckets: Dict[str, Dict[str, Any]] = {}
    for row in launches:
        bucket = buckets.get(row.signature)
        if bucket is None:
            bucket = {
                "signature": row.signature,
                "op": row.op,
                "dims": row.dims,
                "family": row.family,
                "shape_class": row.shape_class,
                "launches": 0,
                "total_ms": 0.0,
                "dtype_variants": {},
            }
            buckets[row.signature] = bucket
        bucket["launches"] += 1
        bucket["total_ms"] += row.gpu_us / 1000.0
        variant = bucket["dtype_variants"].setdefault(
            str(row.dtypes),
            {"dtypes": row.dtypes, "launches": 0, "total_ms": 0.0},
        )
        variant["launches"] += 1
        variant["total_ms"] += row.gpu_us / 1000.0

    rows = list(buckets.values())
    for row in rows:
        variants = sorted(
            row["dtype_variants"].values(),
            key=lambda item: (-item["total_ms"], -item["launches"]),
        )
        for variant in variants:
            variant["share_pct"] = (
                variant["launches"] / row["launches"] * 100.0
                if row["launches"]
                else 0.0
            )
        dominant = variants[0]
        row["dtypes"] = dominant["dtypes"]
        row["dtype_stability"] = (
            dominant["launches"] / row["launches"] if row["launches"] else 0.0
        )
        row["dtype_variants"] = variants
        row["mean_ms"] = (
            row["total_ms"] / row["launches"] if row["launches"] else 0.0
        )
        row["share_pct"] = row["total_ms"] / total_ms * 100.0 if total_ms else 0.0
    rows.sort(key=lambda row: (-row["total_ms"], -row["launches"], row["op"]))
    return rows


def _aggregate_patterns(
    launches: Sequence[Launch],
    n: int,
) -> Tuple[Dict[Tuple[str, ...], Dict[str, Any]], int]:
    windows = len(launches) - n + 1
    buckets: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for index in range(max(windows, 0)):
        window = launches[index : index + n]
        key = tuple(row.signature for row in window)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = {
                "count": 0,
                "total_us": 0.0,
                "traffic_bytes": 0,
                "known_entries": 0,
                "total_entries": 0,
                "full_signatures": Counter(),
                "first_index": index,
            }
            buckets[key] = bucket
        bucket["count"] += 1
        bucket["total_us"] += sum(row.gpu_us for row in window)
        bucket["total_entries"] += n
        bucket["full_signatures"][
            tuple(row.full_signature for row in window)
        ] += 1
        for row in window:
            entry_bytes = _raw_tensor_bytes(row)
            if entry_bytes is None:
                continue
            bucket["traffic_bytes"] += entry_bytes
            bucket["known_entries"] += 1
    return buckets, windows


def _pattern_rows(
    launches: Sequence[Launch],
    n: int,
    min_ms: float,
    top: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    buckets, windows = _aggregate_patterns(launches, n)
    overlapping_sum_ms = sum(bucket["total_us"] for bucket in buckets.values()) / 1000.0
    rows: List[Dict[str, Any]] = []
    for key, bucket in buckets.items():
        count = bucket["count"]
        total_ms = bucket["total_us"] / 1000.0
        if total_ms < min_ms:
            continue
        dominant = bucket["full_signatures"].most_common(1)[0][1]
        stability = dominant / count if count else 0.0
        known = bucket["known_entries"]
        traffic = bucket["traffic_bytes"] if known else None
        roofline = (
            traffic / EFFECTIVE_BANDWIDTH_BYTES_PER_S * 1000.0
            if traffic is not None
            else None
        )
        window = launches[bucket["first_index"] : bucket["first_index"] + n]
        classes = [row.shape_class for row in window]
        rows.append(
            {
                "n": n,
                "count": count,
                "total_ms": total_ms,
                "ms_per_occurrence": total_ms / count if count else 0.0,
                "stability": stability,
                "stable": stability == 1.0,
                "op_sequence": [row.op for row in window],
                "dims_sequence": [row.dims for row in window],
                "dims_text": [str(row.dims) for row in window],
                "signature_sequence": list(key),
                "traffic_bytes": traffic,
                "addressable_gb": (
                    traffic / 1e9 if traffic is not None else None
                ),
                "roofline_ms": roofline,
                "saving_ms": (
                    total_ms - roofline if roofline is not None else None
                ),
                "known_numel_entries": known,
                "total_entries": bucket["total_entries"],
                "all_edge_major": all(item == "edge" for item in classes),
                "all_node_major": all(item == "node" for item in classes),
            }
        )
    rows.sort(
        key=lambda row: (-row["total_ms"], -row["count"], row["signature_sequence"])
    )
    rows = rows[:top]
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    stats = {
        "n": n,
        "windows": max(windows, 0),
        "distinct": len(buckets),
        "reported": len(rows),
        "overlapping_sum_ms": overlapping_sum_ms,
    }
    return rows, stats


def _shape_class_summary(launches: Sequence[Launch]) -> Dict[str, Any]:
    total_ms = sum(row.gpu_us for row in launches) / 1000.0
    classes: Dict[str, Dict[str, Any]] = {}
    for row in launches:
        bucket = classes.setdefault(
            row.shape_class,
            {
                "class": row.shape_class,
                "launches": 0,
                "total_ms": 0.0,
                "runs": 0,
                "max_run_launches": 0,
                "runs_ge2": 0,
                "launches_in_runs_ge2": 0,
                "ms_in_runs_ge2": 0.0,
            },
        )
        bucket["launches"] += 1
        bucket["total_ms"] += row.gpu_us / 1000.0

    runs: List[List[Launch]] = []
    current: List[Launch] = []
    for row in launches:
        if current and current[-1].shape_class == row.shape_class:
            current.append(row)
        else:
            if current:
                runs.append(current)
            current = [row]
    if current:
        runs.append(current)

    for run in runs:
        bucket = classes[run[0].shape_class]
        bucket["runs"] += 1
        bucket["max_run_launches"] = max(bucket["max_run_launches"], len(run))
        if len(run) >= 2:
            bucket["runs_ge2"] += 1
            bucket["launches_in_runs_ge2"] += len(run)
            bucket["ms_in_runs_ge2"] += sum(item.gpu_us for item in run) / 1000.0

    for bucket in classes.values():
        bucket["share_pct"] = (
            bucket["total_ms"] / total_ms * 100.0 if total_ms else 0.0
        )
    rows = sorted(
        classes.values(),
        key=lambda row: (-row["total_ms"], row["class"]),
    )
    for bucket in classes.values():
        bucket["total_ms"] = round(bucket["total_ms"], 6)
        bucket["ms_in_runs_ge2"] = round(bucket["ms_in_runs_ge2"], 6)
        bucket["share_pct"] = round(bucket["share_pct"], 6)
    return {"classes": rows, "runs": len(runs), "total_ms": round(total_ms, 6)}


def analyze(
    trace_path: Path,
    *,
    top: int,
    max_len: int,
    min_ms: float,
) -> Dict[str, Any]:
    events = _load_events(trace_path)
    launches, join = _build_launches(events)
    if not launches:
        raise RuntimeError(
            f"{trace_path}: no joined HIP launch -> kernel -> cpu_op entries"
        )

    sanity = _sanity(launches, events, join)
    families = _family_rows(launches)
    signatures = _signature_rows(launches)
    patterns: Dict[str, List[Dict[str, Any]]] = {}
    pattern_stats: Dict[str, Dict[str, Any]] = {}
    for n in range(2, max_len + 1):
        rows, stats = _pattern_rows(launches, n, min_ms, top)
        patterns[str(n)] = rows
        pattern_stats[str(n)] = stats

    report: Dict[str, Any] = {
        "trace_path": str(trace_path.resolve()),
        "generator": "myproject/tests/deepmd/profile/scan_fusion_patterns.py",
        "units": {"duration": "ms", "traffic": "bytes", "bandwidth": "GB/s"},
        "assumptions": {
            "effective_bandwidth_bytes_per_s": EFFECTIVE_BANDWIDTH_BYTES_PER_S,
            "bandwidth_gbps": EFFECTIVE_BANDWIDTH_BYTES_PER_S / 1e9,
            "bytes_per_element": BYTES_PER_ELEMENT,
            "streams_per_entry": STREAMS_PER_ENTRY,
            "numel_heuristic": (
                "element count of the largest tensor shape in cpu_op Input Dims"
            ),
            "node_major_size": NODE_MAJOR_SIZE,
            "edge_major_sizes": list(EDGE_MAJOR_SIZES),
            "region_suffixes": {
                "forward": DEFAULT_FORWARD_REGION_SUFFIX,
                "backward": DEFAULT_BACKWARD_REGION_SUFFIX,
            },
            "note": (
                "Summed kernel durations and n-gram windows overlap; totals "
                "attribute device time, they are not inference latency."
            ),
        },
        "join": join,
        "sanity": sanity,
        "families": families,
        "signatures": signatures,
        "signature_count": len(signatures),
        "patterns": patterns,
        "pattern_stats": pattern_stats,
        "shape_classes": _shape_class_summary(launches),
    }
    if trace_path.name == REFERENCE_TRACE_NAME:
        checks = _reference_checks(sanity)
        report["reference_checks"] = checks
        report["reference_ok"] = all(check["ok"] for check in checks)
    return report


def _fmt_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}"


def _print_header(report: Dict[str, Any]) -> None:
    sanity = report["sanity"]
    join = report["join"]
    print(f"trace: {report['trace_path']}")
    print(
        f"events: kernels {sanity['kernel_events']}  "
        f"({sanity['kernel_sum_ms']:.3f} ms summed); "
        f"launches {sanity['launch_events']} joined "
        f"({sanity['launch_sum_ms']:.3f} ms)"
    )
    print(
        f"join: {join['launch_api_events']} HIP launch API events -> "
        f"{join['joined_launches']} complete rows "
        f"(missing cpu_op {join['missing_cpu_op']}, "
        f"missing kernel {join['missing_kernel']}, missing ids {join['missing_ids']})"
    )
    forward = sanity["forward"]
    backward = sanity["backward"]
    outside = sanity["outside"]
    print(
        f"forward  region ({forward['suffix']}): {forward['launches']} launches, "
        f"{forward['ms']:.3f} ms"
    )
    print(
        f"backward region ({backward['suffix']}): {backward['launches']} launches, "
        f"{backward['ms']:.3f} ms"
    )
    print(
        f"outside both regions: {outside['launches']} launches, "
        f"{outside['ms']:.3f} ms"
    )
    sentinels = sanity["shape_sentinels"]
    print(
        "shape sentinels: "
        f"3000 in {sentinels['3000']} launches, "
        f"181838 in {sentinels['181838']}, "
        f"181836 in {sentinels['181836']}"
    )


def _print_reference_checks(report: Dict[str, Any]) -> None:
    checks = report.get("reference_checks")
    if not checks:
        return
    print("\n-- sanity checks vs documented dpa4 numbers --")
    print(f"  {'check':<18s} {'computed':>12s} {'reference':>12s} {'delta':>10s}")
    for check in checks:
        print(
            f"  {check['key']:<18s} {_fmt_number(check['computed']):>12s} "
            f"{_fmt_number(check['reference']):>12s} "
            f"{check['delta']:+10.3f}"
        )
    verdict = "OK" if report.get("reference_ok") else "DELTA"
    print(
        f"  [{verdict}] tolerance {MS_TOLERANCE} ms for durations, "
        "0 for counts"
    )


def _print_families(report: Dict[str, Any]) -> None:
    print("\n-- launch families --")
    print(f"  {'family':<12s} {'launches':>8s} {'total_ms':>10s} {'share':>7s}")
    for row in report["families"]:
        print(
            f"  {row['family']:<12s} {row['launches']:8d} "
            f"{row['total_ms']:10.3f} {row['share_pct']:6.2f}%"
        )


def _print_signatures(report: Dict[str, Any], top: int) -> None:
    rows = report["signatures"][:top]
    print(f"\n-- top {len(rows)} single-op signatures (op|dims) --")
    print(f"  {'total_ms':>9s} {'calls':>6s} {'mean_us':>9s} {'family':<12s} signature")
    for row in rows:
        print(
            f"  {row['total_ms']:9.3f} {row['launches']:6d} "
            f"{row['mean_ms'] * 1000.0:9.2f} {row['family']:<12s} "
            f"{row['signature']}"
        )
        if len(row["dtype_variants"]) > 1:
            variants = ", ".join(
                f"{variant['dtypes']} x{variant['launches']}"
                for variant in row["dtype_variants"]
            )
            print(f"        dtypes: {variants}")


def _pattern_traffic_text(row: Dict[str, Any]) -> str:
    known = row["known_numel_entries"]
    total = row["total_entries"]
    suffix = f" (numel known {known}/{total})" if known < total else ""
    if row["traffic_bytes"] is None:
        return f"traffic n/a{suffix}"
    return (
        f"addr {row['addressable_gb']:8.3f} GB  "
        f"roof {row['roofline_ms']:8.3f} ms  "
        f"save {row['saving_ms']:+9.3f} ms{suffix}"
    )


def _pattern_tag(row: Dict[str, Any]) -> str:
    tags: List[str] = []
    if row["all_edge_major"]:
        tags.append("edge-chain")
    elif row["all_node_major"]:
        tags.append("node-chain")
    if not row["stable"]:
        tags.append("UNSTABLE")
    return (" " + " ".join(tags)) if tags else ""


def _print_patterns(report: Dict[str, Any]) -> None:
    print(
        "\n-- adjacent launch patterns (n-grams of consecutive kernels) --"
    )
    for key in sorted(report["patterns"], key=int):
        rows = report["patterns"][key]
        stats = report["pattern_stats"][key]
        print(
            f"\n  n={key}: {stats['distinct']} distinct / {stats['windows']} windows, "
            f"window ms sum {stats['overlapping_sum_ms']:.3f} (overlapping)"
        )
        if not rows:
            print("    (none above --min-ms)")
            continue
        for row in rows:
            print(
                f"  {row['rank']:3d}. {row['count']:4d}x "
                f"{row['total_ms']:9.3f} ms  "
                f"{row['ms_per_occurrence']:7.3f} ms/occ  "
                f"stab {row['stability']:.2f}{_pattern_tag(row)}  "
                f"{_pattern_traffic_text(row)}"
            )
            sequence = " -> ".join(
                f"{op}{dims_text}"
                for op, dims_text in zip(row["op_sequence"], row["dims_text"])
            )
            print(f"       {sequence}")


def _print_shape_classes(report: Dict[str, Any]) -> None:
    summary = report["shape_classes"]
    print(
        "\n-- edge-major vs node-major attribution "
        f"(maximal same-class runs, {summary['runs']} runs) --"
    )
    print(
        f"  {'class':<7s} {'launches':>8s} {'total_ms':>10s} {'share':>7s} "
        f"{'runs':>5s} {'max_run':>7s} {'runs>=2':>7s} {'ms runs>=2':>11s}"
    )
    for row in summary["classes"]:
        print(
            f"  {row['class']:<7s} {row['launches']:8d} {row['total_ms']:10.3f} "
            f"{row['share_pct']:6.2f}% {row['runs']:5d} "
            f"{row['max_run_launches']:7d} {row['runs_ge2']:7d} "
            f"{row['ms_in_runs_ge2']:11.3f}"
        )


def _print_report(report: Dict[str, Any], top: int) -> None:
    _print_header(report)
    _print_reference_checks(report)
    _print_families(report)
    _print_signatures(report, top)
    _print_patterns(report)
    _print_shape_classes(report)
    print(
        "\nnote: a signature is op|Input Dims; stability compares the full "
        "op|dims|dtype signature."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Join HIP launches to their ATen ops and report adjacent launch "
            "patterns, family totals and bandwidth-roofline estimates."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trace", type=Path, required=True, help="Kineto JSON trace")
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="rows per table (signatures and n-grams)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=5,
        help="largest n-gram length to count",
    )
    parser.add_argument(
        "--min-ms",
        type=float,
        default=0.0,
        help="only report n-grams whose summed GPU time is at least this",
    )
    parser.add_argument("--json", type=Path, default=None, help="write JSON here")
    args = parser.parse_args(argv)

    if not args.trace.is_file():
        print(f"trace does not exist: {args.trace}", file=sys.stderr)
        return 2
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.max_len < 2:
        parser.error("--max-len must be at least 2")
    if args.min_ms < 0.0:
        parser.error("--min-ms must be non-negative")

    try:
        report = analyze(
            args.trace,
            top=args.top,
            max_len=args.max_len,
            min_ms=args.min_ms,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    _print_report(report, args.top)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON written to {args.json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
