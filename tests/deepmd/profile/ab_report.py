"""Read the gain of a fused-dispatch A/B from profile summaries.

Answers one question in one table: *did the change actually buy device time?*
The three numbers that matter, in order of trust:

1. ``device ms`` -- the ``dpa4/inference.single_energy_force`` annotation, i.e.
   what the whole call cost the device.  This is the only number that pays.
2. ``delta`` -- against the first tag, which is treated as the baseline.
3. ``conv ms`` -- the fused convolution's own kernels, summed.  A change that
   speeds the conv up but not the total is lying about something else, and a
   conv total above the dense work it replaces (~690 ms at 3000 atoms) cannot
   win no matter how the rest behaves.

``so2_conv`` reads ``on`` only when the run's summary says the routing was
enabled, so a silent fallback to the dense path is visible instead of being
mistaken for "no gain".

Usage::

    python tests/deepmd/profile/ab_report.py "$OUT" base kc3 kc8
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

#: Device time of the fused segment the convolution replaces, at 3000 atoms.
DENSE_CONV_SEGMENT_MS = 690.0
#: Run-to-run spread of the device total on this harness, in ms.
NOISE_MS = 20.0


def _summary(root: Path, tag: str) -> Path | None:
    """Find one run's summary, accepting either ``<tag>`` or ``e2e_<tag>``.

    The directories carry an ``e2e_`` prefix while the conversation -- and the
    shell loop that produced them -- used bare tags, which is exactly how a
    correct set of runs reads as MISSING.
    """
    for name in (tag, tag if tag.startswith("e2e_") else f"e2e_{tag}"):
        paths = sorted(
            glob.glob(str(root / name / "dpa4_*" / "dpa4_inference_profile_summary.json"))
        )
        if paths:
            return Path(paths[-1])
    return None


def _device_ms(summary: dict) -> float | None:
    for event in summary.get("top_profiler_events") or []:
        if str(event.get("name", "")).startswith("dpa4/"):
            return float(event.get("self_device_us", 0.0)) / 1e3
    return None


def _conv_ms(summary: dict) -> float:
    kernels = (summary.get("trace_analysis") or {}).get("top_device_kernels") or []
    return sum(float(k.get("duration_us", 0.0)) for k in kernels if "so2_conv" in str(k.get("name", ""))) / 1e3


def report(root: Path, tags: list[str]) -> list[dict]:
    rows: list[dict] = []
    for tag in tags:
        path = _summary(root, tag)
        if path is None:
            rows.append(
                {
                    "tag": tag,
                    "missing": True,
                    "expected": str(
                        root / (tag if tag.startswith("e2e_") else f"e2e_{tag}")
                        / "dpa4_*/dpa4_inference_profile_summary.json"
                    ),
                }
            )
            continue
        summary = json.loads(path.read_text())
        so2 = summary.get("so2_conv") or {}
        defines = str(so2.get("defines", ""))
        rows.append(
            {
                "tag": tag,
                "device_ms": _device_ms(summary),
                "conv_ms": _conv_ms(summary),
                "launches": summary.get("device_event_count"),
                "so2_enabled": bool(so2.get("enabled")),
                "so2_reason": str(so2.get("reason", "")),
                "defines": defines,
                "path": str(path),
            }
        )
    return rows


def verdict(rows: list[dict]) -> list[str]:
    """Turn the table into the sentence a reader needs."""
    present = [row for row in rows if not row.get("missing") and row.get("device_ms")]
    if len(present) < 2:
        return ["need at least two runs with a device total to compare"]
    base = present[0]["device_ms"]
    lines = []
    for row in present[1:]:
        delta = row["device_ms"] - base
        if not row["so2_enabled"]:
            lines.append(f"{row['tag']}: routed OFF ({row['so2_reason'] or 'no reason recorded'})")
            continue
        if abs(delta) <= NOISE_MS:
            lines.append(f"{row['tag']}: {delta:+.1f} ms -- inside the {NOISE_MS:.0f} ms noise floor")
        elif delta < 0:
            lines.append(f"{row['tag']}: {delta:+.1f} ms ({base / row['device_ms']:.3f}x) -- net win")
        else:
            lines.append(f"{row['tag']}: {delta:+.1f} ms -- slower than the baseline")
        if row["conv_ms"] > DENSE_CONV_SEGMENT_MS:
            lines.append(
                f"    conv alone is {row['conv_ms']:.1f} ms against {DENSE_CONV_SEGMENT_MS:.0f} ms "
                "of dense work it replaces: it cannot pay for itself yet"
            )
    return lines


def _print(rows: list[dict]) -> None:
    print(
        f"{'tag':8s} {'device ms':>10s} {'delta':>8s} {'conv ms':>9s} "
        f"{'launches':>9s}  routed  defines"
    )
    base = next((row["device_ms"] for row in rows if row.get("device_ms")), None)
    for row in rows:
        if row.get("missing"):
            print(f"{row['tag']:8s} {'MISSING':>10s}  (no summary under {row['expected']})")
            continue
        total = row.get("device_ms") or 0.0
        delta = f"{total - base:+.1f}" if base else "-"
        print(
            f"{row['tag']:8s} {total:10.1f} {delta:>8s} {row['conv_ms']:9.1f} "
            f"{row.get('launches') or 0:9d}  "
            f"{'on' if row['so2_enabled'] else 'off':6s} {row.get('defines', '')}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory holding the e2e_<tag> runs")
    parser.add_argument("tags", nargs="+", help="tag names, first one is the baseline")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = report(args.root, args.tags)
    _print(rows)
    lines = verdict(rows)
    if lines:
        print("\n-- verdict --")
        for line in lines:
            print("  " + line)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": rows, "verdict": lines}, indent=2) + "\n")
    missing = [row for row in rows if row.get("missing")]
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
