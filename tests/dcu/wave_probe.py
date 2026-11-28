"""Wave-to-SIMD probe: does one wave fill one SIMD unit or all four?

The question this answers (see ``docs/dcu-architecture-simd.md`` §2): on a
GCN/CDNA CU a 64-lane wave is bound to one 16-lane SIMD unit and executes one
VALU instruction in four passes.  If that is true, a *single* wave reaches only
1/4 of the CU's FFMA rate, and the rate should scale 1x -> 2x -> 4x as the
block grows from 1 to 4 waves, then flatten at 4x (SIMD units are saturated;
extra waves only hide latency).

The probe reuses ``fma_probe.measure``: a pure register FFMA loop with ``nacc``
independent accumulators, timed on the host with CUDA events.  Everything runs
with ``blocks = 1`` so exactly one CU is involved; ``threads = 64 * waves``
selects the number of waves in that block.

Expected (wave-bound, the GCN/CDNA model)::

    waves   threads   rate vs 1 wave
      1       64          1.00
      2      128          2.00
      4      256          4.00
      8      512          4.00   (2 waves/SIMD, latency hiding only)
     16     1024          4.00

If a wave were spread across all four SIMD units, every row would be ~1.00.

Run on a DCU with HIP-enabled PyTorch::

    python3 tests/dcu/wave_probe.py
    python3 tests/dcu/wave_probe.py --waves 1,2,4,8 --iters 100000 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fma_probe  # noqa: E402


def sweep(
    waves=(1, 2, 4, 8, 16),
    *,
    iters: int = 200_000,
    nacc: int = 16,
    repeats: int = 10,
) -> list[dict]:
    """Measure TFLOP/s for each waves-per-CU point on a single CU."""
    rows = []
    for count in waves:
        threads = 64 * count
        tflops = fma_probe.measure(
            blocks=1, threads=threads, iters=iters, nacc=nacc, repeats=repeats
        )
        rows.append(
            {
                "waves": count,
                "threads": threads,
                "tflops": tflops,
                "gflops": tflops * 1e3,
            }
        )
    base = rows[0]["tflops"]
    for row in rows:
        row["ratio"] = row["tflops"] / base if base else float("nan")
    return rows


def verdict(rows: list[dict]) -> str:
    by_wave = {row["waves"]: row["ratio"] for row in rows}
    four = by_wave.get(4, float("nan"))
    if four >= 3.5:
        return (
            "wave-bound: one wave reaches 1/4 of the CU, four waves saturate it. "
            "A 64-lane wave executes on a single 16-lane SIMD in four passes."
        )
    if four <= 1.5:
        return (
            "wave-spread: one wave already reaches ~full CU rate, so it uses all "
            "four SIMD units. The 4-cycle model does not apply to this part."
        )
    return (
        f"inconclusive: 4 waves gave {four:.2f}x of 1 wave (expected ~4x or ~1x); "
        "check wave placement, clock throttling or register spills."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waves", default="1,2,4,8,16", help="comma-separated waves-per-CU points"
    )
    parser.add_argument("--iters", type=int, default=200_000)
    parser.add_argument("--nacc", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    waves = tuple(int(item) for item in args.waves.split(","))
    try:
        rows = sweep(waves, iters=args.iters, nacc=args.nacc, repeats=args.repeats)
    except fma_probe.ProbeError as error:
        print(f"wave_probe: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"rows": rows, "verdict": verdict(rows)}, indent=2))
        return 0

    print(f"{'waves':>5} {'threads':>8} {'GFLOP/s':>10} {'vs 1 wave':>10}")
    for row in rows:
        print(
            f"{row['waves']:>5} {row['threads']:>8} {row['gflops']:>10.1f} "
            f"{row['ratio']:>10.2f}"
        )
    print()
    print(verdict(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
