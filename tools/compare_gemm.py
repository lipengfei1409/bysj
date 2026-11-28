#!/usr/bin/env python3
"""Time every GEMM engine in this tree against rocBLAS on one shape.

    python3 tools/compare_gemm.py --shape 2048x2048x2048 --iterations 20

Engines, all writing a 16x16-tile product of the same shape:

    rocBLAS     torch.matmul -- the vendor library, the number to beat
    dense       csrc/gemm.cpp: 64x64 block tile, ping-pong staging
    scheduled   csrc/dmk/megakernel.cpp: instruction stream, controller/loader/storer
    flat        csrc/dmk/gemm_flat.cpp: instruction stream, one role, double buffered

The script exists because the interesting comparison is a *table*, and because the two
persistent engines take a different path each time a checkout is out of step with the
kernel -- so an engine that is missing prints as `n/a` instead of failing the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from dmk import dispatch, emitters, mk  # noqa: E402


def _parse_shape(text: str) -> tuple[int, int, int]:
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--shape expects MxNxK")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _time(call, iterations: int) -> float:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _scheduled(m: int, n: int, k: int, cus: int, flat: bool | None):
    """A callable for one persistent engine, or None when it is not in this build."""
    if flat is False and not dispatch.available(flat=False):
        return None
    if flat is True and not dispatch.flat_available():
        return None
    globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=0)
    globals_.attach(emitters.GemmEmitter(m, n, k).queues(cus))

    def call():
        if flat:
            dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b,
                                 globals_.c, flat=True)
        else:
            dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b,
                                 globals_.c)

    return call


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shape", type=_parse_shape, default=(2048, 2048, 2048))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cus", type=int, default=4,
                        help="rows for the instruction-stream engines")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("this needs a DCU", file=sys.stderr)
        return 2

    m, n, k = args.shape
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    flops = 2 * m * n * k

    engines: list[tuple[str, object]] = [("rocBLAS", lambda: a @ b)]
    if hasattr(dispatch, "launch_dense") and dispatch.dense_available():
        engines.append(("dense", lambda: dispatch.launch_dense(a, b)))
    for name, flat in (("scheduled", False), ("flat", True)):
        call = _scheduled(m, n, k, args.cus, flat)
        if call is not None:
            engines.append((name, call))

    baseline = None
    print(f"\n{m}x{n}x{k}  ({flops / 1e9:.2f} GFLOP, {args.iterations} iterations)")
    print(f"  {'engine':10s} {'ms':>9s} {'TFLOP/s':>9s} {'vs rocBLAS':>11s}")
    for name, call in engines:
        milliseconds = _time(call, args.iterations)
        tflops = flops / (milliseconds * 1e9)
        if name == "rocBLAS":
            baseline = milliseconds
        ratio = f"{baseline / milliseconds:10.3f}x" if baseline else "         —"
        print(f"  {name:10s} {milliseconds:9.3f} {tflops:9.2f} {ratio:>11s}")
    missing = [name for name in ("dense", "scheduled", "flat")
               if name not in {engine for engine, _ in engines}]
    if missing:
        print(f"  (not in this build: {', '.join(missing)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
