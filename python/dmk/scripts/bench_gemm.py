"""Benchmark one GEMM engine -- the workload a profiler needs.

`rocprofv3` wants a small, quiet program: no test framework, no per-shape bookkeeping,
just warm-up and a loop.  This is that program for the three engines in this tree:

    python -m dmk.scripts.bench_gemm dense  --shape 2048x2048x2048 --iterations 20
    python -m dmk.scripts.bench_gemm flat   --shape 1024x1024x1024 --cus 4
    python -m dmk.scripts.bench_gemm scheduled --shape 512x512x512 --cus 2

It prints one line per launch (milliseconds and TFLOP/s) so a profile can be matched to
a number, and with no `--iterations` it prints the launch plan only, which is what to
read before collecting counters.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from .. import dispatch, emitters, mk


def _parse_shape(text: str) -> tuple[int, int, int]:
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--shape expects MxNxK")
    try:
        m, n, k = (int(part) for part in parts)
    except ValueError as failure:
        raise argparse.ArgumentTypeError(f"--shape: {failure}") from failure
    return m, n, k


def _scheduled(m: int, n: int, k: int, cus: int, iterations: int, flat: bool) -> float:
    globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=0)
    globals_.attach(emitters.GemmEmitter(m, n, k).queues(cus))
    for _ in range(5):
        if flat:
            dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b,
                                 globals_.c, flat=True)
        else:
            dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b,
                                 globals_.c)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        if flat:
            dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b,
                                 globals_.c, flat=True)
        else:
            dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b,
                                 globals_.c)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _dense(m: int, n: int, k: int, iterations: int) -> float:
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    for _ in range(5):
        dispatch.launch_dense(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        dispatch.launch_dense(a, b)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("engine", choices=("dense", "flat", "scheduled"))
    parser.add_argument("--shape", type=_parse_shape, default=(512, 512, 512))
    parser.add_argument("--cus", type=int, default=4,
                        help="rows for the two persistent engines")
    parser.add_argument("--iterations", type=int, default=0,
                        help="launches to time; 0 prints the plan and exits")
    args = parser.parse_args(argv)

    m, n, k = args.shape
    if args.iterations > 0 and not torch.cuda.is_available():
        print("this needs a DCU: torch cannot see a device", file=sys.stderr)
        return 2
    flops = 2 * m * n * k
    plan = {"engine": args.engine, "shape": [m, n, k], "flops": flops}
    if args.engine != "dense":
        plan["cus"] = args.cus
        plan["instructions"] = (m // 16) * -(-(n // 16) // 2)
    print(plan, flush=True)
    if args.iterations <= 0:
        return 0

    started = time.perf_counter()
    if args.engine == "dense":
        milliseconds = _dense(m, n, k, args.iterations)
    else:
        milliseconds = _scheduled(m, n, k, args.cus, args.iterations,
                                  flat=args.engine == "flat")
    tflops = flops / (milliseconds * 1e9) if milliseconds else 0.0
    print(f"{args.engine}: {milliseconds:.3f} ms  {tflops:.2f} TFLOP/s "
          f"({time.perf_counter() - started:.1f} s wall)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
