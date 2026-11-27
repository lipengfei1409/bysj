"""Run the persistent GEMM end to end.

    python -m dmk.scripts.run_gemm 256 256 256 --cus 4           # DCU: launch + check
    python -m dmk.scripts.run_gemm 256 256 256 --cus 4 --emulate # host: Python VM only
    python -m dmk.scripts.run_gemm 256 256 256 --dry-run         # print the schedule

Three modes, one model (`demos/gemm.py`):

* ``--emulate`` runs the schedule through `python_vm`, so the split and the
  encoding are checked with no DCU and no build;
* ``--dry-run`` prints the per-CU rows and exits;
* default launches `csrc/dmk/gemm_launch.cpp` and reports the error and the
  millisecond per iteration.
"""

from __future__ import annotations

import argparse
import sys

from .. import dispatch, mk
from ..demos.gemm import GemmDemo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("m", type=int, help="rows of A / C")
    parser.add_argument("n", type=int, help="columns of B / C")
    parser.add_argument("k", type=int, help="shared dimension (one k-step per 16)")
    parser.add_argument("--cus", type=int, default=4, help="CUs to split across")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1,
                        help="launches to time (default 1)")
    parser.add_argument("--emulate", action="store_true",
                        help="run the schedule in Python instead of launching")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the schedule and exit")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.cus < 1:
        parser.error("--cus must be at least 1")
    try:
        model = GemmDemo(m=args.m, n=args.n, k=args.k, cus=args.cus)
        rows = model.rows()
    except ValueError as failure:
        print(f"bad shape: {failure}", file=sys.stderr)
        return 2

    if args.dry_run:
        from ..instructions import InstructionStream

        print(InstructionStream(rows).describe())
        return 0

    if not args.emulate and not dispatch.available():
        print(
            "launch unavailable: the persistent GEMM kernel is not built "
            "(csrc/dmk/gemm_launch.cpp -> kernels._C.dmk_gemm); use --emulate or "
            "--dry-run on a host",
            file=sys.stderr,
        )
        return 3

    device = "cpu" if args.emulate else args.device
    try:
        globals_ = model.globals(seed=args.seed, device=device)
    except (RuntimeError, AssertionError) as failure:
        print(f"cannot allocate on {device}: {failure}", file=sys.stderr)
        return 2

    if args.emulate:
        error = model.verify(globals_, emulated=True)
        print(f"emulated: max |C - A@B| = {error:.3e}")
        return 0 if error < 1e-3 else 1

    model.launch(globals_)
    error = model.verify(globals_)
    milliseconds = mk.time_gemm(globals_, args.iterations)
    print(f"persistent GEMM: max |C - A@B| = {error:.3e}, {milliseconds:.3f} ms/iter")
    return 0 if error < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
