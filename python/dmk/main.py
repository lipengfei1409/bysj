"""Entry point for the persistent megakernel: build the flow, shard it, run it.

    python -m dmk.main --cus 4 --iterations 10            # run (needs a DCU + dmk build)
    python -m dmk.main --cus 3 --dry-run                  # print the per-CU queues
    python -m dmk.main --cus 2 --dry-run --json           # ... machine-readable

Pipeline, in the order the module names suggest:

1. `build_flow`  -- describe what the model computes as a list of operator calls
   (this is the tiny reference flow; a real model would derive it from the
   graph-lowered FX dump),
2. `shard.shard` -- split that flow into one instruction queue per CU, with the
   per-CU tile sizes from `config.plan_cus`,
3. `launch`      -- hand the queues to the runtime.

`launch` needs the compiled megakernel, which is not part of the default build
(`csrc/dmk/megakernel.cpp` has no extension yet), so it reports that clearly and
`--dry-run` is the working path on a host without a DCU.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import config
from .shard import DataFlow, OpCall, Queues, shard as shard_flow


#: The reference flow: the shape of the profiled 3000-atom box, in the units
#: `config.Work` uses.  Numbers are "enough to exercise the split", not a claim
#: about the model's real work distribution.
def build_flow(*, elementwise_elements: int = 181_838 * 640,
               gemm_tiles: int = 181_838,
               embedding_rows: int = 3_000,
               attention_queries: int = 3_000,
               head_dim: int = 64) -> DataFlow:
    """Describe the compute graph as operator calls."""
    return DataFlow(
        [
            OpCall(config.ELEMENTWISE, elementwise_elements),
            OpCall(config.GEMM, gemm_tiles, {"k_tiles": 4}),
            OpCall(config.EMBEDDING, embedding_rows),
            OpCall(config.ATTENTION, attention_queries, {"dim": head_dim}),
        ]
    )


def launch(queues: Queues, *, iterations: int = 1, device=None):
    """Run the queues on the megakernel.

    The runtime entry point does not exist yet -- `csrc/dmk/megakernel.cpp` is
    built into no extension, and the op pack it launches still needs the model's
    Globals.  Until then this raises with what is missing instead of pretending
    to have run.
    """
    try:
        from kernels import dmk as runtime  # noqa: F401  (expected to be absent today)
    except ImportError as exc:  # pragma: no cover - depends on the build
        raise RuntimeError(
            "the megakernel runtime is not built: csrc/dmk/megakernel.cpp is not "
            "part of any extension yet, so --dry-run is the available path "
            f"({exc})"
        ) from exc
    raise RuntimeError(
        "kernels.dmk exists but the launch entry point is not wired to "
        "shard.Queues yet"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cus", type=int, default=4,
                        help="number of CUs (workgroups) to split across")
    parser.add_argument("--iterations", type=int, default=1,
                        help="how many times to run the queues")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and split the queues, print them, run nothing")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--device", default=None,
                        help="device for the instruction tensor, e.g. cuda or cpu")
    args = parser.parse_args(argv)

    if args.cus < 1:
        parser.error("--cus must be at least 1")

    flow = build_flow()
    queues = shard_flow(flow, args.cus)
    work = flow.work()

    if args.json:
        payload = {
            "cus": queues.blocks,
            "length": queues.length,
            "work": {kind: work.slots(kind) for kind in work.kinds()},
            "counts": queues.counts,
            "tiles": [
                {
                    "cu": plan.cu,
                    "elementwise": plan.tiles.elementwise,
                    "gemm": [plan.tiles.gemm_m, plan.tiles.gemm_n],
                    "embedding": plan.tiles.embedding,
                    "attention": plan.tiles.attention_query,
                }
                for plan in queues.plans
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(queues.describe())

    if args.dry_run:
        return 0

    try:
        launch(queues, iterations=args.iterations, device=args.device)
    except RuntimeError as error:
        print(f"\nlaunch unavailable: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
