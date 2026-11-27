"""Host side of the persistent megakernel: config, shard, run.

* :mod:`dmk.config` -- per-CU operator configuration (tile sizes for each kind),
  and the split of the model's total work into per-CU shares.
* :mod:`dmk.shard` -- turn a data flow plus those plans into one instruction
  queue per CU.
* :mod:`dmk.main` -- the entry point: build the flow, shard it, launch it.

The encoding of an individual instruction lives in :mod:`kernels.mk_stream`,
which mirrors the opcodes in `csrc/ops/config.h`; this package decides *who does
what*, not how an instruction is laid out.
"""

from .config import CuPlan, TilePlan, Work, plan_cus
from .shard import DataFlow, OpCall, Queues, shard

__all__ = [
    "CuPlan",
    "DataFlow",
    "OpCall",
    "Queues",
    "TilePlan",
    "Work",
    "plan_cus",
    "shard",
]
