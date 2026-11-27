"""Host side of the persistent megakernel: config, shard, run.

* :mod:`dmk.config` -- per-CU operator configuration (tile sizes for each kind),
  and the split of the model's total work into per-CU shares.
* :mod:`dmk.shard` -- turn a data flow plus those plans into one instruction
  queue per CU.
* :mod:`dmk.instructions` -- the instruction objects, one per operator, with
  ``serialize()`` (32 words) and ``cost()`` for the scheduler;
* :mod:`dmk.scheduler` -- assign instructions to CUs: a dependency-aware heap
  (`assign_dag_to_cus`) or round robin, the reference's two strategies;
* :mod:`dmk.mk` -- the model kernel: globals plus the launch;
* :mod:`dmk.main` -- the entry point: build the flow, shard it, launch it.

The encoding of an individual instruction lives in :mod:`kernels.mk_stream`,
which mirrors the opcodes in `csrc/ops/config.h`; this package decides *who does
what*, not how an instruction is laid out.
"""

from . import instructions, scheduler
from .config import CuPlan, TilePlan, Work, plan_cus
from .instructions import Attention, Embedding, Elementwise, Gemm, Instruction, NoOp
from .mk import GemmGlobals, run_gemm, verify_gemm
from .scheduler import DagNode, assign_dag_to_cus, linear_chain, round_robin_assign
from .shard import DataFlow, OpCall, Queues, shard

__all__ = [
    "Attention",
    "CuPlan",
    "DagNode",
    "DataFlow",
    "Elementwise",
    "Embedding",
    "Gemm",
    "GemmGlobals",
    "Instruction",
    "NoOp",
    "OpCall",
    "Queues",
    "TilePlan",
    "Work",
    "assign_dag_to_cus",
    "instructions",
    "linear_chain",
    "plan_cus",
    "round_robin_assign",
    "run_gemm",
    "scheduler",
    "shard",
    "verify_gemm",
]
