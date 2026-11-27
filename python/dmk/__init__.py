"""Host side of the persistent megakernel, laid out like `Megakernels/megakernels`.

* :mod:`dmk.model_types` -- the value types a schedule is written in (work,
  per-CU tile configuration), and the device/kind constants.
* :mod:`dmk.utils` -- how many CUs, how work divides between them.
* :mod:`dmk.instructions` -- one dataclass per operator: ``serialize()`` to the
  32 words the runtime reads, ``cost()`` for the scheduler.
* :mod:`dmk.scheduler` -- assign instructions to CUs: a dependency-aware heap
  (``assign_dag_to_cus``) or round robin.
* :mod:`dmk.emitters` -- a model's instruction sequence: one instruction per
  16x16 output tile for the GEMM.
* :mod:`dmk.dispatch` -- hand a schedule to the runtime.
* :mod:`dmk.mk` -- the model kernel: globals plus the launch.
* :mod:`dmk.python_vm` -- a Python emulator of the instruction stream, so a
  schedule can be checked without a DCU.

The numeric opcodes belong to the C++ side (`ops::config`) and are mirrored in
:mod:`kernels.mk_stream`; this package decides who runs what, not how a word is
laid out.
"""

from . import (dispatch, emitters, instructions, mk, model_types, python_vm,
               scheduler, utils)
from .emitters import (AttentionEmitter, ElementwiseEmitter, EmbeddingEmitter,
                       GemmEmitter, ModelSchedule, OperatorEmitter, emitter_for)
from .instructions import Attention, Embedding, Elementwise, Gemm, Instruction, NoOp
from .mk import GemmGlobals, launch_gemm
from .model_types import CuPlan, CuTileConfig, Work
from .python_vm import emulate_gemm
from .scheduler import assign_dag_to_cus, linear_chain, round_robin_assign
from .utils import get_cu_count, plan_cus, split_units

__all__ = [
    "Attention",
    "AttentionEmitter",
    "CuPlan",
    "CuTileConfig",
    "Elementwise",
    "Embedding",
    "ElementwiseEmitter",
    "EmbeddingEmitter",
    "Gemm",
    "GemmEmitter",
    "GemmGlobals",
    "Instruction",
    "ModelSchedule",
    "NoOp",
    "OperatorEmitter",
    "Work",
    "assign_dag_to_cus",
    "dispatch",
    "emulate_gemm",
    "emitter_for",
    "emitters",
    "get_cu_count",
    "instructions",
    "launch_gemm",
    "linear_chain",
    "mk",
    "model_types",
    "plan_cus",
    "python_vm",
    "round_robin_assign",
    "scheduler",
    "split_units",
    "utils",
]
