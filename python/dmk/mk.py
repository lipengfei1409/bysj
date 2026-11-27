"""The model kernel: globals plus the launch (the reference's ``mk.py`` role).

`Megakernels/megakernels/mk.py` is a thin interpreter that loads a model-specific
builder and hands the schedule to the VM.  Ours has one model -- a dense GEMM --
and the same two pieces:

    globals_ = GemmGlobals.from_shapes(m, n, k)        # device tensors
    globals_.instructions = emitters.gemm_queues(...)  # the schedule
    launch_gemm(globals_)                                # one launch

`GemmGlobals` mirrors `BaseGlobals`: the tensors the kernel reads, the model
constants, and ``sm_count()`` -- here the number of CU rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import dispatch, emitters, python_vm
from .instructions import InstructionStream
from .model_types import GEMM_TILE


@dataclass
class GemmGlobals:
    """Everything one persistent-GEMM launch reads."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    instructions: torch.Tensor | None = None
    num_instructions: int = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.c.shape[0], self.c.shape[1], self.a.shape[1])

    def sm_count(self) -> int:
        return 0 if self.instructions is None else int(self.instructions.shape[0])

    @classmethod
    def from_shapes(cls, m: int, n: int, k: int, *, seed: int = 0,
                    device="cuda") -> "GemmGlobals":
        if m % GEMM_TILE or n % GEMM_TILE or k % GEMM_TILE:
            raise ValueError(
                "the persistent GEMM tiles by 16 and has no edge path, so M, N "
                f"and K must be multiples of 16 (got {m}x{n}x{k})"
            )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.randn(m, k, generator=generator).to(device)
        b = torch.randn(k, n, generator=generator).to(device)
        c = torch.zeros(m, n, device=device)
        return cls(a=a, b=b, c=c)

    def attach(self, rows: list[list]) -> "GemmGlobals":
        """Attach per-CU instruction rows produced by `emitters`."""
        stream = InstructionStream(rows) if not hasattr(rows, "words") else rows
        self.instructions = stream.tensor(device=self.a.device)
        self.num_instructions = stream.length
        return self


def launch_gemm(globals_: GemmGlobals, *, flat: bool | None = None) -> GemmGlobals:
    """Launch the persistent GEMM once (device).

    ``flat=True`` runs the role-free kernel on the same schedule; ``None`` follows
    ``DPA4_GEMM_FLAT``.  Both write ``globals_.c``.
    """
    if globals_.instructions is None or globals_.num_instructions <= 0:
        raise RuntimeError("GemmGlobals has no instruction stream: attach one first")
    dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b, globals_.c,
                         flat=flat)
    return globals_


def emulate_gemm(globals_: GemmGlobals) -> torch.Tensor:
    """Run the same schedule in Python -- no DCU, no build (see `python_vm`)."""
    if globals_.instructions is None:
        raise RuntimeError("GemmGlobals has no instruction stream: attach one first")
    return python_vm.emulate_gemm(globals_.instructions, globals_.a, globals_.b)


def verify(globals_: GemmGlobals, *, emulated: bool = False,
           rtol: float = 1e-4, atol: float = 1e-4) -> float:
    """Max absolute error against a dense ``a @ b``."""
    expected = globals_.a @ globals_.b
    if emulated:
        got = emulate_gemm(globals_)
    else:
        got = globals_.c
    return float((got.to(expected.dtype) - expected).abs().max())


def time_gemm(globals_: GemmGlobals, iterations: int = 1, *,
              flat: bool | None = None) -> float:
    """Milliseconds per launch, warmed up once."""
    launch_gemm(globals_, flat=flat)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iterations)):
        launch_gemm(globals_, flat=flat)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / max(1, iterations)


__all__ = ["GemmGlobals", "emulate_gemm", "launch_gemm", "time_gemm", "verify"]
