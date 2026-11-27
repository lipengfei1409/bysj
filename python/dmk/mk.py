"""The model kernel: globals plus the launch (the reference's ``mk.py`` role).

In `Megakernels/megakernels/mk.py` this is a thin interpreter that loads a
model-specific builder and hands the schedule to the C++ VM.  Ours is the same
shape with one operator in it today -- the persistent GEMM -- because that is the
only op pack the runtime builds (`csrc/dmk/gemm_launch.cpp`).

    globals = GemmGlobals.from_shapes(m, n, k, cus)   # device tensors
    globals.instructions = queues.tensor(device=...)  # the schedule, as words
    run_gemm(globals)                                 # one launch

`GemmGlobals` mirrors `BaseGlobals` from the reference: the tensors the kernel
reads, the model constants, and `sm_count()` -- the number of CUs a launch fans
out to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from kernels.native import _extension


@dataclass
class GemmGlobals:
    """Everything one persistent-GEMM launch reads.

    ``a``/``b``/``c`` are the dense operands; ``instructions`` is the per-CU
    schedule produced by `shard`; ``num_instructions`` is the row length the
    kernel walks.
    """

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    instructions: torch.Tensor | None = None
    num_instructions: int = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.c.shape[0], self.c.shape[1], self.a.shape[1])

    def sm_count(self) -> int:
        """CUs the launch fans out to: one block per instruction row."""
        return 0 if self.instructions is None else int(self.instructions.shape[0])

    @classmethod
    def from_shapes(cls, m: int, n: int, k: int, *, seed: int = 0,
                    device: str | torch.device = "cuda") -> "GemmGlobals":
        """Allocate operands of the given shape with reproducible contents."""
        if m % 16 or n % 16 or k % 16:
            raise ValueError(
                "the persistent GEMM tiles by 16 and has no edge path, so M, N "
                f"and K must be multiples of 16 (got {m}x{n}x{k})"
            )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.randn(m, k, generator=generator).to(device)
        b = torch.randn(k, n, generator=generator).to(device)
        c = torch.zeros(m, n, device=device)
        return cls(a=a, b=b, c=c)


def run_gemm(globals: GemmGlobals) -> GemmGlobals:
    """Launch the persistent GEMM once; returns the globals for chaining."""
    if globals.instructions is None or globals.num_instructions <= 0:
        raise RuntimeError("GemmGlobals has no instruction stream: shard first")
    _extension("dmk_gemm").dmk_gemm(globals.instructions, globals.a, globals.b,
                                    globals.c)
    return globals


def verify_gemm(globals: GemmGlobals, *, rtol: float = 1e-4,
                atol: float = 1e-4) -> float:
    """Max absolute error against a dense product (the accuracy contract)."""
    torch.cuda.synchronize() if globals.a.is_cuda else None
    expected = globals.a @ globals.b
    return float((globals.c - expected).abs().max())


def time_gemm(globals: GemmGlobals, iterations: int = 1) -> float:
    """Milliseconds per launch, warmed up once."""
    run_gemm(globals)
    if globals.a.is_cuda:
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iterations)):
        run_gemm(globals)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / max(1, iterations)


__all__ = ["GemmGlobals", "run_gemm", "time_gemm", "verify_gemm"]
