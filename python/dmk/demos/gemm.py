"""The GEMM demo: the reference's `llama.py` role, with one dense product.

A demo is what `mk.py` loads: it knows a model's shapes, builds the instruction
sequence for it, and says how to verify the result.  Ours is a single dense GEMM
because that is the op pack the runtime builds (`csrc/dmk/gemm_launch.cpp`).

    model = GemmDemo(m=256, n=256, k=256, cus=4)
    globals_ = model.globals(device="cuda")
    model.attach(globals_)          # instruction rows -> int32 tensor
    model.launch(globals_)          # one persistent launch
    model.verify(globals_)          # max |C - A @ B|
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .. import generators, mk


@dataclass
class GemmDemo:
    """A dense ``C = A @ B`` as one persistent kernel launch."""

    m: int
    n: int
    k: int
    cus: int = 4

    def rows(self):
        """The per-CU instruction rows for this shape."""
        return generators.gemm_queues(self.m, self.n, self.k, self.cus)

    def globals(self, *, seed: int = 0, device="cuda") -> mk.GemmGlobals:
        """Allocate the operands and attach this shape's schedule."""
        return mk.GemmGlobals.from_shapes(self.m, self.n, self.k, seed=seed,
                                          device=device).attach(self.rows())

    def launch(self, globals_: mk.GemmGlobals) -> mk.GemmGlobals:
        return mk.launch_gemm(globals_)

    def emulate(self, globals_: mk.GemmGlobals) -> torch.Tensor:
        return mk.emulate_gemm(globals_)

    def verify(self, globals_: mk.GemmGlobals, *, emulated: bool = False) -> float:
        return mk.verify(globals_, emulated=emulated)


__all__ = ["GemmDemo"]
