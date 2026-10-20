from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from dcu_dpa4.catalog import OPERATORS


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class OpTrace:
    operator: str
    stage: str
    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]


class PlaceholderBackend:
    """Explicit operator boundary for native/reference implementations."""

    def run(self, operator: str, inputs: Mapping[str, torch.Tensor], outputs: Mapping[str, TensorSpec], *, stage: str):
        if operator not in OPERATORS:
            raise KeyError(operator)
        raise NotImplementedError(f"operator {operator!r} at stage {stage!r} is not implemented")


class ShapeBackend(PlaceholderBackend):
    """Execute a tensor-shape trace on meta tensors; no numerical prediction."""

    def __init__(self):
        self.trace: list[OpTrace] = []

    def run(self, operator, inputs, outputs, *, stage):
        if operator not in OPERATORS:
            raise KeyError(operator)
        specs = {name: TensorSpec(tuple(t.shape), t.dtype) for name, t in inputs.items() if t is not None}
        self.trace.append(OpTrace(operator, stage, specs, dict(outputs)))
        return {name: torch.empty(spec.shape, dtype=spec.dtype, device="meta") for name, spec in outputs.items()}
