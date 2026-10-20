from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AtomicBatch:
    positions: torch.Tensor  # [N,3], flattened frames
    atype: torch.Tensor  # [N]
    frame_index: torch.Tensor  # [N], frame membership
    n_frames: int
    cells: torch.Tensor | None = None  # [F,3,3]
    charge_spin: torch.Tensor | None = None  # [F,2] or broadcastable [2]

    def validate(self):
        n = self.positions.shape[0]
        if self.positions.shape != (n, 3) or self.atype.shape != (n,) or self.frame_index.shape != (n,):
            raise ValueError("batch requires positions [N,3], atype [N], frame_index [N]")
        if self.n_frames <= 0:
            raise ValueError("n_frames must be positive")
        if self.atype.dtype != torch.int64 or self.frame_index.dtype != torch.int64:
            raise TypeError("atype and frame_index must be int64")
        if self.cells is not None and self.cells.shape != (self.n_frames, 3, 3):
            raise ValueError("cells must have shape [F,3,3]")
        if self.charge_spin is not None:
            if self.charge_spin.shape not in ((2,), (1, 2), (self.n_frames, 2)):
                raise ValueError("charge_spin must have shape [2], [1,2] or [F,2]")
            if self.charge_spin.dtype == torch.bool or not (
                self.charge_spin.is_floating_point()
                or self.charge_spin.dtype in (torch.int32, torch.int64)
            ):
                raise TypeError("charge_spin must have a real numeric dtype")
        tensors = (self.atype, self.frame_index) + (() if self.cells is None else (self.cells,)) + (() if self.charge_spin is None else (self.charge_spin,))
        if any(t.device != self.positions.device for t in tensors):
            raise ValueError("batch tensors must share a device")


@dataclass
class EdgeGraph:
    edge_index: torch.Tensor  # [2,E]: src, dst
    shifts: torch.Tensor  # [E,3], integer periodic images
    frame_index: torch.Tensor  # [E]
    mask: torch.Tensor  # [E], bool, including capacity padding

    def validate(self):
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        e = self.edge_index.shape[1]
        if self.shifts.shape != (e, 3) or self.frame_index.shape != (e,) or self.mask.shape != (e,):
            raise ValueError("edge graph tensor shapes disagree")
        if any(t.dtype != torch.int64 for t in (self.edge_index, self.shifts, self.frame_index)) or self.mask.dtype != torch.bool:
            raise TypeError("graph indices/shifts must be int64 and mask bool")
        if any(t.device != self.edge_index.device for t in (self.shifts, self.frame_index, self.mask)):
            raise ValueError("graph tensors must share a device")


@dataclass
class ModelOutput:
    energy: torch.Tensor  # [F,1]
    atom_energy: torch.Tensor  # [N,1]
    force: torch.Tensor | None  # [N,3]
    virial: torch.Tensor | None  # [F,3,3]


@dataclass
class EdgeFeatures:
    """Numerical front-end boundary, before Wigner/GIE and message passing."""
    vec: torch.Tensor  # [E,3]
    distance: torch.Tensor  # [E,1]
    unit: torch.Tensor  # [E,3]
    rbf: torch.Tensor  # [E,R]
    env: torch.Tensor  # [E,1]
    type_feat: torch.Tensor  # [N,C]
    pair: torch.Tensor  # [E,C], zero on padding
    degree: torch.Tensor  # [N]
    inv_sqrt: torch.Tensor  # [N,1,1]
    radial: torch.Tensor  # [E,L0+1,C], before type fusion
    radial_fused: torch.Tensor  # [E,L0+1,C]
