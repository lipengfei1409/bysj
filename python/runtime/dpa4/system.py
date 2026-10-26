"""Load fixed-atom-count DeePMD ``deepmd/npy`` systems for model validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DeepmdSystem:
    coords: np.ndarray  # [F, N, 3], Å
    cells: np.ndarray | None  # [F, 3, 3], Å
    atype: np.ndarray  # [N], indices into type_map
    type_map: tuple[str, ...]
    energy: np.ndarray | None  # [F, 1], eV
    force: np.ndarray | None  # [F, N, 3], eV/Å


def load_deepmd_system(
    directory: str | Path, *, set_name: str = "set.000", max_frames: int | None = None
) -> DeepmdSystem:
    """Read one set of a fixed-composition DeePMD NPY system."""
    directory = Path(directory).expanduser().resolve()
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    atype = np.loadtxt(directory / "type.raw", dtype=np.int64, ndmin=1).reshape(-1)
    type_map_path = directory / "type_map.raw"
    type_map = (
        tuple(type_map_path.read_text().split()) if type_map_path.exists() else ()
    )
    set_dir = directory / set_name
    coords_raw = np.load(set_dir / "coord.npy")
    if coords_raw.ndim != 2 or coords_raw.shape[1] != 3 * len(atype):
        raise ValueError("coord.npy must have shape [F, 3*N] matching type.raw")
    if max_frames is not None:
        coords_raw = coords_raw[:max_frames]
    nframes = coords_raw.shape[0]
    coords = coords_raw.reshape(nframes, len(atype), 3)

    box_path = set_dir / "box.npy"
    cells = None
    if box_path.exists():
        cells_raw = np.load(box_path)[:nframes]
        if cells_raw.shape != (nframes, 9):
            raise ValueError("box.npy must have shape [F, 9]")
        cells = cells_raw.reshape(nframes, 3, 3)

    energy_path = set_dir / "energy.npy"
    energy = None
    if energy_path.exists():
        energy_raw = np.load(energy_path)[:nframes]
        if energy_raw.size != nframes:
            raise ValueError("energy.npy must have one value per frame")
        energy = energy_raw.reshape(nframes, 1)

    force_path = set_dir / "force.npy"
    force = None
    if force_path.exists():
        force_raw = np.load(force_path)[:nframes]
        if force_raw.size != nframes * len(atype) * 3:
            raise ValueError("force.npy must have shape [F, 3*N]")
        force = force_raw.reshape(nframes, len(atype), 3)

    return DeepmdSystem(coords, cells, atype, type_map, energy, force)
