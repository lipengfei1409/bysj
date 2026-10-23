"""Deterministic model-level test inputs for the OMol25 Neo checkpoint."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from model.modeling.types import AtomicBatch, EdgeGraph
from model.system import DeepmdSystem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT.parent / "DPA4-OMol" / "DPA4-Neo-OMol25-100M-v20260820.pt"
)


def resolve_checkpoint() -> Path:
    """Resolve an explicit checkpoint or the documented local Neo default."""
    path = Path(os.environ.get("DPA4_MODEL_CHECKPOINT", DEFAULT_CHECKPOINT)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"model-test checkpoint not found: {path}; set DPA4_MODEL_CHECKPOINT"
        )
    return path.resolve()


def _directed_nonperiodic_edges(coords: np.ndarray, cutoff: float):
    n_frames, n_atoms, _ = coords.shape
    pairs = []
    edge_frames = []
    for frame in range(n_frames):
        delta = coords[frame, :, None, :] - coords[frame, None, :, :]
        distance2 = np.sum(delta * delta, axis=-1)
        src, dst = np.nonzero((distance2 > 1.0e-12) & (distance2 < cutoff * cutoff))
        pairs.extend((int(i + frame * n_atoms), int(j + frame * n_atoms)) for i, j in zip(src, dst))
        edge_frames.extend([frame] * len(src))
    if pairs:
        edge_index = np.asarray(pairs, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    return edge_index, np.asarray(edge_frames, dtype=np.int64)


def _make_batch_graph(coords, symbols, type_map, cutoff, *, device="cpu", charge_spin=None):
    coords = np.asarray(coords, dtype=np.float32)
    n_frames, n_atoms, _ = coords.shape
    if len(symbols) != n_atoms:
        raise ValueError("one element symbol is required per atom")
    try:
        atom_types = np.asarray([type_map.index(symbol) for symbol in symbols], dtype=np.int64)
    except ValueError as exc:
        raise ValueError("test structure contains an element outside the checkpoint type map") from exc
    atype = np.tile(atom_types, n_frames)
    atom_frames = np.repeat(np.arange(n_frames, dtype=np.int64), n_atoms)
    edge_index, edge_frames = _directed_nonperiodic_edges(coords, cutoff)
    n_edges = edge_index.shape[1]

    position_tensor = torch.as_tensor(coords.reshape(-1, 3), dtype=torch.float32, device=device)
    atype_tensor = torch.as_tensor(atype, dtype=torch.int64, device=device)
    atom_frame_tensor = torch.as_tensor(atom_frames, dtype=torch.int64, device=device)
    if charge_spin is None:
        conditions = None
    else:
        values = np.asarray(charge_spin, dtype=np.float32).reshape(1, 2)
        values = np.repeat(values, n_frames, axis=0)
        conditions = torch.as_tensor(values, dtype=torch.float32, device=device)
    batch = AtomicBatch(
        position_tensor, atype_tensor, atom_frame_tensor, n_frames,
        cells=None, charge_spin=conditions,
    )
    graph = EdgeGraph(
        torch.as_tensor(edge_index, dtype=torch.int64, device=device),
        torch.zeros((n_edges, 3), dtype=torch.int64, device=device),
        torch.as_tensor(edge_frames, dtype=torch.int64, device=device),
        torch.ones((n_edges,), dtype=torch.bool, device=device),
    )
    return batch, graph


def make_water_accuracy_case(config, *, device="cpu"):
    """A tiny deterministic non-periodic structure for oracle-parity smoke tests."""
    coords = np.asarray([[
        [0.0000, 0.0000, 0.0000],
        [0.9572, 0.0000, 0.0000],
        [-0.2390, 0.9270, 0.0000],
    ]], dtype=np.float32)
    symbols = ("O", "H", "H")
    system = DeepmdSystem(
        coords=coords,
        cells=None,
        atype=np.asarray([0, 1, 1], dtype=np.int64),
        type_map=("O", "H"),
        energy=None,
        force=None,
    )
    batch, graph = _make_batch_graph(
        coords, symbols, config.type_map, config.rcut,
        device=device, charge_spin=config.default_charge_spin,
    )
    return system, batch, graph


def make_latency_case(config, n_atoms: int, *, device):
    """Build a deterministic non-periodic graph with a configurable atom count."""
    if n_atoms < 2:
        raise ValueError("latency case requires at least two atoms")
    side = int(np.ceil(n_atoms ** (1.0 / 3.0)))
    coords = np.asarray([
        (2.0 * (i % side), 2.0 * ((i // side) % side), 2.0 * (i // (side * side)))
        for i in range(n_atoms)
    ], dtype=np.float32)[None, :, :]
    symbols = tuple(("H", "C", "N", "O")[i % 4] for i in range(n_atoms))
    return _make_batch_graph(
        coords, symbols, config.type_map, config.rcut,
        device=device, charge_spin=config.default_charge_spin,
    )
