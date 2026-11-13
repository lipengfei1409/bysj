"""Three hotspot profiles: single-card DPA4 eval, single-/multi-card LAMMPS.

See README.md beside this script for commands, prerequisites and interpretation.
Direct inference defaults to memory-based periodic replication (70% of available
memory), followed by 5 same-size warmup calls and exactly one recorded eval.
Use --fixed-input to disable scaling. Model, type map and input must agree.
LAMMPS runs real MD for 20 steps by default and requires the PTExpt backend's
DP_PROFILER window support: skip 5 model calls, then capture 10. One rank uses
one GPU; multiple ranks require MPI and a with-comm .pt2 model. --world-size
must match the requested GPU count when provided. Rank traces are independent.
These profiles locate hotspots; collection wall time is not inference latency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "python"))
_MIN_BYTES_PER_ATOM = 64 * 1024
_MAX_SIZING_PROBES = 6


def _normalize_system(coords: Any, atom_types: Any, cells: Any):
    coords = np.asarray(coords)
    atom_types = np.asarray(atom_types)
    if coords.ndim == 2 and coords.shape[-1] == 3:
        coords = coords[None, ...]
    elif coords.ndim == 2 and coords.shape[-1] % 3 == 0:
        coords = coords.reshape(coords.shape[0], -1, 3)
    if coords.ndim != 3 or coords.shape[0] < 1 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (atoms, 3) or (frames, atoms, 3)")
    coords = np.asarray(coords[:1], dtype=np.float64)

    if atom_types.ndim == 1:
        atom_types = atom_types[None, ...]
    if atom_types.ndim != 2 or atom_types.shape[0] < 1:
        raise ValueError("atom_types must have shape (atoms,) or (frames, atoms)")
    atom_types = np.asarray(atom_types[:1], dtype=np.int32)
    if atom_types.shape[1] != coords.shape[1]:
        raise ValueError(
            f"coords has {coords.shape[1]} atoms but atom_types has "
            f"{atom_types.shape[1]} entries"
        )

    if cells is None:
        normalized_cells = None
    else:
        cells = np.asarray(cells, dtype=np.float64)
        if cells.shape == (3, 3):
            cells = cells.reshape(1, 9)
        elif cells.ndim == 3 and cells.shape[1:] == (3, 3):
            cells = cells.reshape(cells.shape[0], 9)
        elif cells.ndim == 1 and cells.size == 9:
            cells = cells.reshape(1, 9)
        if cells.ndim != 2 or cells.shape[0] < 1 or cells.shape[1] != 9:
            raise ValueError("cells must have shape (9,), (frames, 9), or (3, 3)")
        normalized_cells = np.asarray(cells[:1], dtype=np.float64)
    return coords, atom_types, normalized_cells


def _load_npz_system(path: Path):
    with np.load(path) as sample:
        missing = {"coords", "atom_types"} - set(sample.files)
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {sorted(missing)}")
        cells = sample["cells"] if "cells" in sample else None
        return _normalize_system(sample["coords"], sample["atom_types"], cells)


def _read_lammps_system(path: Path):
    """Read wrapped coordinates; optional image flags are not coordinates."""
    bounds = {}
    atoms = []
    layout = None
    in_atoms = False
    layouts = {"atomic": (5, 1, 2), "charge": (6, 1, 3),
               "molecular": (6, 2, 3), "full": (7, 2, 4)}
    for line in path.read_text().splitlines():
        body, _, comment = line.partition("#")
        fields = body.split()
        if not fields:
            continue
        if fields[-3:] == ["xy", "xz", "yz"]:
            if any(abs(float(value)) > 1.0e-12 for value in fields[:3]):
                raise ValueError("triclinic LAMMPS boxes require a cell-aware NPZ")
            continue
        for axis in ("x", "y", "z"):
            if len(fields) == 4 and fields[-2:] == [f"{axis}lo", f"{axis}hi"]:
                bounds[axis] = (float(fields[0]), float(fields[1]))
        if fields[0] == "Atoms":
            style = comment.strip().split()[0] if comment.strip() else ""
            if style not in layouts:
                raise ValueError("unsupported or missing atom style; use Atoms # atomic/charge/molecular/full")
            layout = layouts[style]
            in_atoms = True
            continue
        if not in_atoms:
            continue
        if not fields[0].lstrip("+-").isdigit():
            break
        width, type_col, coord_col = layout
        if len(fields) not in (width, width + 3):
            raise ValueError(f"invalid Atoms row: expected {width} columns with optional image flags")
        atom_id, atom_type = int(fields[0]), int(fields[type_col])
        if atom_id <= 0 or atom_type <= 0:
            raise ValueError("LAMMPS atom IDs and types must be positive")
        x, y, z = map(float, fields[coord_col:coord_col + 3])
        atoms.append((atom_id, atom_type, x, y, z))
    if len(bounds) != 3 or not atoms:
        raise ValueError(f"could not parse orthorhombic LAMMPS geometry from {path}")
    atoms.sort(key=lambda row: row[0])
    if len({row[0] for row in atoms}) != len(atoms):
        raise ValueError("duplicate LAMMPS atom IDs")
    coords = np.asarray([[row[2], row[3], row[4]] for row in atoms], dtype=np.float64)
    atom_types = np.asarray([row[1] - 1 for row in atoms], dtype=np.int32)
    lengths = [bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z")]
    if any(length <= 0 for length in lengths):
        raise ValueError("LAMMPS box lengths must be positive")
    cells = np.diag(lengths).reshape(1, 9)
    return _normalize_system(coords, atom_types, cells)


def _parse_triplet(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("replication must be x,y,z integers") from error
    if len(parts) != 3 or any(item < 1 for item in parts):
        raise argparse.ArgumentTypeError("replication must be three positive integers")
    return parts


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    try:
        ids = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU ids must be comma-separated integers") from error
    if not ids or any(index < 0 for index in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("GPU ids must be nonnegative and unique")
    return ids


def _balanced_replication_factors(max_replica_product: int) -> tuple[int, int, int]:
    """Maximize replicated cells under a limit, preferring a compact supercell."""
    limit = max(1, int(max_replica_product))
    best = (1, 1, 1)
    best_product = 1
    best_aspect = float("inf")
    max_x = int(round(limit ** (1.0 / 3.0))) + 2
    for x in range(1, max_x + 1):
        max_y = int(math.sqrt(limit / x)) + 1
        for y in range(x, max_y + 1):
            z = limit // (x * y)
            if z < y:
                continue
            product = x * y * z
            aspect = z / x
            if product > best_product or (product == best_product and aspect < best_aspect):
                best = (x, y, z)
                best_product = product
                best_aspect = aspect
    return best


def _replicate_system(system, factors: tuple[int, int, int]):
    coords, atom_types, cells = system
    rx, ry, rz = factors
    if factors == (1, 1, 1) or cells is None:
        return coords.copy(), atom_types.copy(), None if cells is None else cells.copy()
    cell = cells[0].reshape(3, 3)
    coords_copies = []
    type_copies = []
    for ix in range(rx):
        for iy in range(ry):
            for iz in range(rz):
                shift = ix * cell[0] + iy * cell[1] + iz * cell[2]
                coords_copies.append(coords[0] + shift)
                type_copies.append(atom_types[0])
    super_coords = np.concatenate(coords_copies, axis=0)[None, ...]
    super_types = np.concatenate(type_copies, axis=0)[None, ...]
    super_cell = np.diag([rx, ry, rz]) @ cell
    return super_coords, super_types, super_cell.reshape(1, 9)


def generate_test_data(source_system, factors: tuple[int, int, int]):
    """Build a reusable periodic test system by replicating supplied geometry."""
    try:
        factors = tuple(factors)
    except TypeError as error:
        raise ValueError("replication must contain three positive integers") from error
    if len(factors) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < 1
        for value in factors
    ):
        raise ValueError("replication must contain three positive integers")
    factors = tuple(int(value) for value in factors)

    coords, atom_types, cells = source_system
    if cells is None:
        raise ValueError("periodic test-data generation requires a source cell")
    if np.any(atom_types < 0):
        raise ValueError("atom type indices must be nonnegative")
    cell = cells[0].reshape(3, 3)
    if not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) <= 1.0e-12:
        raise ValueError("source cell must be finite and nonsingular")
    if not np.isfinite(coords).all():
        raise ValueError("source coordinates must be finite")
    return _replicate_system((coords, atom_types, cells), factors)


def _write_generated_npz(output_path: Path, system) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coords, atom_types, cells = system
    try:
        stream = output_path.open("xb")
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite generated input: {output_path}") from None
    try:
        with stream:
            np.savez_compressed(
                stream,
                coords=coords,
                atom_types=atom_types,
                cells=cells,
            )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return output_path


def _outputs_to_arrays(outputs):
    if isinstance(outputs, dict):
        energy = outputs.get("energy", outputs.get("energy_redu"))
        force = outputs.get("force", outputs.get("energy_derv_r"))
        virial = outputs.get("virial", outputs.get("virial_redu"))
    else:
        values = tuple(outputs)
        energy = values[0] if len(values) > 0 else None
        force = values[1] if len(values) > 1 else None
        virial = values[2] if len(values) > 2 else None
    if energy is None or force is None:
        raise RuntimeError("DeepMD inference did not return energy and force")
    arrays = {}
    for name, value in (("energy", energy), ("force", force), ("virial", virial)):
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise RuntimeError(f"DeepMD inference returned non-finite {name}")
        arrays[name] = array
    return arrays


def _memory_delta_for_inference(model, system, device: torch.device) -> int:
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    _outputs_to_arrays(model.eval(system[0], system[2], system[1]))
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    return max(peak - baseline, 0)


def _choose_large_system(model, source_system, args):
    base_atoms = int(source_system[0].shape[1])
    if source_system[2] is None:
        return source_system, (1, 1, 1), {
            "mode": "input has no cell; replication disabled",
            "warning": "provide periodic cells to enable atom scaling",
            "source_atoms": base_atoms,
            "selected_atoms": base_atoms,
            "sizing_probes": [],
        }

    free_before, total_bytes = torch.cuda.mem_get_info(args.device)
    calibration_delta = _memory_delta_for_inference(model, source_system, args.device)
    torch.cuda.empty_cache()
    free_after, _ = torch.cuda.mem_get_info(args.device)
    bytes_per_atom = max(
        int(math.ceil(calibration_delta / max(base_atoms, 1))), _MIN_BYTES_PER_ATOM
    )
    memory_budget = int(free_after * args.memory_fraction)
    if calibration_delta > memory_budget:
        raise RuntimeError("source input already exceeds the profiling memory budget; provide a smaller seed or increase --memory-fraction")
    target_atoms = max(base_atoms, memory_budget // bytes_per_atom)
    if args.max_atoms > 0:
        target_atoms = min(target_atoms, args.max_atoms)

    print(
        "DPA4 atom sizing estimate: "
        + json.dumps(
            {
                "source_atoms": base_atoms,
                "target_atoms_estimate": int(target_atoms),
                "free_bytes": free_after,
                "safety_fraction": args.memory_fraction,
                "estimated_bytes_per_atom": bytes_per_atom,
                "memory_budget_bytes": memory_budget,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    probes: list[dict[str, Any]] = [
        {
            "atoms": base_atoms,
            "status": "calibration_success",
            "peak_increment_bytes": calibration_delta,
        }
    ]
    selected_system = source_system
    selected_factors = (1, 1, 1)
    selected_peak_delta = calibration_delta

    for _ in range(_MAX_SIZING_PROBES):
        factors = _balanced_replication_factors(max(1, target_atoms // base_atoms))
        candidate = _replicate_system(source_system, factors)
        candidate_atoms = int(candidate[0].shape[1])
        if candidate_atoms <= int(selected_system[0].shape[1]):
            break
        try:
            peak_delta = _memory_delta_for_inference(model, candidate, args.device)
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as error:
            failed_probe = {
                "atoms": candidate_atoms,
                "replicas": list(factors),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            probes.append(failed_probe)
            print("DPA4 sizing probe: " + json.dumps(failed_probe), flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            target_atoms = max(base_atoms, int(candidate_atoms * 0.70))
            if target_atoms <= int(selected_system[0].shape[1]):
                break
            continue

        if peak_delta > memory_budget:
            probes.append({"atoms": candidate_atoms, "replicas": list(factors),
                           "status": "over_budget", "peak_increment_bytes": peak_delta})
            print("DPA4 sizing probe: " + json.dumps(probes[-1]), flush=True)
            target_atoms = max(base_atoms, min(int(candidate_atoms * 0.70),
                               int(candidate_atoms * memory_budget / peak_delta)))
            if target_atoms <= int(selected_system[0].shape[1]):
                break
            continue

        successful_probe = {
            "atoms": candidate_atoms,
            "replicas": list(factors),
            "status": "success",
            "peak_increment_bytes": peak_delta,
        }
        probes.append(successful_probe)
        print("DPA4 sizing probe: " + json.dumps(successful_probe), flush=True)
        selected_system = candidate
        selected_factors = factors
        selected_peak_delta = peak_delta
        torch.cuda.empty_cache()
        if args.max_atoms > 0 and candidate_atoms >= args.max_atoms:
            break

        measured_per_atom = max(
            int(math.ceil(peak_delta / max(candidate_atoms, 1))), _MIN_BYTES_PER_ATOM
        )
        next_target = max(candidate_atoms, int(memory_budget / measured_per_atom))
        if args.max_atoms > 0:
            next_target = min(next_target, args.max_atoms)
        if next_target <= int(candidate_atoms * 1.15):
            break
        target_atoms = next_target

    return selected_system, selected_factors, {
        "mode": "memory-estimated periodic supercell sizing",
        "source_atoms": base_atoms,
        "selected_atoms": int(selected_system[0].shape[1]),
        "replication_factors": list(selected_factors),
        "memory_fraction": args.memory_fraction,
        "free_bytes_before_calibration": free_before,
        "free_bytes_after_calibration": free_after,
        "total_device_bytes": total_bytes,
        "memory_budget_bytes": memory_budget,
        "estimated_bytes_per_atom": bytes_per_atom,
        "selected_peak_increment_bytes": selected_peak_delta,
        "max_atoms_override": args.max_atoms or None,
        "sizing_probes": probes,
    }


def _prepare_direct_system(model, source_system, args):
    atoms = int(source_system[0].shape[1])
    if args.max_atoms > 0 and atoms > args.max_atoms:
        raise ValueError("input exceeds --max-atoms; cannot shrink the source structure")
    if args.auto_size:
        return _choose_large_system(model, source_system, args)
    return source_system, (1, 1, 1), {
        "mode": "fixed input", "source_atoms": atoms, "selected_atoms": atoms,
        "sizing_probes": [],
    }


def _warmup_model(model, system, device, calls):
    for _ in range(calls):
        _outputs_to_arrays(model.eval(system[0], system[2], system[1]))
    torch.cuda.synchronize(device)


_NATIVE_KERNELS = {
    "gie": ("gie_zonal_scatter_",),
    "rmsnorm": ("rmsnorm_forward_kernel", "rmsnorm_backward_kernel"),
    "so2_envelope_reduce": ("so2_reduce_",),
}


def _summarize_trace(path, native_ops=()):
    payload = json.loads(Path(path).read_text())
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    groups = {"device": {}, "cpu": {}, "communication": {}}
    for event in events:
        if event.get("ph") != "X" or float(event.get("dur", 0)) <= 0:
            continue
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", ""))
        device = category in ("kernel", "gpu_kernel", "hip_kernel", "cuda_kernel")
        kinds = ["device"] if device else (["cpu"] if "cpu_op" in category else [])
        if any(token in name.lower() for token in ("nccl", "rccl", "mpi_", "border_op", "communicat")):
            kinds.append("communication")
        for kind in kinds:
            row = groups[kind].setdefault(name, {"name": name, "calls": 0, "duration_us": 0.0})
            row["calls"] += 1
            row["duration_us"] += float(event["dur"])
    if not groups["device"]:
        raise RuntimeError(f"trace has no device kernel events: {path}")
    def top(kind):
        return sorted(groups[kind].values(), key=lambda row: row["duration_us"], reverse=True)[:40]
    native = {}
    for op in native_ops:
        matched = [name for name in groups["device"] if any(
            token in name for token in _NATIVE_KERNELS.get(op, ())
        )]
        native[op] = {"status": "observed" if matched else "unconfirmed", "kernel_names": matched}
    return {
        "trace_file": str(path),
        "device_kernel_calls": sum(row["calls"] for row in groups["device"].values()),
        "top_device_kernels": top("device"), "top_cpu_ops": top("cpu"),
        "communication_events": top("communication"), "native_execution": native,
        "duration_note": "Summed event durations can overlap; they are not inference latency. Communication names are heuristic; absence does not prove no communication.",
    }


def _collect_lammps_profiles(stem, ranks, skip, active):
    reports = []
    for rank in range(ranks):
        prefix = f"{stem.name}_rank{rank}_gpu" if ranks > 1 else f"{stem.name}*_gpu"
        traces = [p for p in stem.parent.glob(prefix + "*.json") if not p.name.endswith(".summary.json")]
        if len(traces) != 1:
            raise RuntimeError(f"rank {rank}: expected one .pt2 trace, found {len(traces)}; rebuild DeepMD with PTExpt DP_PROFILER support")
        trace = traces[0]
        window_path = Path(str(trace) + ".summary.json")
        if not window_path.is_file():
            raise RuntimeError(f"rank {rank}: missing profiler window summary; rebuild DeepMD")
        window = json.loads(window_path.read_text())
        if (not window.get("complete") or window.get("aborted")
                or window.get("captured_calls") != active
                or window.get("skip_calls") != skip or window.get("active_calls") != active):
            raise RuntimeError(f"rank {rank}: incomplete or mismatched profiler window: {window}")
        report = _summarize_trace(trace)
        report.update(rank=rank, window=window)
        reports.append(report)
    return reports


def _profile_events(profiler) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for event in profiler.key_averages():
        device_us = 0.0
        for field in ("self_device_time_total", "self_cuda_time_total"):
            value = getattr(event, field, None)
            if value is not None:
                device_us = float(value)
                if device_us > 0.0:
                    break
        rows.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "self_cpu_us": float(event.self_cpu_time_total),
                "total_cpu_us": float(event.cpu_time_total),
                "self_device_us": device_us,
                "self_cpu_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
                "self_device_bytes": int(
                    getattr(event, "self_device_memory_usage", 0)
                ),
            }
        )
    sort_key = (
        "self_device_us"
        if any(row["self_device_us"] > 0.0 for row in rows)
        else "self_cpu_us"
    )
    rows.sort(key=lambda row: row[sort_key], reverse=True)
    return sort_key, rows


def _parse_native_ops(raw: str | None):
    if raw is None:
        return None
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _loaded_deepmd_device_modules():
    return [
        sys.modules[name]
        for name in ("deepmd.pt_expt.utils.env", "deepmd.pt.utils.env")
        if name in sys.modules
    ]


def _construct_dpa4_model(args):
    rank = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    initialized_modules = _loaded_deepmd_device_modules()
    for module in initialized_modules:
        if torch.device(module.DEVICE) != torch.device(args.device):
            raise RuntimeError(
                f"DeepMD already initialized on {module.DEVICE}, requested {args.device}"
            )
    prior_device = os.environ.get("DEVICE")
    prior_rank = os.environ.get("LOCAL_RANK")
    if not initialized_modules:
        os.environ["DEVICE"] = "cuda"
        os.environ["LOCAL_RANK"] = str(rank)
    try:
        from model import DPA4Model

        return DPA4Model(
            args.model_file,
            native_ops=_parse_native_ops(args.native_ops),
            auto_batch_size=False,
        )
    finally:
        if not initialized_modules:
            if prior_device is None:
                os.environ.pop("DEVICE", None)
            else:
                os.environ["DEVICE"] = prior_device
            if prior_rank is None:
                os.environ.pop("LOCAL_RANK", None)
            else:
                os.environ["LOCAL_RANK"] = prior_rank


def _run_dpa4(args) -> int:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("direct DPA4 profile requires a HIP/DCU PyTorch runtime")
    if not Path(args.model_file).is_file():
        raise FileNotFoundError(f"model file does not exist: {args.model_file}")
    if args.input_npz:
        if not Path(args.input_npz).is_file():
            raise FileNotFoundError(f"input NPZ does not exist: {args.input_npz}")
        source_system = _load_npz_system(Path(args.input_npz))
        input_source = str(Path(args.input_npz).resolve())
    else:
        if not Path(args.geometry).is_file():
            raise FileNotFoundError(f"geometry file does not exist: {args.geometry}")
        source_system = _read_lammps_system(Path(args.geometry))
        input_source = str(Path(args.geometry).resolve())

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be cuda:N on this HIP/DCU host")
    if device.index is None:
        device = torch.device("cuda:0")
    if device.index >= torch.cuda.device_count():
        raise ValueError(
            f"requested {device}, but only {torch.cuda.device_count()} device(s) are visible"
        )
    device_name = torch.cuda.get_device_name(device)
    if args.expected_dcu_name and args.expected_dcu_name.casefold() not in device_name.casefold():
        raise RuntimeError(
            f"expected device containing {args.expected_dcu_name!r}, got {device_name!r}"
        )

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / f"dpa4_{time.time_ns()}"
    output_dir.mkdir()
    trace_path = output_dir / "dpa4_inference_trace.json"
    summary_path = output_dir / "dpa4_inference_profile_summary.json"
    torch.cuda.set_device(device)
    args.device = str(device)
    model = _construct_dpa4_model(args)
    try:
        selected_system, replicas, sizing = _prepare_direct_system(model, source_system, args)
        _warmup_model(model, selected_system, device, args.warmup)
        coords, atom_types, cells = selected_system
        memory_before = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            with torch.profiler.record_function("dpa4/inference.single_energy_force"):
                outputs = model.eval(coords, cells, atom_types)
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        arrays = _outputs_to_arrays(outputs)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        sort_metric, all_events = _profile_events(profiler)
        inference_range = next(
            (event for event in all_events if event["name"] == "dpa4/inference.single_energy_force"),
            None,
        )
        if inference_range is None or inference_range["calls"] != 1:
            raise RuntimeError("trace does not contain exactly one model inference range")
        device_events = [event for event in all_events if event["self_device_us"] > 0.0]
        if not device_events:
            raise RuntimeError(
                "PyTorch profiler recorded no HIP device events; check profiler support"
            )
        report = {
            "mode": "direct_dpa4_single_card",
            "model_file": str(Path(args.model_file).resolve()),
            "input_source": input_source,
            "backend": str(getattr(model, "backend", "unknown")),
            "execution_kind": "eager_checkpoint" if Path(args.model_file).suffix == ".pt" else "exported_artifact",
            "native_patch_applicable": Path(args.model_file).suffix == ".pt",
            "native_ops": list(getattr(model, "native_ops", ())),
            "native_ops_patched": dict(getattr(model, "native_ops_patched", {})),
            "device": str(device),
            "device_name": device_name,
            "ranks": 1,
            "gpu_ids": [device.index],
            "return_code": 0,
            "output_dir": str(output_dir),
            "command": args.cli_command,
            "hip_version": torch.version.hip,
            "torch_version": str(torch.__version__),
            "profiled_inference_calls": 1,
            "warmup_calls": args.warmup,
            "world_size": 1,
            "analysis_goal": "hotspots",
            "atom_count": int(coords.shape[1]),
            "source_atom_count": int(source_system[0].shape[1]),
            "replication_factors": list(replicas),
            "sizing_probe_calls": len(sizing.get("sizing_probes", [])),
            "sizing": sizing,
            "profile_wall_ms": float(elapsed_ms),
            "profile_wall_ms_note": "Includes profiler startup, capture and shutdown; not inference latency.",
            "memory_before_bytes": memory_before,
            "profile_peak_allocated_bytes": peak_memory,
            "profile_peak_increment_bytes": max(peak_memory - memory_before, 0),
            "profiler_sort_metric": sort_metric,
            "device_event_count": len(device_events),
            "top_profiler_events": all_events[:40],
            "outputs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
            "trace_file": str(trace_path),
        }
        profiler.export_chrome_trace(str(trace_path))
        report["trace_analysis"] = _summarize_trace(trace_path, report["native_ops"])
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"DPA4 trace: {trace_path}", flush=True)
        print(f"DPA4 summary: {summary_path}", flush=True)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    finally:
        model.close()


def _run_generate_data(args) -> int:
    source_path = args.input_npz if args.input_npz is not None else args.geometry
    if not source_path.is_file():
        raise FileNotFoundError(f"source geometry does not exist: {source_path}")
    if args.output_npz.exists() or args.output_npz.is_symlink():
        raise FileExistsError(f"refusing to overwrite generated input: {args.output_npz}")
    source_system = (
        _load_npz_system(source_path)
        if args.input_npz is not None
        else _read_lammps_system(source_path)
    )
    generated_system = generate_test_data(source_system, args.replicate)
    output_path = _write_generated_npz(args.output_npz, generated_system)
    summary = {
        "source": str(source_path.resolve()),
        "output_npz": str(output_path.resolve()),
        "source_atoms": int(source_system[0].shape[1]),
        "generated_atoms": int(generated_system[0].shape[1]),
        "replication_factors": list(args.replicate),
        "atom_types_preserved": True,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _metadata_has_comm_artifact(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "has_comm_artifact" and nested in (True, "true", "True", 1):
                return True
            if _metadata_has_comm_artifact(nested):
                return True
    elif isinstance(value, list):
        return any(_metadata_has_comm_artifact(nested) for nested in value)
    return False


def _check_with_comm_model(model_path: Path) -> None:
    if not zipfile.is_zipfile(model_path):
        raise ValueError("multi-rank LAMMPS requires a frozen with-comm .pt2 archive")
    with zipfile.ZipFile(model_path) as archive:
        names = archive.namelist()
        has_comm_path = any(
            name.endswith("model/extra/forward_lower_with_comm.pt2")
            or name.endswith("forward_lower_with_comm.pt2")
            for name in names
        )
        has_comm_metadata = False
        for name in names:
            if not name.endswith("metadata.json"):
                continue
            try:
                metadata = json.loads(archive.read(name).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if _metadata_has_comm_artifact(metadata):
                has_comm_metadata = True
                break
    if not (has_comm_path or has_comm_metadata):
        raise ValueError(
            "multi-rank LAMMPS requires a DPA4 .pt2 with "
            "model/extra/forward_lower_with_comm.pt2 (has_comm_artifact=true)"
        )


def _validate_lammps_model_for_ranks(model_path: Path, ranks: int) -> None:
    if ranks > 1:
        _check_with_comm_model(model_path)


def _validate_lammps_template(text: str, replicate: tuple[int, int, int]) -> None:
    # Ignore comments so an example in a comment cannot satisfy the contract.
    active = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    if not re.search(r"^\s*read_data\s+\$\{data\}(?:\s|$)", active, re.IGNORECASE | re.MULTILINE):
        raise ValueError("invalid LAMMPS input template: read_data must use ${data}")
    if not re.search(r"^\s*pair_style\s+deepmd\b[^\n]*\$\{model\}", active, re.IGNORECASE | re.MULTILINE):
        raise ValueError("invalid LAMMPS input template: pair_style deepmd must use ${model}")
    if not re.search(r"^\s*atom_modify\s+map\s+yes(?:\s|$)", active, re.IGNORECASE | re.MULTILINE):
        raise ValueError("invalid LAMMPS input template: DPA4 requires atom_modify map yes")
    if not re.search(r"^\s*pair_coeff\b", active, re.IGNORECASE | re.MULTILINE):
        raise ValueError("invalid LAMMPS input template: define pair_coeff and the model type map")
    if not re.search(r"^\s*run\s+\$\{steps\}(?:\s|$)", active, re.IGNORECASE | re.MULTILINE):
        raise ValueError("LAMMPS template must use `run ${steps}`")
    if replicate != (1, 1, 1):
        read_data = re.search(r"^\s*read_data\s+\$\{data\}.*$", active, re.IGNORECASE | re.MULTILINE)
        replica = re.search(
            r"^\s*replicate\s+\$\{rx\}\s+\$\{ry\}\s+\$\{rz\}(?:\s|$)",
            active,
            re.IGNORECASE | re.MULTILINE,
        )
        if read_data is None or replica is None or replica.start() < read_data.end():
            raise ValueError(
                "nontrivial --replicate requires `replicate ${rx} ${ry} ${rz}` "
                "after read_data in the input template"
            )


def _lammps_atom_count(data_path: Path) -> int | None:
    for line in data_path.read_text(errors="replace").splitlines():
        match = re.match(r"\s*(\d+)\s+atoms\b", line)
        if match:
            return int(match.group(1))
    return None


def _mpi_rank_wrapper_text() -> str:
    return '''#!/usr/bin/env python3
import os
import sys
rank = None
for key in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "PMI_LOCAL_RANK", "SLURM_LOCALID", "MV2_COMM_WORLD_LOCAL_RANK"):
    if key in os.environ:
        rank = os.environ[key]
        break
if rank is None:
    raise SystemExit("cannot determine local MPI rank for DeepMD GPU mapping")
os.environ["LOCAL_RANK"] = str(rank)
os.environ["RANK"] = str(rank)
os.environ["DEVICE"] = "cuda"
stem = os.environ["DPA4_DP_PROFILER_STEM"]
os.environ["DP_PROFILER"] = f"{stem}_rank{rank}"
log_dir = os.environ["DPA4_RANK_LOG_DIR"]
rank_log = open(os.path.join(log_dir, f"lammps_rank{rank}.log"), "w", buffering=1)
os.dup2(rank_log.fileno(), 1)
os.dup2(rank_log.fileno(), 2)
os.execvp(sys.argv[1], sys.argv[1:])
'''


def _build_lammps_command(
    args,
    *,
    ranks: int,
    lmp_exe: str,
    mpi_exe: str | None,
    mode_dir: Path,
    work_dir: Path,
) -> list[str]:
    command_lmp = [
        str(lmp_exe),
        "-log",
        str(mode_dir / "lammps.log"),
        "-var",
        "model",
        "model.pt2",
        "-var",
        "data",
        "system.data",
        "-var",
        "steps",
        str(args.steps),
        "-var",
        "rx",
        str(args.replicate[0]),
        "-var",
        "ry",
        str(args.replicate[1]),
        "-var",
        "rz",
        str(args.replicate[2]),
        "-in",
        "profile.in",
    ]
    if ranks == 1:
        return command_lmp
    if mpi_exe is None:
        raise ValueError("multi-rank LAMMPS command requires an MPI launcher")
    wrapper = work_dir / "rank_exec.py"
    wrapper.write_text(_mpi_rank_wrapper_text())
    return [
        str(mpi_exe),
        "-np",
        str(ranks),
        sys.executable,
        str(wrapper),
        *command_lmp,
    ]


def _lammps_rank_count(args) -> int:
    fixed_modes = {"lammps-single": 1, "lammps-mpi4": 4}
    if args.mode == "lammps":
        ranks = len(args.gpu_ids)
    elif args.mode in fixed_modes:
        ranks = fixed_modes[args.mode]
    else:
        raise ValueError(f"unsupported LAMMPS mode: {args.mode}")
    world_size = getattr(args, "world_size", None)
    if world_size is not None and (world_size < 1 or world_size != ranks):
        raise ValueError("--world-size must equal the number of --gpu-ids and match the selected mode")
    if len(args.gpu_ids) != ranks:
        raise ValueError(f"{args.mode} requires exactly {ranks} distinct --gpu-ids")
    return ranks


def _gpu_visibility(gpu_ids, env):
    masks = [env[key].strip() for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
             if key in env]
    if len(set(masks)) > 1:
        raise ValueError("conflicting HIP/CUDA visibility masks; align them before launching")
    if not masks:
        return ",".join(str(index) for index in gpu_ids)
    devices = [token.strip() for token in masks[0].split(",")]
    if any(index >= len(devices) for index in gpu_ids):
        raise ValueError("GPU selection exceeds the inherited visibility mask")
    return ",".join(devices[index] for index in gpu_ids)


def _run_lammps(args) -> int:
    if args.steps < 1:
        raise ValueError("--steps must be positive for steady-state MD profiling")
    if args.profile_skip < 1 or args.profile_calls < 1:
        raise ValueError("--profile-skip and --profile-calls must be positive")
    if args.steps < args.profile_skip + args.profile_calls:
        raise ValueError("--steps must be at least --profile-skip + --profile-calls")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    ranks = _lammps_rank_count(args)
    model_path = Path(args.model_file).resolve()
    data_path = Path(args.data_file).resolve()
    input_path = Path(args.lammps_input).resolve()
    for path, label in ((model_path, "model"), (data_path, "data file"), (input_path, "LAMMPS input")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    _validate_lammps_model_for_ranks(model_path, ranks)
    template = input_path.read_text()
    _validate_lammps_template(template, args.replicate)

    lmp_exe = shutil.which(args.lammps_exe) or (args.lammps_exe if Path(args.lammps_exe).is_file() else None)
    if lmp_exe is None:
        raise FileNotFoundError(f"LAMMPS executable not found: {args.lammps_exe}")
    lmp_exe = str(Path(lmp_exe).resolve())
    mpi_exe = None
    if ranks > 1:
        requested_mpi_exe = getattr(args, "mpi_exe", None)
        if not requested_mpi_exe:
            raise ValueError("--mpi-exe is required when using multiple GPU IDs")
        mpi_exe = shutil.which(requested_mpi_exe) or (
            requested_mpi_exe if Path(requested_mpi_exe).is_file() else None
        )
        if mpi_exe is None:
            raise FileNotFoundError(f"MPI launcher not found: {requested_mpi_exe}")
        mpi_exe = str(Path(mpi_exe).resolve())

    gpu_ids = args.gpu_ids
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("LAMMPS profile launcher requires a HIP/DCU PyTorch runtime")
    visible_device_count = torch.cuda.device_count()
    if visible_device_count < ranks:
        raise RuntimeError(
            f"requested {ranks} GPUs, but only {visible_device_count} are visible"
        )
    if any(index >= visible_device_count for index in gpu_ids):
        raise ValueError(
            f"GPU ids {gpu_ids} exceed the {visible_device_count} visible device indices"
        )

    output_root = Path(args.output_dir).resolve()
    mode_root = output_root / args.mode
    mode_root.mkdir(parents=True, exist_ok=True)
    mode_dir = mode_root / f"run_{time.time_ns()}"
    mode_dir.mkdir()
    work_dir = mode_dir / "work"
    work_dir.mkdir()
    for alias, source in (("model.pt2", model_path), ("system.data", data_path), ("input.template", input_path)):
        alias_path = work_dir / alias
        if alias_path.exists() or alias_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite staged input: {alias_path}")
        alias_path.symlink_to(source)
    generated_input = work_dir / "profile.in"
    generated_input.write_text("timer full\ninclude input.template\n")

    log_path = mode_dir / "lammps_stdout.log"
    profiler_stem = mode_dir / f"deepmd_profile_{time.time_ns()}"
    env = os.environ.copy()
    visible = _gpu_visibility(gpu_ids, env)
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["HIP_VISIBLE_DEVICES"] = visible
    env["DEVICE"] = "cuda"
    env["WORLD_SIZE"] = str(ranks)
    env["DP_PROFILER_SKIP"] = str(args.profile_skip)
    env["DP_PROFILER_ACTIVE"] = str(args.profile_calls)
    env["DPA4_DP_PROFILER_STEM"] = str(profiler_stem)
    env["DPA4_RANK_LOG_DIR"] = str(mode_dir)
    if ranks == 1:
        env["LOCAL_RANK"] = "0"
        env["RANK"] = "0"
        env["DP_PROFILER"] = str(profiler_stem)
    command = _build_lammps_command(
        args,
        ranks=ranks,
        lmp_exe=str(lmp_exe),
        mpi_exe=None if mpi_exe is None else str(mpi_exe),
        mode_dir=mode_dir,
        work_dir=work_dir,
    )

    started = time.perf_counter()
    with log_path.open("w") as output:
        result = subprocess.run(
            command,
            cwd=work_dir,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
    elapsed_s = time.perf_counter() - started
    profile_files = sorted(
        str(path) for path in mode_dir.glob(f"{profiler_stem.name}*.json")
        if not path.name.endswith(".summary.json")
    )
    rank_logs = sorted(str(path) for path in mode_dir.glob("lammps_rank*.log"))
    data_atoms = _lammps_atom_count(data_path)
    total_atoms = (
        None
        if data_atoms is None
        else data_atoms * math.prod(args.replicate)
    )
    report = {
        "mode": args.mode,
        "cli_command": args.cli_command,
        "model_file": str(model_path),
        "data_file": str(data_path),
        "lammps_input_template": str(input_path),
        "lammps_executable": str(lmp_exe),
        "mpi_executable": None if mpi_exe is None else str(mpi_exe),
        "ranks": ranks,
        "world_size": ranks,
        "analysis_goal": "hotspots",
        "operator_source": "exported artifact; no Python DPA4Model patches",
        "profile_window": {"skip_model_calls": args.profile_skip, "active_model_calls": args.profile_calls},
        "gpu_ids": list(gpu_ids),
        "child_visible_devices": visible,
        "replication_factors": list(args.replicate),
        "source_atom_count": data_atoms,
        "requested_atom_count": total_atoms,
        "steps": args.steps,
        "command": command,
        "elapsed_seconds": elapsed_s,
        "elapsed_seconds_note": "Whole subprocess including loading and startup; not steady-state latency.",
        "return_code": result.returncode,
        "stdout_log": str(log_path),
        "rank_logs": rank_logs,
        "deepmd_profiler_stem": str(profiler_stem),
        "deepmd_profiler_files": profile_files,
    }
    summary_path = mode_dir / "profile_summary.json"
    try:
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)
        report["rank_profiles"] = _collect_lammps_profiles(
            profiler_stem, ranks, args.profile_skip, args.profile_calls
        )
        report["profile_valid"] = True
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        report["profile_valid"] = False
        report["profile_error"] = str(error)
        raise
    finally:
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"LAMMPS profile summary: {summary_path}", flush=True)
        print(json.dumps(report, sort_keys=True), flush=True)
    return 0


def _add_lammps_arguments(
    parser: argparse.ArgumentParser, *, mpi: bool, mpi_optional: bool = False
) -> None:
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--lammps-input", type=Path, required=True)
    parser.add_argument("--lammps-exe", required=True, help="LAMMPS binary path or PATH name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicate", type=_parse_triplet, default=(1, 1, 1))
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="MD steps (default 20); must cover the skip and active model-call window",
    )
    parser.add_argument(
        "--gpu-ids",
        type=_parse_gpu_ids,
        required=True,
        help=(
            "comma-separated visible GPU indices; in `lammps`, the number of IDs "
            "selects the MPI rank count (one rank per GPU)"
        ),
    )
    parser.add_argument("--world-size", type=int, default=None,
                        help="MPI ranks; must match --gpu-ids count (default inferred)")
    parser.add_argument("--profile-skip", type=int, default=5,
                        help="model calls excluded before capture, including initial force")
    parser.add_argument("--profile-calls", type=int, default=10,
                        help="model calls captured after warmup")
    parser.add_argument("--timeout", type=float, default=None)
    if mpi or mpi_optional:
        parser.add_argument(
            "--mpi-exe",
            required=mpi,
            help="mpirun/mpiexec path or PATH name; required for multiple GPUs",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile direct DPA4 inference or LAMMPS DPA4 simulations."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    dpa4 = subparsers.add_parser(
        "dpa4", help="profile one direct single-card DPA4Model.eval call"
    )
    dpa4.add_argument("--model-file", type=Path, required=True)
    source = dpa4.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npz", type=Path)
    source.add_argument("--geometry", type=Path)
    dpa4.add_argument("--output-dir", type=Path, required=True)
    dpa4.add_argument("--device", default="cuda:0")
    dpa4.add_argument("--expected-dcu-name", default="K500SM_AI")
    dpa4.add_argument("--max-atoms", type=int, default=0)
    dpa4.add_argument("--memory-fraction", type=float, default=0.70)
    sizing_group = dpa4.add_mutually_exclusive_group()
    sizing_group.add_argument("--auto-size", dest="auto_size", action="store_true",
                              help="estimate a large periodic system from free memory (default)")
    sizing_group.add_argument("--fixed-input", dest="auto_size", action="store_false",
                              help="profile exactly the supplied input without replication")
    dpa4.set_defaults(auto_size=True)
    dpa4.add_argument("--warmup", type=int, default=5,
                      help="same-size inference calls before capture (default 5)")
    dpa4.add_argument(
        "--native-ops",
        default=None,
        help=(
            "comma-separated native operators (gie,rmsnorm by default; "
            "so2_envelope_reduce is opt-in for attention-free SO2Convolution); "
            "pass an empty string to run the DeepMD baseline"
        ),
    )

    lammps_single = subparsers.add_parser(
        "lammps-single", help="profile a one-rank LAMMPS DPA4 simulation"
    )
    _add_lammps_arguments(lammps_single, mpi=False)
    lammps_mpi4 = subparsers.add_parser(
        "lammps-mpi4", help="profile one distributed four-rank LAMMPS simulation"
    )
    _add_lammps_arguments(lammps_mpi4, mpi=True)
    lammps = subparsers.add_parser(
        "lammps",
        help="profile a LAMMPS DPA4 simulation using the requested GPU count",
        description=(
            "Use one MPI rank per --gpu-ids entry on this host. A single GPU "
            "runs LAMMPS directly; multiple GPUs launch one distributed "
            "single-host simulation."
        ),
    )
    _add_lammps_arguments(lammps, mpi=False, mpi_optional=True)

    generate = subparsers.add_parser(
        "generate-data",
        help="replicate a supplied periodic structure into a reusable NPZ input",
        description=(
            "Create a larger, deterministic profile input by periodically copying "
            "a supplied NPZ or orthorhombic LAMMPS geometry. Atom type indices "
            "are preserved; ensure they match the model used for profiling."
        ),
    )
    source = generate.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npz", type=Path)
    source.add_argument("--geometry", type=Path)
    generate.add_argument(
        "--replicate",
        type=_parse_triplet,
        required=True,
        help="periodic replication factors x,y,z (each must be positive)",
    )
    generate.add_argument("--output-npz", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args.cli_command = [sys.executable, str(Path(__file__).resolve()), *raw_argv]
    try:
        if args.mode == "generate-data":
            return _run_generate_data(args)
        if args.mode == "dpa4":
            if not args.model_file.is_file():
                raise FileNotFoundError(f"model file does not exist: {args.model_file}")
            if args.input_npz and not args.input_npz.is_file():
                raise FileNotFoundError(f"input NPZ does not exist: {args.input_npz}")
            if args.geometry and not args.geometry.is_file():
                raise FileNotFoundError(f"geometry file does not exist: {args.geometry}")
            if args.warmup < 1:
                raise ValueError("--warmup must be positive")
            if args.max_atoms < 0:
                raise ValueError("--max-atoms must be zero (auto) or a positive ceiling")
            if not 0.05 <= args.memory_fraction <= 0.85:
                raise ValueError("--memory-fraction must be in [0.05, 0.85]")
            return _run_dpa4(args)
        if args.mode in ("lammps", "lammps-single", "lammps-mpi4"):
            return _run_lammps(args)
        parser.error(f"unknown mode: {args.mode}")
    except Exception as error:
        print(f"profile failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 1


class DPA4ProfilerCLITests(unittest.TestCase):
    def test_direct_and_lammps_modes_parse_paths_and_options(self):
        parser = build_parser()
        direct = parser.parse_args(
            [
                "dpa4",
                "--model-file", "model.pt",
                "--input-npz", "system.npz",
                "--output-dir", "out/direct",
            ]
        )
        self.assertEqual(direct.mode, "dpa4")
        self.assertEqual(direct.model_file, Path("model.pt"))
        self.assertIsNone(_parse_native_ops(direct.native_ops))
        self.assertEqual(_parse_native_ops(""), ())
        self.assertEqual(_parse_native_ops("gie,rmsnorm"), ("gie", "rmsnorm"))
        direct_native = parser.parse_args(
            [
                "dpa4",
                "--model-file", "model.pt",
                "--input-npz", "system.npz",
                "--output-dir", "out/native",
                "--native-ops", "gie,rmsnorm",
            ]
        )
        self.assertEqual(
            _parse_native_ops(direct_native.native_ops), ("gie", "rmsnorm")
        )

        mpi = parser.parse_args(
            [
                "lammps-mpi4",
                "--model-file", "model.pt2",
                "--data-file", "system.data",
                "--lammps-input", "input.template",
                "--lammps-exe", "/opt/lammps/bin/lmp",
                "--mpi-exe", "/opt/mpi/bin/mpirun",
                "--gpu-ids", "0,1,2,3",
                "--replicate", "2,2,2",
                "--output-dir", "out/mpi",
            ]
        )
        self.assertEqual(mpi.mode, "lammps-mpi4")
        self.assertEqual(mpi.gpu_ids, (0, 1, 2, 3))
        self.assertEqual(mpi.replicate, (2, 2, 2))
        self.assertEqual(_lammps_rank_count(mpi), 4)

        dynamic = parser.parse_args(
            [
                "lammps",
                "--model-file", "model.pt2",
                "--data-file", "water.data",
                "--lammps-input", "water.template",
                "--lammps-exe", "lmp",
                "--mpi-exe", "mpirun",
                "--gpu-ids", "0,2,3",
                "--output-dir", "out/dynamic",
            ]
        )
        self.assertEqual(dynamic.mode, "lammps")
        self.assertEqual(dynamic.gpu_ids, (0, 2, 3))
        self.assertEqual(_lammps_rank_count(dynamic), 3)

        generate = parser.parse_args(
            [
                "generate-data",
                "--input-npz", "seed.npz",
                "--replicate", "2,3,4",
                "--output-npz", "expanded.npz",
            ]
        )
        self.assertEqual(generate.mode, "generate-data")
        self.assertEqual(generate.replicate, (2, 3, 4))
        self.assertEqual(generate.output_npz, Path("expanded.npz"))

    def test_four_rank_artifact_requires_comm_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.pt2"
            with zipfile.ZipFile(model, "w") as archive:
                archive.writestr("metadata.json", '{"model": {"has_comm_artifact": true}}')
            _check_with_comm_model(model)

            with zipfile.ZipFile(model, "w") as archive:
                archive.writestr("metadata.json", '{"has_comm_artifact": false}')
            with self.assertRaisesRegex(ValueError, "forward_lower_with_comm"):
                _check_with_comm_model(model)

    def test_dynamic_lammps_rank_count_and_command_generation(self):
        parser = build_parser()
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_gpu_ids("0,0")
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_gpu_ids("0,-1")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for ranks in (1, 2, 4):
                gpu_ids = ",".join(str(index) for index in range(ranks))
                argv = [
                    "lammps",
                    "--model-file", "model.pt2",
                    "--data-file", "water.data",
                    "--lammps-input", "water.template",
                    "--lammps-exe", "lmp",
                    "--gpu-ids", gpu_ids,
                    "--output-dir", "out",
                ]
                mpi_exe = None if ranks == 1 else "/bin/mpirun"
                if mpi_exe is not None:
                    argv.extend(["--mpi-exe", mpi_exe])
                args = parser.parse_args(argv)
                self.assertEqual(_lammps_rank_count(args), ranks)

                work_dir = root / f"work-{ranks}"
                work_dir.mkdir()
                command = _build_lammps_command(
                    args,
                    ranks=ranks,
                    lmp_exe="/bin/lmp",
                    mpi_exe=mpi_exe,
                    mode_dir=root / f"mode-{ranks}",
                    work_dir=work_dir,
                )
                if ranks == 1:
                    self.assertEqual(command[0], "/bin/lmp")
                else:
                    self.assertEqual(command[:3], ["/bin/mpirun", "-np", str(ranks)])

            args = parser.parse_args(
                [
                    "lammps",
                    "--model-file", "model.pt2",
                    "--data-file", "water.data",
                    "--lammps-input", "water.template",
                    "--lammps-exe", "lmp",
                    "--gpu-ids", "0,1",
                    "--output-dir", "out",
                ]
            )
            work_dir = root / "work-no-mpi"
            work_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "requires an MPI launcher"):
                _build_lammps_command(
                    args,
                    ranks=2,
                    lmp_exe="/bin/lmp",
                    mpi_exe=None,
                    mode_dir=root / "mode-no-mpi",
                    work_dir=work_dir,
                )

            fixed = parser.parse_args(
                [
                    "lammps-mpi4",
                    "--model-file", "model.pt2",
                    "--data-file", "water.data",
                    "--lammps-input", "water.template",
                    "--lammps-exe", "lmp",
                    "--mpi-exe", "mpirun",
                    "--gpu-ids", "0,1",
                    "--output-dir", "out",
                ]
            )
            with self.assertRaisesRegex(ValueError, "requires exactly 4"):
                _lammps_rank_count(fixed)

    def test_multi_rank_lammps_always_requires_with_comm_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.pt2"
            model.write_bytes(b"plain model")
            _validate_lammps_model_for_ranks(model, 1)
            with self.assertRaisesRegex(ValueError, "multi-rank LAMMPS"):
                _validate_lammps_model_for_ranks(model, 2)

            with zipfile.ZipFile(model, "w") as archive:
                archive.writestr("metadata.json", '{"has_comm_artifact": true}')
            _validate_lammps_model_for_ranks(model, 2)
            _validate_lammps_model_for_ranks(model, 4)

    def test_sample_water_template_matches_profile_contract(self):
        template_path = Path(__file__).with_name("dpa4_water_profile.template")
        template = template_path.read_text()
        self.assertIn("atom_style full", template)
        self.assertIn("pair_coeff * * O H", template)
        _validate_lammps_template(template, (2, 2, 2))

    def test_mpi_command_passes_model_data_and_replication_as_lammps_vars(self):
        args = build_parser().parse_args(
            [
                "lammps-mpi4",
                "--model-file", "m.pt2",
                "--data-file", "d.data",
                "--lammps-input", "in.template",
                "--lammps-exe", "lmp",
                "--mpi-exe", "mpirun",
                "--gpu-ids", "0,1,2,3",
                "--replicate", "2,3,4",
                "--steps", "9",
                "--output-dir", "out",
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mode_dir = root / "mode"
            work_dir = root / "work"
            work_dir.mkdir()
            command = _build_lammps_command(
                args,
                ranks=4,
                lmp_exe="/bin/lmp",
                mpi_exe="/bin/mpirun",
                mode_dir=mode_dir,
                work_dir=work_dir,
            )
            self.assertEqual(command[:3], ["/bin/mpirun", "-np", "4"])
            for variable, value in (
                ("model", "model.pt2"),
                ("data", "system.data"),
                ("steps", "9"),
                ("rx", "2"),
                ("ry", "3"),
                ("rz", "4"),
            ):
                index = command.index(variable)
                self.assertEqual(command[index + 1], value)
            self.assertEqual(command[-2:], ["-in", "profile.in"])
            wrapper = Path(command[4])
            self.assertTrue(wrapper.is_file())
            wrapper_source = wrapper.read_text()
            self.assertIn('os.environ["LOCAL_RANK"]', wrapper_source)
            self.assertIn('os.environ["DP_PROFILER"]', wrapper_source)
            self.assertIn("lammps_rank{rank}.log", wrapper_source)


class DPA4ProfilerGeometryTests(unittest.TestCase):
    def test_balanced_replication_and_scaled_cell(self):
        self.assertEqual(_balanced_replication_factors(8), (2, 2, 2))
        source = (
            np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
            np.asarray([[0, 1]], dtype=np.int32),
            np.diag([3.0, 4.0, 5.0]).reshape(1, 9),
        )
        coords, atom_types, cells = _replicate_system(source, (2, 2, 2))
        self.assertEqual(coords.shape, (1, 16, 3))
        self.assertEqual(atom_types.shape, (1, 16))
        np.testing.assert_allclose(cells.reshape(3, 3), np.diag([6.0, 8.0, 10.0]))
        self.assertEqual(np.unique(coords[0], axis=0).shape[0], 16)

    def test_generate_test_data_preserves_types_and_expands_cell(self):
        source = (
            np.asarray([[[0.2, 0.3, 0.4], [1.2, 1.3, 1.4]]]),
            np.asarray([[0, 2]], dtype=np.int32),
            np.diag([3.0, 4.0, 5.0]).reshape(1, 9),
        )
        original_coords = source[0].copy()
        generated = generate_test_data(source, (2, 1, 2))
        coords, atom_types, cells = generated
        self.assertEqual(coords.shape, (1, 8, 3))
        np.testing.assert_array_equal(atom_types, [[0, 2, 0, 2, 0, 2, 0, 2]])
        np.testing.assert_allclose(cells.reshape(3, 3), np.diag([6.0, 4.0, 10.0]))
        np.testing.assert_array_equal(coords[0, :2], source[0][0])
        np.testing.assert_array_equal(source[0], original_coords)

    def test_generate_test_data_requires_periodic_cell(self):
        source = (
            np.asarray([[[0.0, 0.0, 0.0]]]),
            np.asarray([[0]], dtype=np.int32),
            None,
        )
        with self.assertRaisesRegex(ValueError, "requires a source cell"):
            generate_test_data(source, (2, 2, 2))

    def test_generated_npz_round_trips_through_direct_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "seed.npz"
            output_path = root / "generated" / "profile.npz"
            np.savez_compressed(
                source_path,
                coords=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                atom_types=np.asarray([0, 1], dtype=np.int32),
                cells=np.diag([3.0, 4.0, 5.0]),
            )
            result = main(
                [
                    "generate-data",
                    "--input-npz", str(source_path),
                    "--replicate", "2,1,1",
                    "--output-npz", str(output_path),
                ]
            )
            self.assertEqual(result, 0)
            generated = _load_npz_system(output_path)
            self.assertEqual(generated[0].shape, (1, 4, 3))
            np.testing.assert_array_equal(generated[1], [[0, 1, 0, 1]])
            np.testing.assert_allclose(
                generated[2].reshape(3, 3), np.diag([6.0, 4.0, 5.0])
            )

    def test_generate_data_accepts_lammps_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "seed.data"
            output_path = root / "expanded.npz"
            geometry.write_text(
                """2 atoms
2 atom types

0 3 xlo xhi
0 4 ylo yhi
0 5 zlo zhi

Atoms # atomic

1 1 0.0 0.0 0.0
2 2 1.0 0.0 0.0
"""
            )
            result = main(
                [
                    "generate-data",
                    "--geometry", str(geometry),
                    "--replicate", "2,1,1",
                    "--output-npz", str(output_path),
                ]
            )
            self.assertEqual(result, 0)
            generated = _load_npz_system(output_path)
            self.assertEqual(generated[0].shape, (1, 4, 3))
            np.testing.assert_array_equal(generated[1], [[0, 1, 0, 1]])

    def test_generated_npz_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "existing.npz"
            output_path.write_bytes(b"keep this file")
            system = (
                np.asarray([[[0.0, 0.0, 0.0]]]),
                np.asarray([[0]], dtype=np.int32),
                np.eye(3).reshape(1, 9),
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _write_generated_npz(output_path, system)
            self.assertEqual(output_path.read_bytes(), b"keep this file")

    def test_replication_factor_product_never_exceeds_limit(self):
        for limit in range(1, 80):
            factors = _balanced_replication_factors(limit)
            self.assertLessEqual(math.prod(factors), limit)
            self.assertGreaterEqual(factors[0], 1)
            self.assertLessEqual(factors[0], factors[1])
            self.assertLessEqual(factors[1], factors[2])

    def test_lammps_template_contract(self):
        valid = """
        atom_modify map yes
        read_data ${data}
        replicate ${rx} ${ry} ${rz}
        pair_style deepmd ${model}
        pair_coeff * * O H
        run ${steps}
        """
        _validate_lammps_template(valid, (2, 2, 2))
        with self.assertRaisesRegex(ValueError, "atom_modify map yes"):
            _validate_lammps_template(valid.replace("atom_modify map yes", ""), (1, 1, 1))


if __name__ == "__main__":
    raise SystemExit(main())
