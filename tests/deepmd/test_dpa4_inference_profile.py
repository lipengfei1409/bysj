"""DPA4 profiling CLI: direct inference, one-card LAMMPS, or four-card MPI LAMMPS.

Every model, input, executable, and output path is passed as a CLI argument.
The three modes are distinct:

1. ``dpa4`` profiles exactly one direct ``DPA4Model.eval`` on one HCU. It
   estimates a large periodic system from available memory, then records one
   successful call; sizing probes occur outside the trace.
2. ``lammps-single`` runs one LAMMPS rank on one HCU.
3. ``lammps-mpi4`` runs one distributed LAMMPS simulation across exactly four
   MPI ranks/cards. It requires a DPA4 frozen model with the with-comm artifact;
   it is not four independent Python inference processes.

Examples::

    python3 tests/deepmd/test_dpa4_inference_profile.py dpa4 \
        --model-file /models/dpa4.pt --input-npz /systems/water.npz \
        --output-dir /tmp/dpa4-direct

    python3 tests/deepmd/test_dpa4_inference_profile.py lammps-single \
        --model-file /models/dpa4.pt2 --data-file /systems/water.data \
        --lammps-input /inputs/profile.template --lammps-exe /opt/dtk/bin/lmp \
        --gpu-ids 0 --output-dir /tmp/dpa4-lmp-one

    python3 tests/deepmd/test_dpa4_inference_profile.py lammps-mpi4 \
        --model-file /models/dpa4-with-comm.pt2 --data-file /systems/water.data \
        --lammps-input /inputs/profile.template --lammps-exe /opt/dtk/bin/lmp \
        --mpi-exe /opt/mpi/bin/mpirun --gpu-ids 0,1,2,3 \
        --replicate 2,2,2 --steps 0 --output-dir /tmp/dpa4-lmp-four

LAMMPS templates must use ``${model}`` in ``pair_style deepmd``, ``${data}``
in ``read_data``, include ``atom_modify map yes``, and use ``run ${steps}``.
For replication, include ``replicate ${rx} ${ry} ${rz}`` after ``read_data``.
The runner stages short symlink aliases in each run directory so LAMMPS variable
values do not contain spaces. ``--steps 0`` (the default) profiles the initial
force evaluation without advancing dynamics; use a positive value for a longer
simulation. LAMMPS profiling uses DeepMD's ``DP_PROFILER`` Kineto trace
(rank-specific in MPI mode) and ``timer full`` output.

For LAMMPS DPA4, the ``lammps-mpi4`` model must be a frozen artifact containing
``model/extra/forward_lower_with_comm.pt2`` (or metadata marking a comm artifact).
A plain ``DPA4Model.eval`` call does not distribute one inference across ranks.
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "python"))
_MIN_BYTES_PER_ATOM = 1024 * 1024
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
    bounds: dict[str, tuple[float, float]] = {}
    atoms: list[tuple[int, int, float, float, float]] = []
    in_atoms = False
    for line in path.read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) >= 6 and fields[-3:] == ["xy", "xz", "yz"]:
            if any(abs(float(value)) > 1.0e-12 for value in fields[:3]):
                raise ValueError(
                    "triclinic LAMMPS boxes are unsupported by this parser; "
                    "provide the geometry through a cell-aware NPZ instead"
                )
            continue
        for axis in ("x", "y", "z"):
            if len(fields) >= 4 and fields[-2:] == [f"{axis}lo", f"{axis}hi"]:
                bounds[axis] = (float(fields[0]), float(fields[1]))
                break
        if fields[0] == "Atoms":
            in_atoms = True
            continue
        if not in_atoms:
            continue
        if not fields[0].lstrip("+-").isdigit():
            if atoms:
                break
            continue
        if len(fields) < 5:
            continue
        atom_id = int(fields[0])
        atom_type = int(fields[1]) if len(fields) == 5 else int(fields[2])
        x, y, z = (float(value) for value in fields[-3:])
        atoms.append((atom_id, atom_type, x, y, z))

    if len(bounds) != 3 or not atoms:
        raise ValueError(f"could not parse orthorhombic LAMMPS geometry from {path}")
    atoms.sort(key=lambda row: row[0])
    coords = np.asarray([[row[2], row[3], row[4]] for row in atoms], dtype=np.float64)
    atom_types = np.asarray([row[1] - 1 for row in atoms], dtype=np.int32)
    lengths = [bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z")]
    cells = np.diag(np.asarray(lengths, dtype=np.float64)).reshape(1, 9)
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
            free_now, _ = torch.cuda.mem_get_info(args.device)
        except Exception as error:
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
        next_target = max(candidate_atoms, int((free_now * args.memory_fraction) / measured_per_atom))
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


def _parse_tilelang_ops(raw: str | None):
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
            tilelang_ops=_parse_tilelang_ops(args.tilelang_ops),
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
    model = _construct_dpa4_model(args)
    try:
        selected_system, replicas, sizing = _choose_large_system(model, source_system, args)
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
            "tilelang_ops": list(getattr(model, "tilelang_ops", ())),
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
            "atom_count": int(coords.shape[1]),
            "source_atom_count": int(source_system[0].shape[1]),
            "replication_factors": list(replicas),
            "sizing_probe_calls": len(sizing.get("sizing_probes", [])),
            "sizing": sizing,
            "profile_wall_ms": float(elapsed_ms),
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
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"DPA4 trace: {trace_path}", flush=True)
        print(f"DPA4 summary: {summary_path}", flush=True)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    finally:
        model.close()


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
        raise ValueError("four-rank LAMMPS requires a frozen with-comm .pt2 archive")
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
            "four-rank LAMMPS requires a DPA4 .pt2 with "
            "model/extra/forward_lower_with_comm.pt2 (has_comm_artifact=true)"
        )


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
        raise ValueError("four-rank LAMMPS command requires an MPI launcher")
    wrapper = work_dir / "rank_exec.py"
    wrapper.write_text(_mpi_rank_wrapper_text())
    return [
        str(mpi_exe),
        "-np",
        "4",
        sys.executable,
        str(wrapper),
        *command_lmp,
    ]


def _run_lammps(args) -> int:
    if args.steps < 0:
        raise ValueError("--steps must be zero or a positive integer")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    ranks = 1 if args.mode == "lammps-single" else 4
    model_path = Path(args.model_file).resolve()
    data_path = Path(args.data_file).resolve()
    input_path = Path(args.lammps_input).resolve()
    for path, label in ((model_path, "model"), (data_path, "data file"), (input_path, "LAMMPS input")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if ranks == 4:
        _check_with_comm_model(model_path)
    template = input_path.read_text()
    _validate_lammps_template(template, args.replicate)

    lmp_exe = shutil.which(args.lammps_exe) or (args.lammps_exe if Path(args.lammps_exe).is_file() else None)
    if lmp_exe is None:
        raise FileNotFoundError(f"LAMMPS executable not found: {args.lammps_exe}")
    mpi_exe = None
    if ranks == 4:
        mpi_exe = shutil.which(args.mpi_exe) or (args.mpi_exe if Path(args.mpi_exe).is_file() else None)
        if mpi_exe is None:
            raise FileNotFoundError(f"MPI launcher not found: {args.mpi_exe}")

    gpu_ids = args.gpu_ids
    if len(gpu_ids) != ranks:
        raise ValueError(f"{args.mode} requires exactly {ranks} distinct --gpu-ids")
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
    visible = ",".join(str(index) for index in gpu_ids)
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["HIP_VISIBLE_DEVICES"] = visible
    env["DEVICE"] = "cuda"
    env["DPA4_DP_PROFILER_STEM"] = str(profiler_stem)
    env["DPA4_RANK_LOG_DIR"] = str(mode_dir)
    if ranks == 1:
        env["LOCAL_RANK"] = "0"
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
        "gpu_ids": list(gpu_ids),
        "replication_factors": list(args.replicate),
        "source_atom_count": data_atoms,
        "requested_atom_count": total_atoms,
        "steps": args.steps,
        "command": command,
        "elapsed_seconds": elapsed_s,
        "return_code": result.returncode,
        "stdout_log": str(log_path),
        "rank_logs": rank_logs,
        "deepmd_profiler_stem": str(profiler_stem),
        "deepmd_profiler_files": profile_files,
    }
    summary_path = mode_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"LAMMPS profile summary: {summary_path}", flush=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    if not profile_files:
        raise RuntimeError(
            "LAMMPS completed but DeepMD DP_PROFILER produced no JSON trace; "
            "check that the DeepMD PyTorch profiler is enabled in this build"
        )
    return 0


def _add_lammps_arguments(parser: argparse.ArgumentParser, *, mpi: bool) -> None:
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--lammps-input", type=Path, required=True)
    parser.add_argument("--lammps-exe", required=True, help="LAMMPS binary path or PATH name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicate", type=_parse_triplet, default=(1, 1, 1))
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="LAMMPS steps; zero (default) profiles the initial force evaluation only",
    )
    parser.add_argument(
        "--gpu-ids",
        type=_parse_gpu_ids,
        required=True,
        help="comma-separated visible GPU indices (one for single, four for MPI)",
    )
    parser.add_argument("--timeout", type=float, default=None)
    if mpi:
        parser.add_argument("--mpi-exe", required=True, help="mpirun/mpiexec path or PATH name")


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
    dpa4.add_argument("--memory-fraction", type=float, default=0.50)
    dpa4.add_argument("--tilelang-ops", default=None)

    lammps_single = subparsers.add_parser(
        "lammps-single", help="profile a one-rank LAMMPS DPA4 simulation"
    )
    _add_lammps_arguments(lammps_single, mpi=False)
    lammps_mpi4 = subparsers.add_parser(
        "lammps-mpi4", help="profile one distributed four-rank LAMMPS simulation"
    )
    _add_lammps_arguments(lammps_mpi4, mpi=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args.cli_command = [sys.executable, str(Path(__file__).resolve()), *raw_argv]
    try:
        if args.mode == "dpa4":
            if not args.model_file.is_file():
                raise FileNotFoundError(f"model file does not exist: {args.model_file}")
            if args.input_npz and not args.input_npz.is_file():
                raise FileNotFoundError(f"input NPZ does not exist: {args.input_npz}")
            if args.geometry and not args.geometry.is_file():
                raise FileNotFoundError(f"geometry file does not exist: {args.geometry}")
            if args.max_atoms < 0:
                raise ValueError("--max-atoms must be zero (auto) or a positive ceiling")
            if not 0.05 <= args.memory_fraction <= 0.85:
                raise ValueError("--memory-fraction must be in [0.05, 0.85]")
            return _run_dpa4(args)
        if args.mode in ("lammps-single", "lammps-mpi4"):
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
