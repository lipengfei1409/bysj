"""Opt-in full-model DPA4-Neo latency benchmark for a DTK/DCU machine."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics

import torch

from dcu_dpa4.native import native_available
from model.dpa4.modeling.dpa4 import DPA4Model
from .case import make_latency_case, resolve_checkpoint


def _percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.90),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _outputs_finite(output, *, compute_forces):
    tensors = [output.energy, output.atom_energy]
    if compute_forces:
        tensors.extend((output.force, output.virial))
    for tensor in tensors:
        if tensor is None or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("model produced a missing or non-finite output during latency setup")


def benchmark(args):
    if os.environ.get("DPA4_RUN_MODEL_LATENCY_TESTS") != "1":
        raise RuntimeError("set DPA4_RUN_MODEL_LATENCY_TESTS=1 to opt in to model latency timing")
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("model latency benchmark requires a DTK PyTorch build and a DCU")
    if not native_available():
        raise RuntimeError("HIP extension is unavailable; build with DPA4_BUILD_HIP=1 first")
    if min(args.atoms, args.warmup, args.batches, args.iterations) <= 0:
        raise ValueError("atoms, warmup, batches, and iterations must all be positive")

    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else resolve_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device("cuda", torch.cuda.current_device())
    model = DPA4Model.from_checkpoint(
        checkpoint, primitive_backend="hip", linear_backend="hip",
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    batch, graph = make_latency_case(model.config, args.atoms, device=device)
    stream = torch.cuda.current_stream(device)

    def invoke(compute_forces):
        if compute_forces:
            return model(batch, graph, compute_forces=True)
        with torch.no_grad():
            return model(batch, graph, compute_forces=False)

    results = {}
    for name, compute_forces in (("energy_only", False), ("energy_force_virial", True)):
        for _ in range(args.warmup):
            output = invoke(compute_forces)
        torch.cuda.synchronize(device)
        _outputs_finite(output, compute_forces=compute_forces)

        samples = []
        for _ in range(args.batches):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            for _ in range(args.iterations):
                output = invoke(compute_forces)
            stop.record(stream)
            stop.synchronize()
            samples.append(start.elapsed_time(stop) / args.iterations)
        _outputs_finite(output, compute_forces=compute_forces)
        results[name] = _summarize(samples)

    result = {
        "model": "DPA4-Neo-OMol25-100M",
        "checkpoint": str(checkpoint),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "primitive_backend": model.backend.primitive_backend,
        "linear_backend": "hip",
        "dtype": str(batch.positions.dtype),
        "input": {
            "atoms": int(batch.positions.shape[0]),
            "edges": int(graph.edge_index.shape[1]),
            "frames": int(batch.n_frames),
            "edge_capacity": int(graph.mask.numel()),
        },
        "timing_scope": "complete DPA4Model call; checkpoint loading, graph creation, and H2D excluded",
        "warmup": args.warmup,
        "batches": args.batches,
        "iterations_per_batch": args.iterations,
        "latency": results,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Neo checkpoint; defaults to the documented adjacent path")
    parser.add_argument("--atoms", type=int, default=32, help="synthetic non-periodic model input atom count")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args(argv)
    print(json.dumps(benchmark(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
