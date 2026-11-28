#!/usr/bin/env python3
"""Attribute every bmm/mm in one DPA4 forward to the source line that issued it.

The device profile says *what* the GEMM shapes are; this says *where* they come from.
It runs one forward under `torch.profiler` with `with_stack=True` and `record_shapes=True`,
then groups the matmul ops by (shape, innermost `deepmd/pt/` frame):

    python3 tools/bmm_attribution.py \
        --model /root/private_data/lipengfei/model/DPA4-OMol/DPA4-Air-OMol25-100M-v20260820.pt \
        --input-npz /root/private_data/lipengfei/test_result/water_spce_3000.npz \
        --device cuda:0

CPU works too (`--device cpu`), just slower; shapes and source lines are identical.
The trace this project already ships (`dpa4_inference_trace.json`) was recorded without
stacks, which is why the shapes had to be attributed by arithmetic until now.
"""

from __future__ import annotations

import argparse
import collections
import sys

import numpy as np


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="a .pt model file")
    parser.add_argument("--input-npz", required=True,
                        help="npz with coords / atom_types / cells")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=4,
                        help="how many stack frames to keep per op")
    parser.add_argument("--top", type=int, default=25)
    return parser.parse_args(argv)


def _load_pot(model: str, device: str):
    from deepmd.infer import DeepPot

    return DeepPot(model, device=device)


def _inputs(path: str):
    data = np.load(path)
    coordinates = data["coords"]
    atom_types = data["atom_types"].astype(int)
    cells = data.get("cells")
    # the model's type map is element names; the npz stores indices into the *input*
    # type map, which for this dataset is (O, H)
    names = np.array(["O", "H"], dtype=object)[atom_types]
    return coordinates, cells, names


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
    except ImportError as failure:  # pragma: no cover
        print(f"torch is required: {failure}", file=sys.stderr)
        return 2

    potential = _load_pot(args.model, args.device)
    coordinates, cells, names = _inputs(args.input_npz)
    potential.eval(coordinates, cells, names)  # warming / graph build

    activities = [ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True,
                 with_stack=True) as recorded:
        potential.eval(coordinates, cells, names)

    wanted = ("aten::bmm", "aten::mm", "aten::matmul", "aten::addmm")
    by_shape: collections.Counter = collections.Counter()
    by_frame: collections.Counter = collections.Counter()
    shape_source: dict[tuple, collections.Counter] = collections.defaultdict(
        collections.Counter)

    for event in recorded.key_averages(group_by_stack_n=args.frames * 2):
        if event.key not in wanted:
            continue
        shapes = tuple(tuple(shape) for shape in (event.input_shapes or []))
        by_shape[(event.key, shapes)] += event.count
        frame = "?"
        for candidate in getattr(event, "stack", None) or []:
            text = str(candidate)
            if "deepmd/pt/" in text:
                frame = " ".join(text.split())
                break
        by_frame[(event.key, frame)] += event.count
        shape_source[shapes][frame] += event.count

    print(f"\n{'count':>6s}  {'op':12s} shape")
    for (key, shapes), count in by_shape.most_common(args.top):
        print(f"{count:6d}  {key:12s} {shapes}")

    print(f"\n{'count':>6s}  {'op':12s} source frame")
    for (key, frame), count in by_frame.most_common(args.top):
        print(f"{count:6d}  {key:12s} {frame}")

    print("\nshape -> source frame")
    for shapes, frames in sorted(shape_source.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {shapes}")
        for frame, count in frames.most_common(3):
            print(f"    {count:4d}x {frame}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
