"""Run DeePMD-backed DPA4 energy, force, and virial inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.inference import create_runner


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="trusted DeePMD DPA4 .pt checkpoint")
    parser.add_argument("--system", required=True, help="DeepMD-NPY system directory")
    parser.add_argument("--head", help="required only for multi-head checkpoints")
    parser.add_argument("--device", choices=("cpu", "dcu"), default="cpu")
    parser.add_argument("--operator-backend", choices=("deepmd", "reference", "hip"), default="deepmd")
    parser.add_argument("--neighbor-backend", choices=("native", "cpu-bridge"), default="native")
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--output", type=Path, help="optional .npz output path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from model.dpa4.system import load_deepmd_system

    system = load_deepmd_system(args.system, max_frames=args.max_frames)
    with create_runner(
        "dpa4",
        "energy-force-virial",
        checkpoint=args.checkpoint,
        head=args.head,
        device=args.device,
        operator_backend=args.operator_backend,
        neighbor_backend=args.neighbor_backend,
    ) as runner:
        prediction = runner.predict(system)

    summary = {
        "family": "dpa4",
        "owner": "DeepMD-kit",
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "frames": int(system.coords.shape[0]),
        "atoms_per_frame": int(system.coords.shape[1]),
        "energy_shape": list(prediction.energy.shape),
        "force_shape": list(prediction.force.shape),
        "virial_shape": list(prediction.virial.shape),
        "energy": prediction.energy.tolist(),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.savez_compressed(
            args.output,
            energy=prediction.energy,
            force=prediction.force,
            virial=prediction.virial,
        )
        summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
