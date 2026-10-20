"""DPA4 shape tools and opt-in inference through installed DeePMD."""
import argparse
from dataclasses import asdict
import json

import torch

from dcu_dpa4.catalog import OPERATORS
from .checkpoint import load_checkpoint_spec
from .modeling import AtomicBatch, DPA4Config, DPA4Model, ShapeBackend


def tensor_info(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def shape_report(config, atoms, edges, frames):
    if atoms < 0 or edges < 0 or frames < 1:
        raise ValueError("atoms/edges must be nonnegative and frames positive")
    batch = AtomicBatch(torch.empty(atoms, 3, device="meta"),
        torch.empty(atoms, dtype=torch.int64, device="meta"),
        torch.empty(atoms, dtype=torch.int64, device="meta"), frames,
        torch.empty(frames, 3, 3, device="meta"))
    backend = ShapeBackend()
    out = DPA4Model(config, backend)(batch, edge_capacity=edges)
    return {
        "numerical_prediction": False,
        "scope": "major-stage shape skeleton; optional composite internals are not expanded",
        "config": asdict(config),
        "outputs": {name: tensor_info(value) for name, value in vars(out).items() if value is not None},
        "trace": [{"operator": t.operator, "stage": t.stage,
            "inputs": {k: tensor_info(v) for k, v in t.inputs.items()},
            "outputs": {k: tensor_info(v) for k, v in t.outputs.items()}} for t in backend.trace],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shape = commands.add_parser("shape", help="trace main DPA4 tensor shapes, not predictions")
    shape.add_argument("--checkpoint", help="trusted .pt checkpoint; default uses toy config")
    shape.add_argument("--head")
    shape.add_argument("--atoms", type=int, default=192, help="total flattened atoms over all frames")
    shape.add_argument("--edges", type=int, default=1000, help="directed edge capacity, including padding")
    shape.add_argument("--frames", type=int, default=1)
    commands.add_parser("operators", help="operator contracts as JSON")
    infer=commands.add_parser("infer",help="evaluate using DeePMD with optional primitive adapters")
    infer.add_argument("--checkpoint",required=True)
    infer.add_argument("--head")
    infer.add_argument("--system",required=True,help="deepmd/npy system directory")
    infer.add_argument("--device",choices=("cpu","dcu"),default="cpu")
    infer.add_argument("--operator-backend",choices=("deepmd","reference","hip"),default="deepmd")
    infer.add_argument("--neighbor-backend",choices=("native","cpu-bridge"),default="native")
    infer.add_argument("--max-frames",type=int,default=1)
    args = parser.parse_args()
    if args.command == "operators":
        report = [asdict(o) for o in OPERATORS.values()]
    elif args.command=="infer":
        from .runner import DPA4Runner,compute_metrics
        from .system import load_deepmd_system
        system=load_deepmd_system(args.system,max_frames=args.max_frames)
        with DPA4Runner(args.checkpoint,head=args.head,device=args.device,
            operator_backend=args.operator_backend,neighbor_backend=args.neighbor_backend) as runner:
            prediction=runner.eval(system)
            report={"numerical_prediction":True,"model_owner":"deepmd-kit","operator_backend":args.operator_backend,
                "neighbor_backend":args.neighbor_backend,"outputs":{k:{"shape":list(v.shape)} for k,v in vars(prediction).items()},
                "operator_report":runner.operator_report}
            if system.energy is not None and system.force is not None:
                report["label_metrics"]=compute_metrics(prediction.energy,prediction.force,system.energy,system.force)
    else:
        config = DPA4Config() if args.checkpoint is None else DPA4Config.from_model_params(load_checkpoint_spec(args.checkpoint, args.head).model_params)
        report = shape_report(config, args.atoms, args.edges, args.frames)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
