#!/usr/bin/env python3
"""Build the whole DPA4 model on CPU and print it.

"Print the model" means the full `torch.nn.Module` tree: descriptor, fitting net, and every
submodule with its parameters.  The model is reconstructed exactly the way inference does
it (`deepmd/pt/infer/deep_eval.py`):

    params = checkpoint["model"]["_extra_state"]["model_params"]
    wrapper = ModelWrapper(get_model(params))
    wrapper.load_state_dict(checkpoint["model"], strict=False)

    python3 tools/print_dpa4_model.py --model DPA4-Air-OMol25-100M-v20260820.pt
    python3 tools/print_dpa4_model.py --model ... --out model.txt       # to a file
    python3 tools/print_dpa4_model.py --model ... --descriptor          # one subtree
    python3 tools/print_dpa4_model.py --model ... --params              # name + shape rows
    python3 tools/print_dpa4_model.py --model ... --summary             # per-submodule table

Everything runs on the CPU: no CUDA/DCU device is touched, and `torch.jit.script` is
skipped (the inference path disables it for SEZM/DPA4 anyway).
"""

from __future__ import annotations

import argparse
import sys


def _load(path: str):
    """Return (model_params, state_dict) from a `.pt` checkpoint."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    if "_extra_state" not in state:
        raise SystemExit(f"{path}: no '_extra_state' in the checkpoint")
    params = state["_extra_state"]["model_params"]
    return params, state


def _build(params: dict):
    import torch

    from deepmd.pt.model.model import get_model
    from deepmd.pt.train.wrapper import ModelWrapper

    model = get_model(params)
    model = model.to(torch.device("cpu"))
    wrapper = ModelWrapper(model)
    return wrapper


def _heads(wrapper):
    """The per-head models, which is where the real modules live."""
    inner = getattr(wrapper, "model", None)
    if isinstance(inner, dict):
        return inner
    return {"Default": inner}


def _table(module, label: str) -> None:
    rows = []
    for name, child in module.named_children():
        parameters = sum(p.numel() for p in child.parameters())
        rows.append((name, type(child).__name__, parameters,
                     sum(1 for _ in child.modules())))
    print(f"\n=== {label}: direct submodules ===")
    width = max((len(row[0]) for row in rows), default=12)
    print(f"{'submodule'.ljust(width)}  {'type':34s} {'params':>12s} {'modules':>8s}")
    for name, kind, parameters, count in rows:
        print(f"{name.ljust(width)}  {kind:34s} {parameters:12,d} {count:8d}")
    total = sum(p.numel() for p in module.parameters())
    print(f"{'total'.ljust(width)}  {'':34s} {total:12,d} "
          f"{sum(1 for _ in module.modules()):8d}")


def _leaf_table(module) -> None:
    print("\n=== parameters (name, shape, numel) ===")
    for name, tensor in module.named_parameters():
        print(f"{name:78s} {str(tuple(tensor.shape)):24s} {tensor.numel():>9,d}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="a .pt checkpoint")
    parser.add_argument("--descriptor", action="store_true",
                        help="print only the descriptor subtree")
    parser.add_argument("--summary", action="store_true",
                        help="print the per-submodule table instead of the full repr")
    parser.add_argument("--params", action="store_true",
                        help="print every parameter with its shape")
    parser.add_argument("--out", help="write the text here instead of stdout")
    parser.add_argument("--head", default=None, help="model head (default: all)")
    args = parser.parse_args(argv)

    params, state = _load(args.model)
    wrapper = _build(params)
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"# weights: missing {len(missing)}, unexpected {len(unexpected)}",
              file=sys.stderr)

    heads = _heads(wrapper)
    if args.head:
        heads = {args.head: heads[args.head]}

    target = heads
    if args.descriptor:
        target = {name: head.atomic_model.descriptor for name, head in heads.items()}

    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit(f"# {args.model}")
    emit(f"# descriptor type: {params['descriptor'].get('type')}, "
         f"lmax={params['descriptor'].get('lmax')}, "
         f"mmax={params['descriptor'].get('mmax')}, "
         f"channels={params['descriptor'].get('channels')}, "
         f"n_blocks={params['descriptor'].get('n_blocks')}")
    emit(f"# type_map: {len(params.get('type_map') or [])} elements")

    for name, module in target.items():
        emit(f"\n{'=' * 78}\n== head {name}\n{'=' * 78}")
        if args.summary:
            emit(str(type(module).__name__))
            _redirected(emit, _table, module, name)
        else:
            emit(repr(module))
            _redirected(emit, _table, module, name)
        if args.params:
            _redirected(emit, _leaf_table, module)

    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
        print(f"wrote {len(lines)} lines to {args.out}")
    else:
        print(text)
    return 0


def _redirected(emit, function, *arguments) -> None:
    """Run a printing helper so its output lands in the same list."""
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        function(*arguments)
    for line in buffer.getvalue().splitlines():
        emit(line)


if __name__ == "__main__":
    sys.exit(main())
