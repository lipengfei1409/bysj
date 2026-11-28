#!/usr/bin/env python3
"""Build the SEZM/DPA4 descriptor on CPU and print it.

The descriptor is `DescrptSeZM` in `deepmd/pt/model/descriptor/sezm.py` (registered as
`sezm` / `dpa4`).  This script takes its constructor arguments from a checkpoint, builds
the module on the CPU, and prints the result: first the full `repr`, then a short table of
which submodules hold the parameters.

    python3 tools/print_sezm_model.py --model DPA4-Air-OMol25-100M-v20260820.pt
    python3 tools/print_sezm_model.py --model ... --summary      # table only
    python3 tools/print_sezm_model.py --model ... --weights      # also load the weights

`--model` may be the `.pt` checkpoint (whose `_extra_state` carries `model_params`) or the
`.json` descriptor config.  `ntypes` comes from the checkpoint's `type_map` when it is not
in the descriptor config, because the descriptor does not store it.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import sys


def _descriptor_config(path: str) -> tuple[dict, dict]:
    """Return (descriptor config, full model params) from a .pt or .json file."""
    if path.endswith(".json"):
        with open(path) as handle:
            params = json.load(handle)
        params = params.get("model", params)
        return params["descriptor"], params

    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    params = state["_extra_state"]["model_params"]
    descriptor = dict(params["descriptor"])
    # the descriptor does not carry `ntypes`; it comes from the model's type map
    type_map = params.get("type_map") or []
    descriptor.setdefault("ntypes", len(type_map) if type_map else 0)
    return descriptor, params


def _filter_kwargs(cls, config: dict) -> dict:
    signature = inspect.signature(cls.__init__)
    accepted = {
        name for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    }
    dropped = sorted(set(config) - accepted)
    if dropped:
        print(f"# ignoring config keys the constructor does not take: {dropped}",
              file=sys.stderr)
    return {key: value for key, value in config.items() if key in accepted}


def _summary(module) -> None:
    rows = []
    for name, child in module.named_children():
        parameters = sum(p.numel() for p in child.parameters())
        rows.append((name, type(child).__name__, parameters, sum(
            1 for _ in child.modules())))
    width = max((len(row[0]) for row in rows), default=10)
    print(f"\n{'submodule'.ljust(width)}  {'type':28s} {'params':>10s} {'modules':>7s}")
    for name, kind, parameters, count in rows:
        print(f"{name.ljust(width)}  {kind:28s} {parameters:10d} {count:7d}")
    total = sum(p.numel() for p in module.parameters())
    print(f"\ntotal: {total:,} parameters in {sum(1 for _ in module.modules())} modules")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True,
                        help="a .pt checkpoint or a .json descriptor config")
    parser.add_argument("--weights", action="store_true",
                        help="also load the descriptor weights from the checkpoint")
    parser.add_argument("--summary", action="store_true",
                        help="print the submodule table instead of the full repr")
    parser.add_argument("--head", default="model.Default.atomic_model.descriptor",
                        help="state-dict prefix of the descriptor (for --weights)")
    parser.add_argument("--shapes", action="store_true",
                        help="print a sample forward's shapes (needs an input)")
    args = parser.parse_args(argv)

    import torch

    from deepmd.pt.model.descriptor.base_descriptor import BaseDescriptor

    config, _ = _descriptor_config(args.model)
    descriptor_type = config.get("type", "dpa4")
    cls = BaseDescriptor.get_class_by_type(descriptor_type)

    with torch.device("cpu"), contextlib.suppress(Exception):
        # construction may consult the global device; CPU is what we want here
        import deepmd.env as env

        env.DEVICE = torch.device("cpu")

    descriptor = cls(**_filter_kwargs(cls, config))
    descriptor = descriptor.to(torch.device("cpu"))
    descriptor.eval()

    if args.weights:
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        prefix = args.head + "."
        subset = {
            key[len(prefix):]: value for key, value in state.items()
            if key.startswith(prefix)
        }
        missing, unexpected = descriptor.load_state_dict(subset, strict=False)
        print(f"# weights: {len(subset)} tensors, missing {len(missing)}, "
              f"unexpected {len(unexpected)}", file=sys.stderr)

    if args.summary:
        _summary(descriptor)
    else:
        print(descriptor)
        _summary(descriptor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
