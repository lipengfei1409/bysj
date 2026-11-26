"""Print the FX graph of a native DeepMD DPA4 module, one target at a time.

One entry point, a registry of targets, and everything traced on CPU:

    python tests/deepmd/profile/fx_graph.py --list-targets
    python tests/deepmd/profile/fx_graph.py --model-file "$MODEL" --target model
    python tests/deepmd/profile/fx_graph.py --model-file "$MODEL" --target so2_conv

Targets
-------
``model``
    The whole DPA4/SeZM model as DeepMD itself traces it for AOTInductor:
    ``forward_common_lower`` under ``make_fx(symbolic)``, energy path.  The
    force path builds an ``autograd.grad`` inside the graph, which ``make_fx``
    cannot trace on these checkpoints, so ``--mode full`` reports that instead
    of silently downgrading.

``so2_conv``
    One ``SO2Convolution`` instance -- the per-edge span that
    ``dpa4_so2_conv`` fuses.  A forward pre-hook captures the tensors the
    descriptor actually passes, then that module alone is traced.  Slicing it
    out of the whole-model graph does not work: ``make_fx`` records no
    ``nn_module_stack``, so an anchor slice takes in the entire prefix.

Adding a target is one entry in ``TARGETS``: a builder with the signature
``(model, sample, options) -> (graph, note)`` plus a one-line description.

Output is one line per node with its shape (``--raw`` prints the SSA text
instead), preceded by the op-family table.  ``--out`` writes the same text.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _sibling(name: str, module: str):
    spec = importlib.util.spec_from_file_location(module, _HERE / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError(f"cannot load {name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _model_graph(model, sample, traced, options) -> tuple:
    """The graph DeepMD compiles: ``forward_common_lower`` under make_fx."""
    note = (
        f"forward_common_lower under make_fx(symbolic), mode={options.mode}, "
        f"nloc={options.nloc}, target={options.device}, traced on {options.trace_device}"
    )
    return traced.graph, note


def _so2_conv_graph(model, sample, traced, options) -> tuple:
    """Trace one ``SO2Convolution`` on the inputs the descriptor really passes."""
    extractor = _sibling("extract_fx_subgraph", "_fx_extractor")
    conv_graph, note, _ = extractor.trace_conv(model, sample, index=options.index)
    return conv_graph.graph, note


#: target name -> (one-line description, builder)
TARGETS = {
    "model": ("whole DPA4/SeZM model (forward_common_lower, energy path)", _model_graph),
    "so2_conv": ("one SO2Convolution instance (the span dpa4_so2_conv fuses)", _so2_conv_graph),
}


def print_targets() -> None:
    width = max(len(name) for name in TARGETS)
    for name, (description, _) in TARGETS.items():
        print(f"  {name:{width}s}  {description}")


def _observed_device(graph) -> str:
    """Device the traced tensors actually live on -- the trace's own evidence."""
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        meta = (node.meta or {}).get("tensor_meta")
        device = getattr(meta, "device", None)
        if device is not None:
            return str(device)
        if isinstance(meta, (tuple, list)) and meta:
            device = getattr(meta[0], "device", None)
            if device is not None:
                return str(device)
    return "unknown"


def _describe(graph, options, printer, extractor) -> tuple[collections.Counter, collections.Counter, str]:
    # Only real operations are counted; ``get_attr`` nodes are parameters and
    # constants, which show up as arguments in the listing anyway.
    nodes = [n for n in graph.nodes if n.op in ("call_function", "call_module")]
    ops: collections.Counter = collections.Counter()
    families: collections.Counter = collections.Counter()
    for node in nodes:
        ops[printer.op_name(node.target)] += 1
        families[printer.family_of(node.target)] += 1
    if options.raw:
        text = str(graph)
    else:
        text = extractor.render(nodes)
    return ops, families, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--model-file")
    parser.add_argument("--target", default="model", choices=sorted(TARGETS))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                        help="target the checkpoint's kernel levels resolve for")
    parser.add_argument("--trace-device", default="cpu", choices=("cpu", "cuda"),
                        help="device the graph is traced on (needs a DCU for cuda)")
    parser.add_argument("--mode", default="atomic", choices=("atomic", "embedding"),
                        help="model target only: energy path or descriptor only")
    parser.add_argument("--nloc", type=int, default=7)
    parser.add_argument("--index", type=int, default=0,
                        help="so2_conv target: which instance to print (0-based)")
    parser.add_argument("--stub-lib", action="store_true",
                        help="satisfy deepmd.lib from the source tree (no compiled extension)")
    parser.add_argument("--raw", action="store_true", help="print the SSA text, not the shape table")
    parser.add_argument("--summary", action="store_true", help="skip the node listing")
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.list_targets:
        print("targets:")
        print_targets()
        return 0
    if not args.model_file:
        parser.error("--model-file is required unless --list-targets is given")

    printer = _sibling("print_fx_graph", "_fx_printer")
    extractor = _sibling("extract_fx_subgraph", "_fx_extractor")
    if args.stub_lib:
        print(f"deepmd.lib : stubbed at {printer.install_stub_lib()}")

    builder = TARGETS[args.target][1]
    traced, sample, model = printer.build_graph(
        args.model_file,
        nloc=args.nloc,
        target_device=args.device,
        mode=args.mode if args.target == "model" else "atomic",
        trace_device=args.trace_device,
    )
    graph, note = builder(model, sample, traced, args)

    ops, families, text = _describe(graph, args, printer, extractor)
    total = sum(families.values())
    print(f"checkpoint : {args.model_file}")
    print(f"target     : {args.target}  ({TARGETS[args.target][0]})")
    print(f"{'mode':11s}: {args.mode if args.target == 'model' else 'atomic (eager capture)'}")
    print(f"traced on   : {args.trace_device}  (observed: {_observed_device(graph)})")
    print(f"note       : {note}")
    print(f"nodes      : {total} call nodes, {len(ops)} distinct ops")

    print("\n-- families --")
    for family, count in families.most_common():
        print(f"  {family:12s} {count:6d}  {count * 100.0 / total:5.1f}%")
    print("\n-- top ops --")
    for op, count in ops.most_common(20):
        print(f"  {op:36s} {count:6d}  {count * 100.0 / total:5.1f}%")

    lines = text.splitlines()
    if args.max_nodes and len(lines) > args.max_nodes:
        lines = lines[: args.max_nodes] + [f"... ({len(lines) - args.max_nodes} more lines)"]
    if not args.summary:
        print("\n-- nodes --")
        print("\n".join(lines))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(
                [
                    f"# {args.model_file}",
                    f"# target={args.target} nodes={total} note={note}",
                    "",
                    "\n".join(lines),
                ]
            )
            + "\n"
        )
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
