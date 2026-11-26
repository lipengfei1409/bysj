"""Extract one module's subgraph from the whole-model FX graph.

The whole-model graph is 3400 nodes and 54.6% layout; reading the SO(2)
convolution out of that by eye is not possible.  This slices it out and prints
only the slice, with the shape of every node -- which is what a report needs.

``--select conv`` (default)
    Capture the real arguments of the first ``SO2Convolution`` instance with a
    forward pre-hook during one eager pass, then trace *that module alone*.  The
    result is exactly the convolution's graph, with no guessing about where it
    starts and ends -- the whole-model trace leaves no module scope on the nodes,
    so slicing the 3749-node graph by an anchor swallows the entire prefix.

Two selectors for when a module cannot be isolated:

``--module REGEX``
    Keep every node whose ``node.meta["nn_module_stack"]`` names a module
    matching the regex.  Exact, and the default when the trace carries scope.

``--anchor REGEX``
    Keep the largest region around each node whose target matches the regex:
    its transitive inputs, stopping at nodes outside the anchor's module scope
    (or, with no scope at all, at the previous anchor).  A fallback, and the
    only option when the trace has no module stack.

Usage::

    python tests/deepmd/profile/extract_fx_subgraph.py --model-file "$MODEL" \\
        --stub-lib --module 'SO2Convolution|so2_conv' --out /tmp/so2_conv.fx.txt
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _printer():
    spec = importlib.util.spec_from_file_location("_pfg", _HERE / "print_fx_graph.py")
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError("cannot load print_fx_graph.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_stack(node) -> list[str]:
    """``[(qualname, class)]`` the node was traced under, outermost first."""
    stack = (node.meta or {}).get("nn_module_stack") or {}
    values = []
    for value in stack.values():
        if isinstance(value, (tuple, list)) and len(value) == 2:
            values.append(str(value[0]))
        else:
            values.append(str(value))
    return values


def shape_of(node):
    meta = (node.meta or {}).get("tensor_meta")
    if meta is None:
        return None
    shape = getattr(meta, "shape", None)
    if shape is None and isinstance(meta, (tuple, list)) and meta:
        shape = getattr(meta[0], "shape", None)
    if shape is None:
        return None
    return tuple(int(d) if isinstance(d, int) else str(d) for d in shape)


def slice_by_module(graph, pattern: re.Pattern) -> tuple[list, str]:
    selected = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        stack = module_stack(node)
        if any(pattern.search(name) for name in stack):
            selected.append(node)
    scoped = sum(1 for n in graph.nodes if module_stack(n))
    return selected, f"{scoped}/{len(list(graph.nodes))} nodes carry a module stack"


def slice_by_anchor(graph, pattern: re.Pattern) -> tuple[list, str]:
    """Inputs of each matching node, stopping at the previous anchor."""
    nodes = list(graph.nodes)
    anchors = [n for n in nodes if n.op == "call_function" and pattern.search(str(n.target))]
    if not anchors:
        return [], "no node matched the anchor"
    order = {node: index for index, node in enumerate(nodes)}
    keep: set = set()
    for anchor in anchors:
        stack = set(module_stack(anchor))
        todo = [anchor]
        seen: set = set()
        while todo:
            node = todo.pop()
            if node in seen or node.op not in ("call_function", "get_attr"):
                continue
            seen.add(node)
            # Stop at a node owned by a different module when scope exists;
            # without scope, stop at another anchor to keep the span local.
            other_stack = set(module_stack(node))
            if stack and other_stack and not (stack & other_stack):
                continue
            if node is not anchor and node in anchors:
                keep.add(node)  # keep the anchor itself, do not cross it
                continue
            keep.add(node)
            todo.extend(node.all_input_nodes)
    ordered = sorted(keep, key=lambda n: order[n])
    return ordered, f"{len(anchors)} anchors, {len(ordered)} nodes in their input closures"


def trace_conv(model, sample, index: int = 0) -> tuple:
    """Trace the first ``SO2Convolution`` instance on its real inputs.

    The whole-model trace records no ``nn_module_stack``, so the module cannot be
    selected out of it.  Capturing the module's own call is exact: the hook sees
    the tensors the descriptor really passes, and ``make_fx`` then traces that
    one module.
    """
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Convolution

    coord, atype, edge_index, edge_vec, scatter, mask, spin, fparam, aparam, charge_spin = sample
    captured: dict = {}

    def pre_hook(module, args):
        # One entry per instance: each block is called once per forward.
        if id(module) not in captured:
            captured[id(module)] = (module, args)

    convs = [m for m in model.modules() if isinstance(m, SO2Convolution)]
    handles = [conv.register_forward_pre_hook(pre_hook) for conv in convs]
    try:
        with torch.no_grad():
            model.forward_common_lower(
                coord, atype, edge_index, edge_vec, scatter, mask,
                None, None, None, None, None,
                charge_spin=charge_spin, spin=spin,
                use_compile=False, atomic_output_only=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if not captured:
        raise RuntimeError("the eager pass never called an SO2Convolution instance")
    calls = list(captured.values())
    if not 0 <= index < len(calls):
        raise IndexError(
            f"this checkpoint has {len(calls)} SO2Convolution instances; --index {index} is out of range"
        )

    module, args = calls[index]
    # `forward(x, edge_cache, radial_feat)`: the cache is a container of tensors,
    # and make_fx cannot build concrete arguments for it, so close over it and
    # trace a tensor-only callable instead.  The cache then appears in the graph
    # as the constants it holds, which is what we want to read.
    edge_cache = args[1]
    tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]

    def call(tensor_args):
        x, radial_feat = tensor_args[0], tensor_args[-1]
        return module(x, edge_cache, radial_feat)

    traced = make_fx(call, tracing_mode="real")(tensor_args)
    note = (
        f"{len(calls)} SO2Convolution instances; traced #{index} "
        f"(lmax={getattr(module, 'lmax', '?')}, channels={getattr(module, 'channels', '?')}) "
        f"on {len([a for a in args if hasattr(a, 'shape')])} tensor inputs"
    )
    return traced, note, index


def render(nodes: list) -> str:
    lines = []
    for node in nodes:
        shape = shape_of(node)
        name = str(node.target).replace("torch.ops.", "")
        args = ", ".join(
            (a.name if hasattr(a, "name") else repr(a)) for a in node.args
        )
        suffix = f"  {shape}" if shape else ""
        lines.append(f"%{node.name:28s} {name:38s} ({args}){suffix}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--nloc", type=int, default=7)
    parser.add_argument("--mode", default="atomic", choices=("atomic", "embedding"))
    parser.add_argument("--module", default=r"SO2Convolution|so2_conv|sezm_nn\.so2")
    parser.add_argument("--anchor", default=r"_softmax")
    parser.add_argument("--select", default="conv", choices=("conv", "auto", "module", "anchor"))
    parser.add_argument("--stub-lib", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-nodes", type=int, default=0)
    args = parser.parse_args(argv)

    printer = _printer()
    if args.stub_lib:
        print(f"deepmd.lib : stubbed at {printer.install_stub_lib()}")
    traced, sample, model = printer.build_graph(
        args.model_file, nloc=args.nloc, target_device=args.device, mode=args.mode
    )
    graph = traced.graph

    pattern = re.compile(args.module)
    if args.select == "conv":
        conv_graph, note, _ = trace_conv(model, sample)
        graph = conv_graph.graph
        selected, _ = slice_by_module(graph, re.compile(r".*"))
        if not selected:
            selected = [n for n in graph.nodes if n.op == "call_function"]
        how = "conv(module trace)"
    else:
        selected, note = slice_by_module(graph, pattern)
        how = "module"
        if args.select == "anchor" or (args.select == "auto" and not selected):
            selected, note = slice_by_anchor(graph, re.compile(args.anchor))
            how = f"anchor({args.anchor})"

    print(f"graph         : {len(list(graph.nodes))} nodes")
    print(f"selector      : {how}  -> {len(selected)} nodes")
    print(f"note          : {note}")
    if not selected:
        print("\nno subgraph matched; try --select anchor --anchor '_softmax' "
              "or a different --module regex")
        return 1

    families: collections.Counter = collections.Counter()
    ops: collections.Counter = collections.Counter()
    for node in selected:
        families[printer.family_of(node.target)] += 1
        ops[printer.op_name(node.target)] += 1
    total = len(selected)
    print("\n-- families --")
    for family, count in families.most_common():
        print(f"  {family:12s} {count:5d}  {count * 100.0 / total:5.1f}%")
    print("\n-- top ops --")
    for op, count in ops.most_common(20):
        print(f"  {op:36s} {count:5d}  {count * 100.0 / total:5.1f}%")

    text = render(selected)
    lines = text.splitlines()
    if args.max_nodes and len(lines) > args.max_nodes:
        lines = lines[: args.max_nodes] + [f"... ({len(lines) - args.max_nodes} more)"]
    print("\n-- nodes --")
    print("\n".join(lines))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(
                [
                    f"# {args.model_file}",
                    f"# selector={how} nodes={total} mode={args.mode}",
                    text,
                ]
            )
            + "\n"
        )
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
