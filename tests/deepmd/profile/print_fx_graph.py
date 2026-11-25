"""Print the whole DPA4 / SeZM model as an FX graph.

The graph to print is the one DeepMD itself traces for its AOTInductor freeze:
``ModelWrapper`` -> ``forward_common_lower_exportable``, which runs ``make_fx``
over ``forward_common_lower`` so the inner ``autograd.grad(energy, edge_vec)``
that produces the force is unfolded into ordinary ATen nodes.  That is the
complete model minus the host-side neighbor-list construction, which is exactly
where the graph stops being a tensor program.

Tracing always happens on CPU (DeepMD pins its own freeze the same way); the
``--device`` flag only selects the *target* the checkpoint's kernel levels are
resolved for, which changes device-dependent Python branches baked into the
graph (block-diagonal SO(2) contraction on any non-CPU target, the fused GIE
scatter on CUDA).  So ``--device cpu`` is the honest CPU graph, and
``--device cuda`` is the DCU-oriented graph, both traced on CPU.

Usage::

    python tests/deepmd/profile/print_fx_graph.py --model-file "$MODEL" --summary
    python tests/deepmd/profile/print_fx_graph.py --model-file "$MODEL" \\
        --device cuda --out /tmp/dpa4_cuda.fx.txt

The full text goes to stdout and, with ``--out``, to a file; ``--summary``
prints only the header and the op-family table.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

#: ATen base name -> family. Anything unlisted lands in ``other``.
_FAMILIES = {
    "gemm": ("bmm", "mm", "addmm", "baddbmm", "matmul", "linear", "einsum"),
    "conv": ("convolution", "conv2d", "conv1d"),
    "index": (
        "gather",
        "index",
        "index_select",
        "index_add",
        "index_add_",
        "index_copy",
        "index_copy_",
        "index_put_",
        "_index_put_impl_",
        "scatter",
        "scatter_add",
        "scatter_add_",
        "scatter_reduce",
        "nonzero",
        "masked_select",
        "take",
        "embedding",
        "unique",
    ),
    "reduce": (
        "sum",
        "mean",
        "max",
        "min",
        "amax",
        "amin",
        "norm",
        "prod",
        "argmax",
        "argmin",
        "any",
        "all",
        "_softmax",
        "_softmax_backward_data",
        "_log_softmax",
        "logsumexp",
        "std",
        "var",
    ),
    "copy": ("copy_", "clone", "to", "_to_copy", "contiguous", "type_as", "_copy"),
    "layout": (
        "view",
        "reshape",
        "permute",
        "transpose",
        "as_strided",
        "expand",
        "squeeze",
        "unsqueeze",
        "cat",
        "stack",
        "split",
        "chunk",
        "narrow",
        "t",
        "_unsafe_view",
        "_reshape_alias",
        "select",
        "flatten",
    ),
    "init": ("zeros", "ones", "empty", "fill_", "zero_", "arange", "full", "new_zeros", "new_empty"),
    "elementwise": (
        "add",
        "add_",
        "sub",
        "mul",
        "mul_",
        "div",
        "neg",
        "pow",
        "sqrt",
        "rsqrt",
        "exp",
        "log",
        "sigmoid",
        "sigmoid_backward",
        "silu",
        "silu_backward",
        "tanh",
        "where",
        "clamp",
        "reciprocal",
        "abs",
        "fma",
        "leaky_relu",
        "gelu",
    ),
}


def _family_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for family, names in _FAMILIES.items():
        for name in names:
            table[name] = family
    return table


_FAMILY_OF = _family_table()


def op_name(target) -> str:
    """A stable name for a node target.

    ``make_fx`` nodes carry an ATen overload (``aten.bmm.default``); a
    ``symbolic_trace`` graph carries plain callables (``torch.bmm``,
    ``operator.mul``).  Both must reduce to the same ``module.name`` shape so the
    family table sees ``bmm`` / ``mul`` either way.
    """
    if hasattr(target, "_schema") or hasattr(target, "_schemas"):
        # ATen overload/packet: ``str`` is already ``aten.bmm.default``.
        text = str(target)
    else:
        name = getattr(target, "__name__", None)
        if isinstance(name, str) and name:
            owner = getattr(target, "__module__", "") or ""
            return f"{owner}.{name}" if owner else name
        text = str(target)
    if "torch.ops." in text:
        text = text.split("torch.ops.", 1)[1]
    for suffix in (".default", ".Tensor", ".Scalar"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def family_of(target) -> str:
    name = op_name(target)
    base = name.split(".")[-1]
    return _FAMILY_OF.get(base, "other")


def summarise(graph) -> dict:
    """Node count per op and per family, plus the call targets."""
    ops: collections.Counter = collections.Counter()
    families: collections.Counter = collections.Counter()
    call_functions: collections.Counter = collections.Counter()
    total = 0
    for node in graph.nodes:
        total += 1
        if node.op == "call_function":
            ops[op_name(node.target)] += 1
            families[family_of(node.target)] += 1
        elif node.op == "call_module":
            call_functions[f"module:{node.target}"] += 1
    return {
        "nodes": total,
        "call_function": sum(ops.values()),
        "ops": ops,
        "families": families,
        "modules": call_functions,
    }


def _pct(part: int, whole: int) -> str:
    return f"{part * 100.0 / whole:5.1f}%" if whole else "  0.0%"


def print_summary(report: dict) -> None:
    print(f"nodes: {report['nodes']}  call_function: {report['call_function']}  "
          f"distinct ops: {len(report['ops'])}")
    if report["modules"]:
        print("call_module targets: " + ", ".join(sorted(report["modules"])[:6]))
    print("\n-- families --")
    for family, count in report["families"].most_common():
        print(f"  {family:12s} {count:6d}  {_pct(count, report['call_function'])}")
    print("\n-- top ops --")
    for op, count in report["ops"].most_common(25):
        print(f"  {op:44s} {count:6d}  {_pct(count, report['call_function'])}")


def build_graph(model_file: str, *, nloc: int, target_device: str) -> tuple:
    """Load the checkpoint, trace the lower graph on CPU, return (graph, inputs)."""
    import torch

    from deepmd.pt.entrypoints.freeze_pt2 import (
        _extract_state_and_params,
        _resolve_nframes,
        _select_model_head,
    )
    from deepmd.pt.model.model import get_model
    from deepmd.pt.train.wrapper import ModelWrapper

    raw = torch.load(model_file, map_location="cpu", weights_only=False)
    state_dict, params = _extract_state_and_params(raw)
    state_dict, params = _select_model_head(state_dict, params, None)
    model = get_model(params)
    ModelWrapper(model).load_state_dict(state_dict)
    model.eval()
    model.to(torch.device(target_device))
    model.fold_vacuum_reference()
    model.to("cpu")

    has_spin = bool(getattr(model, "has_spin", False))
    _, sample = _resolve_nframes(model, nloc=nloc, device=torch.device("cpu"), has_spin=has_spin)
    traced = model.forward_common_lower_exportable(*sample)
    return traced, sample, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                        help="target the checkpoint's kernel levels resolve for; "
                             "tracing is always CPU")
    parser.add_argument("--nloc", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary", action="store_true", help="skip the full graph text")
    parser.add_argument("--max-nodes", type=int, default=0, help="truncate the printed graph")
    args = parser.parse_args(argv)

    traced, sample, model = build_graph(
        args.model_file, nloc=args.nloc, target_device=args.device
    )
    graph = traced.graph
    report = summarise(graph)

    print(f"checkpoint : {args.model_file}")
    print(f"trace on   : cpu (make_fx, symbolic)")
    print(f"target     : {args.device}")
    print(f"nloc       : {args.nloc}   spin: {bool(getattr(model, 'has_spin', False))}")
    shapes = [
        f"{i}:{tuple(t.shape)}" for i, t in enumerate(sample) if t is not None
    ]
    print("inputs     : " + ", ".join(shapes))
    print()
    print_summary(report)

    lines = str(graph).splitlines()
    if args.max_nodes and len(lines) > args.max_nodes:
        lines = lines[: args.max_nodes] + [f"... ({len(lines) - args.max_nodes} more lines)"]
    print("\n-- graph --")
    text = "\n".join(lines)
    if not args.summary:
        print(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(
                [
                    f"# {args.model_file}",
                    f"# trace=cpu target={args.device} nloc={args.nloc}",
                    f"# nodes={report['nodes']} call_function={report['call_function']}",
                    "",
                    text,
                ]
            )
            + "\n"
        )
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
