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

The graph is traced through ``forward_common_lower`` rather than the export
wrapper, because a spin checkpoint (``has_spin``) takes a signature the wrapper
does not accept.  ``--mode atomic`` traces the energy path; the force path calls
``autograd.grad`` inside the graph, and ``make_fx`` cannot trace that on this
checkpoint (``The differentiated Tensor at index 1 appears to not have been
used``), so ``--mode full`` reports that instead of pretending.

``--stub-lib`` lets this run on a CPU-only host that has the DeepMD *source*
but not the compiled ``deepmd.lib``: the graph is the dense reference path, so
the extension is never called.

Usage::

    python tests/deepmd/profile/print_fx_graph.py --model-file "$MODEL" --summary
    python tests/deepmd/profile/print_fx_graph.py --model-file "$MODEL" \\
        --stub-lib --out /tmp/dpa4_cpu.fx.txt

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
    # ``aten.sum.dim_IntList`` / ``aten.mul.Tensor`` / ``aten.bmm.default``:
    # the last component is the overload, not part of the op name.
    parts = text.split(".")
    if len(parts) >= 3 and parts[0] in {"aten", "prims", "quantized", "dim"}:
        return ".".join(parts[: -1])
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


def install_stub_lib() -> str:
    """Satisfy ``deepmd.lib`` from a source checkout with no compiled extension.

    ``deepmd.env`` only needs ``deepmd.lib.__path__[0]`` (plus a
    ``run_config.ini`` beside it) for the pure-Python path, so a module with a
    directory of its own is enough -- and the traced graph is the dense
    reference path, which never calls into the extension.
    """
    import sys
    import tempfile
    import types

    import deepmd

    if hasattr(deepmd, "lib"):
        return "already importable"
    directory = Path(tempfile.mkdtemp(prefix="deepmd_lib_stub_"))
    source = Path(deepmd.__file__).resolve().parents[1] / "source" / "config" / "run_config.ini"
    if source.is_file():
        (directory / "run_config.ini").write_bytes(source.read_bytes())
    stub = types.ModuleType("deepmd.lib")
    stub.__path__ = [str(directory)]
    sys.modules["deepmd.lib"] = stub
    deepmd.lib = stub
    return str(directory)


def build_graph(model_file: str, *, nloc: int, target_device: str,
                mode: str = "atomic", trace_device: str = "cpu") -> tuple:
    """Load the checkpoint, trace the lower graph on CPU, return (graph, inputs)."""
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

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
    # ``trace_device`` is where the graph is traced; ``target_device`` only
    # selects which device-dependent branches the checkpoint resolves for.
    # They differ when a CPU-readable artifact is wanted from a CUDA build.
    model.to(torch.device(trace_device))
    model.fold_vacuum_reference()
    model.to(torch.device(trace_device))

    has_spin = bool(getattr(model, "has_spin", False))
    _, sample = _resolve_nframes(
        model, nloc=nloc, device=torch.device(trace_device), has_spin=has_spin
    )
    flags = {}
    if mode == "atomic":
        flags["atomic_output_only"] = True
    elif mode == "embedding":
        flags["embedding_only"] = True

    def lower_fn(coord, atype, edge_index, edge_vec, edge_scatter_index, edge_mask,
                 spin, charge_spin):
        return model.forward_common_lower(
            coord, atype, edge_index, edge_vec, edge_scatter_index, edge_mask,
            None, None, None, None, None,
            charge_spin=charge_spin, spin=spin, use_compile=False, **flags
        )

    coord, atype, edge_index, edge_vec, scatter, mask, spin, fparam, aparam, charge_spin = sample
    traced = make_fx(lower_fn, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
        coord, atype, edge_index, edge_vec, scatter, mask, spin, charge_spin
    )
    return traced, sample, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                        help="target the checkpoint's kernel levels resolve for")
    parser.add_argument("--trace-device", default="cpu", choices=("cpu", "cuda"),
                        help="device the graph is actually traced on; needs a DCU for "
                             "cuda, and is what makes device-only branches and custom "
                             "operators appear in the graph")
    parser.add_argument("--nloc", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stub-lib", action="store_true",
                        help="satisfy deepmd.lib from the source tree (no compiled extension)")
    parser.add_argument("--mode", default="atomic", choices=("atomic", "embedding", "full"),
                        help="atomic: energy path; embedding: descriptor only; "
                             "full: force path (not traceable on this checkpoint)")
    parser.add_argument("--summary", action="store_true", help="skip the full graph text")
    parser.add_argument("--max-nodes", type=int, default=0, help="truncate the printed graph")
    args = parser.parse_args(argv)

    if args.stub_lib:
        print(f"deepmd.lib : stubbed at {install_stub_lib()}")
    traced, sample, model = build_graph(
        args.model_file, nloc=args.nloc, target_device=args.device,
        mode=args.mode, trace_device=args.trace_device,
    )
    print(f"traced on  : {args.trace_device}")
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
