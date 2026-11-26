"""Print the FX graph of a native DeepMD DPA4 module, one target at a time.

One script, a registry of targets, everything traced on CPU by default:

    python tests/deepmd/profile/fx_graph.py --list-targets
    python tests/deepmd/profile/fx_graph.py --model-file "$MODEL" --target model
    python tests/deepmd/profile/fx_graph.py --model-file "$MODEL" --target so2_conv
    python tests/deepmd/profile/fx_graph.py --model-file "$MODEL" --target so2_conv \\
        --device cuda --trace-device cuda

Targets
-------
``model``
    The whole DPA4/SeZM model as DeepMD itself traces it for AOTInductor:
    ``forward_common_lower`` under ``make_fx(symbolic)``, energy path.  The
    force path builds an ``autograd.grad`` inside the graph, which ``make_fx``
    cannot trace on these checkpoints, so ``--mode full`` is not offered.

``so2_conv``
    One ``SO2Convolution`` instance -- the per-edge span ``dpa4_so2_conv``
    fuses.  A forward pre-hook captures the tensors the descriptor actually
    passes, then that module alone is traced.  Slicing it out of the whole-model
    graph does not work: ``make_fx`` records no ``nn_module_stack``, so an
    anchor slice takes in the entire prefix (2246 of 3749 nodes).

Adding a target is one entry in ``TARGETS``: a builder with the signature
``(model, sample, traced, options) -> (graph, note)`` plus a one-line
description.  ``--list-targets`` prints them.

Devices
-------
``--device`` says which target the checkpoint's kernel levels resolve for (it
decides which device-dependent branches get baked).  ``--trace-device`` says
where the graph is actually computed.  They are independent: a CPU-readable
artifact from a CUDA build is ``--device cuda --trace-device cpu``, and the
graph that contains HIP-only branches and custom operators is
``--device cuda --trace-device cuda`` (needs a DCU).  The output prints both,
plus the device read back from the traced nodes' ``tensor_meta`` as evidence.

``--stub-lib`` runs on a host that has the DeepMD *source* but not the compiled
``deepmd.lib``: ``deepmd.env`` only needs ``deepmd.lib.__path__[0]`` and a
``run_config.ini`` beside it, and the traced graph is the dense reference path,
which never calls into the extension.
"""

from __future__ import annotations

import argparse
import collections
import sys
import tempfile
import types
from pathlib import Path

#: ATen base name -> family. Anything unlisted lands in ``other``.
_FAMILIES = {
    "gemm": ("bmm", "mm", "addmm", "baddbmm", "matmul", "linear", "einsum"),
    "conv": ("convolution", "conv2d", "conv1d"),
    "index": (
        "gather", "index", "index_select", "index_add", "index_add_", "index_copy",
        "index_copy_", "index_put_", "_index_put_impl_", "scatter", "scatter_add",
        "scatter_add_", "scatter_reduce", "nonzero", "masked_select", "take",
        "embedding", "unique",
    ),
    "reduce": (
        "sum", "mean", "max", "min", "amax", "amin", "norm", "prod", "argmax",
        "argmin", "any", "all", "_softmax", "_softmax_backward_data", "_log_softmax",
        "logsumexp", "std", "var",
    ),
    "copy": ("copy_", "clone", "to", "_to_copy", "contiguous", "type_as", "_copy"),
    "layout": (
        "view", "reshape", "permute", "transpose", "as_strided", "expand", "squeeze",
        "unsqueeze", "cat", "stack", "split", "chunk", "narrow", "t", "_unsafe_view",
        "_reshape_alias", "select", "flatten",
    ),
    "init": ("zeros", "ones", "empty", "fill_", "zero_", "arange", "full", "new_zeros",
             "new_empty"),
    "elementwise": (
        "add", "add_", "sub", "mul", "mul_", "div", "neg", "pow", "sqrt", "rsqrt",
        "exp", "log", "sigmoid", "sigmoid_backward", "silu", "silu_backward", "tanh",
        "where", "clamp", "reciprocal", "abs", "fma", "leaky_relu", "gelu",
    ),
}


def _family_table() -> dict:
    table: dict = {}
    for family, names in _FAMILIES.items():
        for name in names:
            table[name] = family
    return table


_FAMILY_OF = _family_table()


def op_name(target) -> str:
    """A stable name for a node target, overload and callable alike.

    ``make_fx`` nodes carry an ATen overload (``aten.bmm.default``,
    ``aten.sum.dim_IntList``); a ``symbolic_trace`` graph carries plain callables
    (``torch.bmm``, ``operator.mul``).  Both reduce to ``module.name`` so the
    family table sees ``bmm`` / ``mul`` / ``sum`` either way.
    """
    if hasattr(target, "_schema") or hasattr(target, "_schemas"):
        text = str(target)  # ATen overload/packet: already ``aten.bmm.default``
    else:
        name = getattr(target, "__name__", None)
        if isinstance(name, str) and name:
            owner = getattr(target, "__module__", "") or ""
            return f"{owner}.{name}" if owner else name
        text = str(target)
    if "torch.ops." in text:
        text = text.split("torch.ops.", 1)[1]
    parts = text.split(".")
    if len(parts) >= 3 and parts[0] in {"aten", "prims", "quantized", "dim"}:
        return ".".join(parts[:-1])
    return text


def family_of(target) -> str:
    return _FAMILY_OF.get(op_name(target).split(".")[-1], "other")


def module_stack(node) -> list:
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


def render(nodes: list) -> str:
    """One line per node: name, target, argument names, output shape."""
    lines = []
    for node in nodes:
        shape = shape_of(node)
        target = op_name(node.target)
        args = ", ".join((a.name if hasattr(a, "name") else repr(a)) for a in node.args)
        suffix = f"  {shape}" if shape else ""
        lines.append(f"%{node.name:28s} {target:38s} ({args}){suffix}")
    return "\n".join(lines)


def install_stub_lib() -> str:
    """Satisfy ``deepmd.lib`` from a source checkout with no compiled extension."""
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
    """Load the checkpoint, trace the lower graph, return (traced, sample, model)."""
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
    # The vacuum reference resolves on the tracing device, so it is folded there
    # and the exported graph carries no reference atoms.
    model.to(torch.device(trace_device))
    model.fold_vacuum_reference()

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


def trace_conv(model, sample, index: int = 0) -> tuple:
    """Trace one ``SO2Convolution`` on the inputs the descriptor really passes.

    The whole-model trace records no ``nn_module_stack``, so the module cannot be
    selected out of it; capturing the module's own call during one eager pass is
    exact.  ``index`` picks which of the interaction blocks to trace.
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
            f"this checkpoint has {len(calls)} SO2Convolution instances; "
            f"--index {index} is out of range"
        )

    module, args = calls[index]
    # `forward(x, edge_cache, radial_feat)`: the cache is a container of tensors
    # and make_fx cannot build concrete arguments for it, so close over it and
    # trace a tensor-only callable.  The cache then shows up as the constants it
    # holds, which is what we want to read.
    edge_cache = args[1]
    tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]

    def call(x, radial_feat):
        return module(x, edge_cache, radial_feat)

    # Two positional tensors, not a list: a list argument makes make_fx wrap the
    # signature in ``tree_flatten_spec``, which buries the real inputs.
    traced = make_fx(call, tracing_mode="real")(tensor_args[0], tensor_args[-1])
    note = (
        f"{len(calls)} SO2Convolution instances; traced #{index} "
        f"(lmax={getattr(module, 'lmax', '?')}, channels={getattr(module, 'channels', '?')}) "
        f"on {len(tensor_args)} tensor inputs"
    )
    return traced, note, index


def _model_graph(model, sample, traced, options) -> tuple:
    note = (
        f"forward_common_lower under make_fx(symbolic), mode={options.mode}, "
        f"nloc={options.nloc}, target={options.device}, traced on {options.trace_device}"
    )
    return traced, note


def _so2_conv_graph(model, sample, traced, options) -> tuple:
    conv_graph, note, _ = trace_conv(model, sample, index=options.index)
    return conv_graph, note


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
    """Device the traced tensors actually live on -- the trace's own evidence.

    A symbolic (``make_fx`` fake-tensor) trace keeps shapes and dtypes but drops
    the device, so "unknown" there is a property of the trace, not a failure to
    look; a real trace (the ``so2_conv`` target) reports the actual device.
    """
    seen_meta = False
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        meta = (node.meta or {}).get("tensor_meta")
        if meta is None:
            continue
        seen_meta = True
        device = getattr(meta, "device", None)
        if device is not None:
            return str(device)
        if isinstance(meta, (tuple, list)) and meta:
            device = getattr(meta[0], "device", None)
            if device is not None:
                return str(device)
    if seen_meta:
        return "unknown (a symbolic trace records no device metadata)"
    return "unknown (no tensor metadata)"


def _describe(module, options) -> tuple:
    # Only real operations are counted; ``get_attr`` nodes are parameters and
    # constants, and show up as arguments in the listing anyway.
    graph = module.graph
    nodes = [n for n in graph.nodes if n.op in ("call_function", "call_module")]
    ops: collections.Counter = collections.Counter()
    families: collections.Counter = collections.Counter()
    for node in nodes:
        ops[op_name(node.target)] += 1
        families[family_of(node.target)] += 1
    if options.format == "ssa":
        text = str(graph)
    elif options.format == "shapes":
        text = render(nodes)
    else:
        # The GraphModule's own print interface.
        text = str(module)
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
    parser.add_argument("--format", default="code", choices=("code", "ssa", "shapes"),
                        help="code: GraphModule's print interface (default); "
                             "ssa: str(graph); shapes: one line per node with its shape")
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

    if args.stub_lib:
        print(f"deepmd.lib : stubbed at {install_stub_lib()}")

    traced, sample, model = build_graph(
        args.model_file,
        nloc=args.nloc,
        target_device=args.device,
        mode=args.mode if args.target == "model" else "atomic",
        trace_device=args.trace_device,
    )
    builder = TARGETS[args.target][1]
    module, note = builder(model, sample, traced, args)

    ops, families, text = _describe(module, args)
    total = sum(families.values())
    print(f"checkpoint : {args.model_file}")
    print(f"target     : {args.target}  ({TARGETS[args.target][0]})")
    print(f"mode       : {args.mode if args.target == 'model' else 'atomic (eager capture)'}")
    print(f"traced on  : {args.trace_device}  (observed: {_observed_device(module.graph)})")
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
