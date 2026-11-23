"""Bind the fused segmented HIP operators into a loaded DeepMD DPA4 model.

DeepMD's SeZM descriptor already exposes the two seams this project needs, so
nothing in DeepMD has to be edited -- the adapter binds callables and every
call keeps a documented fallback to the stock implementation:

``segment_envelope_gated_softmax``
    The eager reference that ``SO2Convolution.attention_weights`` calls when no
    fused softmax is bound (``deepmd/pt/model/descriptor/sezm_nn/attention.py``).
    Replaced by :func:`kernels.segment.segment_softmax`, which runs the whole
    destination-segmented normalization as one CSR kernel: the per-channel null
    logit of ``softplus(z_bias) + eps`` goes in as ``null_logit`` and the
    envelope is folded into the logit inside the kernel.

``SO2Convolution.forward_envelope``
    The attention-free aggregation ``zeros -> index_add_(dst) -> mul_``
    (``sezm_nn/so2.py``). Replaced by one
    :func:`kernels.segment.segment_sum` call with ``edge_scale = envelope *
    source_gate`` and ``node_scale = inv_sqrt_deg``, so the zero-init, the
    atomic scatter and the degree scale collapse into one pass.

Both need the destination CSR view. DeepMD builds and caches it per step for its
own kernels (:func:`deepmd...edge_cache.cached_edge_csr`); the softmax seam only
receives ``dst``, so this module keeps the same view in a small LRU keyed by the
``dst`` storage, and reuses DeepMD's cache for the envelope path.

Everything else -- the gradient of the energy, the fused rotate-back
aggregation and the flash-attention path -- stays on DeepMD's own implementation;
see ``docs/operators.md`` for what is left and why.

Usage::

    from model.fused_deepmd import deepmd_patch, status, uninstall

    stats = deepmd_patch(deep_pot)     # prints the module tree first
    ...                                # run inference
    print(status(stats))
    uninstall(deep_pot)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Iterable

import torch
import torch.nn.functional as F

_CSR_CAPACITY = 4
# Tests drive the seams with CPU tensors and a stand-in operator bundle; the
# device guard below is what keeps a real install from calling the HIP kernels
# on a CPU tensor, so it stays on unless a test explicitly lifts it.
_ACCEPT_ANY_DEVICE = False


@dataclass
class FusedStats:
    """Hit, fallback and error counters for the installed seams."""

    softmax_hits: int = 0
    softmax_fallbacks: int = 0
    envelope_hits: int = 0
    envelope_fallbacks: int = 0
    force_hits: int = 0
    force_fallbacks: int = 0
    aggregate_hits: int = 0
    aggregate_fallbacks: int = 0
    pack_hits: int = 0
    pack_fallbacks: int = 0
    bound_softmax: bool = False
    bound_envelope: bool = False
    bound_force: bool = False
    bound_aggregate: bool = False
    bound_pack: bool = False
    blocks: int = 0
    last_error: str | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "softmax_hits": self.softmax_hits,
            "softmax_fallbacks": self.softmax_fallbacks,
            "envelope_hits": self.envelope_hits,
            "envelope_fallbacks": self.envelope_fallbacks,
            "force_hits": self.force_hits,
            "force_fallbacks": self.force_fallbacks,
            "aggregate_hits": self.aggregate_hits,
            "aggregate_fallbacks": self.aggregate_fallbacks,
            "pack_hits": self.pack_hits,
            "pack_fallbacks": self.pack_fallbacks,
            "bound_softmax": self.bound_softmax,
            "bound_envelope": self.bound_envelope,
            "bound_force": self.bound_force,
            "bound_aggregate": self.bound_aggregate,
            "bound_pack": self.bound_pack,
            "blocks": self.blocks,
            "last_error": self.last_error,
            "errors": dict(self.errors),
        }


def _usable_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and (value.is_cuda or _ACCEPT_ANY_DEVICE)
        and value.dtype == dtype
        and value.is_contiguous()
    )


def _load_slot_ops() -> Any:
    """Return the slot-axis operator module, or ``None`` when it cannot run here."""
    try:
        from kernels import slot as slot_ops
    except ImportError:
        return None
    return slot_ops if hasattr(slot_ops, "slot_pack_scale") else None


def _load_ops() -> Any:
    """Return the fused operator module, or ``None`` when it cannot run here."""
    try:
        from kernels import segment as segment_ops
    except ImportError:
        return None
    if not segment_ops.segment_available():
        return None
    return segment_ops


class _CsrView:
    """Destination CSR view (offsets, order) cached by the ``dst`` storage."""

    def __init__(self, capacity: int = _CSR_CAPACITY) -> None:
        self._capacity = capacity
        self._entries: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._order: list[tuple[int, int, int]] = []

    def get(self, dst: torch.Tensor, n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(dst.data_ptr()), int(dst.numel()), int(n_nodes))
        cached = self._entries.get(key)
        if cached is not None:
            return cached
        from kernels.segment import segment_offsets

        offsets = segment_offsets(dst, n_nodes)
        order = torch.argsort(dst, stable=True)
        self._entries[key] = (offsets, order)
        self._order.append(key)
        while len(self._order) > self._capacity:
            self._entries.pop(self._order.pop(0), None)
        return offsets, order


@dataclass
class _Session:
    ops: Any
    stats: FusedStats
    reference_softmax: Any
    reference_force: Any = None
    slot_ops: Any = None
    csr: _CsrView = field(default_factory=_CsrView)
    warned: bool = False

    # -- softmax seam -------------------------------------------------------
    def softmax(self, logits, edge_env, dst, n_nodes, z_bias_raw, eps, src_weight=None):
        stats = self.stats
        if src_weight is not None:
            stats.softmax_fallbacks += 1
            return self.reference_softmax(
                logits, edge_env, dst, n_nodes, z_bias_raw, eps, src_weight
            )
        if (
            not _usable_tensor(logits)
            or logits.ndim != 3
            or logits.shape[0] == 0
            or not _usable_tensor(edge_env)
            or not isinstance(dst, torch.Tensor)
            or dst.dtype != torch.int64
            or dst.ndim != 1
            or n_nodes <= 0
        ):
            stats.softmax_fallbacks += 1
            return self.reference_softmax(
                logits, edge_env, dst, n_nodes, z_bias_raw, eps, None
            )
        n_edges, n_focus, n_head = logits.shape
        n_channel = n_focus * n_head
        bias = z_bias_raw.to(dtype=torch.float32).reshape(-1)
        if bias.numel() != n_channel or edge_env.numel() != n_edges:
            stats.softmax_fallbacks += 1
            return self.reference_softmax(
                logits, edge_env, dst, n_nodes, z_bias_raw, eps, None
            )
        try:
            offsets, order = self.csr.get(dst, n_nodes)
            null_logit = torch.log(F.softplus(bias) + float(eps))
            null_logit = null_logit.reshape(1, n_channel).expand(n_nodes, n_channel)
            # ``segment_softmax`` returns ``(alpha, null_alpha)`` whenever a
            # null logit is supplied, which this seam always does; tolerate a
            # plain tensor as well so the seam does not depend on that arity.
            alpha = self.ops.segment_softmax(
                logits.reshape(n_edges, n_channel),
                edge_env.reshape(n_edges),
                offsets,
                null_logit.contiguous(),
                order=order,
            )
            if isinstance(alpha, tuple):
                alpha = alpha[0]
        except Exception as error:  # noqa: BLE001 - fall back, never break inference
            self._record(error, "softmax")
            stats.softmax_fallbacks += 1
            return self.reference_softmax(
                logits, edge_env, dst, n_nodes, z_bias_raw, eps, None
            )
        stats.softmax_hits += 1
        return alpha.reshape(n_edges, n_focus, n_head)

    # -- envelope seam ------------------------------------------------------
    def envelope(self, original, block, x, edge_cache, radial_feat):
        stats = self.stats
        dst = getattr(edge_cache, "dst", None)
        edge_env = getattr(edge_cache, "edge_env", None)
        inverse = getattr(edge_cache, "inv_sqrt_deg", None)
        gate = getattr(edge_cache, "edge_src_gate", None)
        if (
            bool(getattr(block, "training", False))
            or not _usable_tensor(x)
            or not _usable_tensor(edge_env)
            or not _usable_tensor(inverse)
            or not isinstance(dst, torch.Tensor)
            or dst.dtype != torch.int64
            or dst.ndim != 1
            or dst.numel() == 0
            or (gate is not None and not _usable_tensor(gate))
        ):
            stats.envelope_fallbacks += 1
            return original(x, edge_cache, radial_feat)
        n_node = x.shape[0]
        n_edges = dst.shape[0]
        if (
            x.ndim != 3
            or inverse.numel() != n_node
            or edge_env.numel() != n_edges
            or not _usable_tensor(dst, dtype=torch.int64)
        ):
            stats.envelope_fallbacks += 1
            return original(x, edge_cache, radial_feat)
        try:
            message, _ = block.edge_message(x, edge_cache, radial_feat)
            if not _usable_tensor(message) or message.shape[0] != n_edges:
                raise ValueError("edge message is not a contiguous fp32 (E, ...) tensor")
            if gate is not None:
                weight = edge_env.reshape(-1) * gate.reshape(-1).to(dtype=torch.float32)
            else:
                weight = edge_env.reshape(-1)
            offsets, order = self.csr.get(dst, n_node)
            out = self.ops.segment_sum(
                message.reshape(n_edges, -1),
                dst,
                offsets,
                weight.contiguous(),
                inverse.reshape(-1).contiguous(),
                order=order,
            )
        except Exception as error:  # noqa: BLE001 - fall back, never break inference
            self._record(error, "envelope")
            stats.envelope_fallbacks += 1
            return original(x, edge_cache, radial_feat)
        stats.envelope_hits += 1
        return out.reshape(x.shape)

    # -- gated activation seam (slot pack) -----------------------------------
    def gated_activation(self, block, x, gate, original):
        """Rebuild ``GatedActivation.forward`` for the ``fndc`` layout.

        The stock body expands the per-gate scalars with
        ``index_select(dim=2, expand_index)``, multiplies by the value rows and
        copies the product into a fresh ``out``. ``slot_pack_scale`` does the
        gather and the multiply in one pass; the copy into ``out`` stays because
        row zero carries the separately computed scalar activation.
        """
        stats = self.stats
        expand_index = getattr(block, "expand_index", None)
        if (
            bool(getattr(block, "training", False))
            or getattr(block, "layout", None) != "fndc"
            or int(getattr(block, "lmax", 0)) < 1
            or not _usable_tensor(x)
            or x.ndim != 4
            or not isinstance(expand_index, torch.Tensor)
            or expand_index.dtype != torch.int64
            or expand_index.numel() == 0
            or not callable(getattr(block, "gate_linear", None))
            or not callable(getattr(block, "scalar_act", None))
            or int(x.shape[2]) != int(expand_index.numel()) + 1
            or (gate is not None and not _usable_tensor(gate))
        ):
            stats.pack_fallbacks += 1
            return _call_forward(original, x, gate)
        n_focus, n_edge, rows, channels = x.shape
        try:
            degree_axis = 2
            source = gate if gate is not None else x
            gate_scalar_source = source.select(dim=degree_axis, index=0)
            if gate is not None:
                x0 = x.narrow(degree_axis, 0, 1) * block.scalar_act(
                    gate.narrow(degree_axis, 0, 1)
                )
            else:
                x0 = block.scalar_act(x.narrow(degree_axis, 0, 1))
            input_dtype = gate_scalar_source.dtype
            gate_src = gate_scalar_source.to(dtype=block.dtype)
            gate_logits = block.gate_linear(gate_src.transpose(0, 1)).transpose(0, 1)
            gating_scalars = torch.sigmoid(gate_logits).to(dtype=input_dtype)
            gating_scalars = gating_scalars.reshape(
                n_focus, gate_scalar_source.shape[1], int(block.lmax), channels
            )
            value_rows = x.narrow(degree_axis, 1, rows - 1)
            source_rows = gating_scalars.reshape(
                n_focus * n_edge, int(block.lmax), channels
            )
            scale_rows = value_rows.reshape(n_focus * n_edge, rows - 1, channels)
            # Narrowing along the degree axis leaves strides that reshape cannot
            # collapse, and the native op requires dense inputs.
            if not source_rows.is_contiguous():
                source_rows = source_rows.contiguous()
            if not scale_rows.is_contiguous():
                scale_rows = scale_rows.contiguous()
            packed = self.slot_ops.slot_pack_scale(
                source_rows, expand_index, scale_rows
            ).reshape(n_focus, n_edge, rows - 1, channels)
            out = x.new_empty(x.shape)
            out.narrow(degree_axis, 0, 1).copy_(x0)
            out.narrow(degree_axis, 1, rows - 1).copy_(packed)
        except Exception as error:  # noqa: BLE001 - fall back, never break inference
            self._record(error, "pack")
            stats.pack_fallbacks += 1
            return _call_forward(original, x, gate)
        stats.pack_hits += 1
        return out

    # -- dense attention aggregation seam ------------------------------------
    _BLOCK_ATTRIBUTES = (
        "ebed_dim_full",
        "attn_n_focus",
        "attn_focus_dim",
        "n_atten_head",
        "head_dim",
        "hidden_channels",
        "compute_dtype",
        "dtype",
    )

    def attention_dense(self, block, x, edge_cache, radial_feat, x_l0_node, original):
        """Rebuild ``forward_attention_dense`` with one segment reduction.

        The stock method materializes the per-head weighted value and then does
        ``zeros -> index_add_(dst) -> head gate``; the reduction is exactly
        :func:`kernels.segment.segment_sum` with the attention weight as the
        per-edge scale, so the zero init, the atomics and one full pass over the
        (E, D, Fa, H, Ch) tensor disappear. Everything else is delegated to the
        block's own building blocks, in the stock order.
        """
        stats = self.stats
        dst = getattr(edge_cache, "dst", None)
        reason = self._aggregate_reason(block, x, dst, x_l0_node)
        if reason is not None:
            self._note("aggregate", "guard: " + reason)
            stats.aggregate_fallbacks += 1
            return original(x, edge_cache, radial_feat, x_l0_node)
        n_node = int(x.shape[0])
        n_edge = int(dst.numel())
        try:
            message, rad_feat = block.edge_message(x, edge_cache, radial_feat)
            alpha = block.attention_weights(x_l0_node, edge_cache, rad_feat)
            head_gate = block.attention_head_gate(x_l0_node)
            value_focus = message.reshape(
                n_edge,
                block.ebed_dim_full,
                block.attn_n_focus,
                block.attn_focus_dim,
            ).to(dtype=block.compute_dtype)
            if getattr(block, "attn_v_proj", None) is not None:
                value_focus = block.attn_v_proj(value_focus)
            value_heads = value_focus.reshape(
                n_edge,
                block.ebed_dim_full,
                block.attn_n_focus,
                block.n_atten_head,
                block.head_dim,
            )
            if alpha.shape != (n_edge, block.attn_n_focus, block.n_atten_head):
                raise ValueError(f"unexpected attention weight shape {tuple(alpha.shape)}")
            weighted = value_heads * alpha.reshape(
                n_edge, 1, block.attn_n_focus, block.n_atten_head, 1
            )
            if not _usable_tensor(weighted, dtype=block.compute_dtype):
                raise ValueError("weighted value is not a contiguous compute_dtype tensor")
            offsets, order = self.csr.get(dst, n_node)
            reduced = self.ops.segment_sum(
                weighted.reshape(n_edge, -1), dst, offsets, None, None, order=order
            )
            out_heads = reduced.reshape(
                n_node,
                block.ebed_dim_full,
                block.attn_n_focus,
                block.n_atten_head,
                block.head_dim,
            )
            out_heads = out_heads * head_gate.reshape(
                n_node, 1, block.attn_n_focus, block.n_atten_head, 1
            )
            out_focus = out_heads.reshape(
                n_node, block.ebed_dim_full, block.attn_n_focus, block.attn_focus_dim
            )
            if getattr(block, "attn_o_proj", None) is not None:
                out_focus = block.attn_o_proj(out_focus)
            result = out_focus.reshape(
                n_node, block.ebed_dim_full, block.hidden_channels
            ).to(dtype=block.dtype)
        except Exception as error:  # noqa: BLE001 - fall back, never break inference
            self._record(error, "aggregate")
            stats.aggregate_fallbacks += 1
            return original(x, edge_cache, radial_feat, x_l0_node)
        stats.aggregate_hits += 1
        return result

    # -- force / virial seam ------------------------------------------------
    @staticmethod
    def _segments_from_csr(offsets: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        """Recover the natural-order segment ids from a CSR view."""
        counts = (offsets[1:] - offsets[:-1]).tolist()
        grouped = torch.repeat_interleave(
            torch.arange(offsets.numel() - 1, dtype=torch.int64, device=order.device),
            torch.tensor(counts, dtype=torch.int64, device=order.device),
        )
        return torch.empty_like(grouped).scatter_(0, order, grouped)

    def force(self, g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr):
        """Assemble force and per-atom virial for DeepMD's ``edge_force_virial``.

        DeepMD hands over the two CSR permutations it just sorted; the dense
        fallback it would otherwise use is four atomic scatters plus an
        ``(E, 9)`` outer product.
        """
        stats = self.stats
        edges = int(g.shape[0]) if isinstance(g, torch.Tensor) else -1
        reason = self._force_reason(
            g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr
        )
        if reason is not None:
            self._note("force", "guard: " + reason)
            stats.force_fallbacks += 1
            return self._force_fallback(
                g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr
            )
        try:
            force, virial = self.ops.force_assembly(
                g.reshape(edges, 3).contiguous(),
                edge_vec.reshape(edges, 3).contiguous(),
                None,
                None,
                dst_row_ptr,
                src_row_ptr,
                src_order,
                dst_order=dst_order,
            )
        except Exception as error:  # noqa: BLE001 - fall back, never break inference
            self._record(error, "force")
            stats.force_fallbacks += 1
            return self._force_fallback(
                g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr
            )
        stats.force_hits += 1
        return force, virial

    def _force_fallback(self, g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr):
        """The dense reference: two atomic scatters plus the materialised outer product."""
        g = g.reshape(-1, 3)
        edge_vec = edge_vec.reshape(-1, 3)
        dst_ext = self._segments_from_csr(dst_row_ptr, dst_order)
        src_ext = self._segments_from_csr(src_row_ptr, src_order)
        n_ext = int(dst_row_ptr.numel()) - 1
        force = g.new_zeros(n_ext, 3)
        force.index_add_(0, dst_ext, g)
        force.index_add_(0, src_ext, -g)
        w_edge = -torch.einsum("ek,ej->ekj", g, edge_vec).reshape(-1, 9)
        virial = g.new_zeros(n_ext, 9)
        virial.index_add_(0, src_ext, w_edge)
        return force, virial

    def _note(self, seam: str, message: str) -> None:
        """Record why a seam's guard rejected a call."""
        self.stats.errors[seam] = message
        self.stats.last_error = f"{seam}: {message}"

    @staticmethod
    def _force_reason(g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr):
        if not isinstance(g, torch.Tensor):
            return "g is not a tensor"
        edges = int(g.shape[0])
        if not _usable_tensor(g):
            return "g is not a contiguous fp32 CUDA tensor"
        if not _usable_tensor(edge_vec):
            return "edge_vec is not a contiguous fp32 CUDA tensor"
        if g.shape != edge_vec.shape or g.shape[-1] != 3:
            return (f"g/edge_vec are not both (E, 3): {tuple(g.shape)} vs "
                    f"{tuple(edge_vec.shape)}")
        if not _usable_tensor(dst_row_ptr, dtype=torch.int64):
            return "dst_row_ptr is not a contiguous int64 CUDA tensor"
        if not _usable_tensor(src_row_ptr, dtype=torch.int64):
            return "src_row_ptr is not a contiguous int64 CUDA tensor"
        if dst_row_ptr.numel() != src_row_ptr.numel():
            return (f"the offsets describe different atom counts: "
                    f"{dst_row_ptr.numel()} vs {src_row_ptr.numel()}")
        for name, order in (("dst_order", dst_order), ("src_order", src_order)):
            if not _usable_tensor(order, dtype=torch.int64):
                return f"{name} is not a contiguous int64 CUDA tensor"
            if order.numel() != edges:
                return f"{name} has {order.numel()} entries for {edges} edges"
        return None

    def _aggregate_reason(self, block, x, dst, x_l0_node):
        if not _usable_tensor(x):
            return "x is not a contiguous fp32 CUDA tensor"
        if (not isinstance(dst, torch.Tensor) or dst.dtype != torch.int64
                or dst.ndim != 1 or dst.numel() == 0):
            return "edge_cache.dst is not a non-empty int64 vector"
        if not _usable_tensor(x_l0_node):
            return "x_l0_node is not a contiguous fp32 CUDA tensor"
        missing = [name for name in self._BLOCK_ATTRIBUTES
                   if not hasattr(block, name)]
        if missing:
            return "block is missing " + ", ".join(missing)
        return None

    def _record(self, error: Exception, seam: str = "") -> None:
        message = f"{type(error).__name__}: {error}"
        if seam:
            self.stats.errors[seam] = message
            message = f"{seam}: {message}"
        self.stats.last_error = message
        if not self.warned:
            self.warned = True
            warnings.warn(
                "fused DeepMD dispatch fell back to the reference path: "
                + self.stats.last_error,
                RuntimeWarning,
                stacklevel=3,
            )


_SESSION: _Session | None = None
_SAVED: dict[str, Any] = {}


def _force_targets():
    """DeepMD's force/virial seam: the module that gates it and the callable.

    ``transform_output.edge_energy_deriv`` takes the fused branch only when
    ``triton_infer_level() >= 1`` (or the CUDA/cuTile equivalents), and then
    imports ``edge_force_assembly`` from the Triton module. Both are rebound here
    so the branch runs this project's operator instead.
    """
    try:
        import deepmd.pt.model.model.transform_output as transform_module
        import deepmd.pt_expt.kernels.triton.sezm.force_assembly as assembly_module
    except ImportError:
        return None
    if not callable(getattr(assembly_module, "edge_force_assembly", None)):
        return None
    return transform_module, assembly_module


def _softmax_namespaces() -> list[Any]:
    """Modules whose ``segment_envelope_gated_softmax`` name must be rebound."""
    import deepmd.pt.model.descriptor.sezm_nn.attention as attention_module
    import deepmd.pt.model.descriptor.sezm_nn.so2 as so2_module

    namespaces = [attention_module, so2_module]
    try:
        import deepmd.pt.model.descriptor.sezm_nn as package

        namespaces.append(package)
    except ImportError:  # pragma: no cover - package always exists with the modules
        pass
    return namespaces


def _blocks(model: Any) -> list[Any]:
    """SO(2) convolution blocks that carry the aggregation seams."""
    return [
        module
        for module in model.modules()
        if callable(getattr(module, "edge_message", None))
        and callable(getattr(module, "forward_envelope", None))
        and hasattr(module, "_segment_softmax_fn")
    ]


def _activations(root: Any) -> list[Any]:
    """``GatedActivation`` modules, and only those.

    ``expand_index`` alone is not a sufficient marker: ``SO3Linear`` and the
    norm layers register one too, and rebinding their one-argument ``forward``
    with the gated-activation shim crashes inference as soon as the shim falls
    back, because the shim would call ``original(x, None)``.
    """
    if root is None:
        return []
    return [
        module
        for module in root.modules()
        if isinstance(getattr(module, "expand_index", None), torch.Tensor)
        and getattr(module, "layout", None) in {"nfdc", "ndfc", "fndc"}
        and callable(getattr(module, "gate_linear", None))
        and callable(getattr(module, "scalar_act", None))
    ]


def _call_forward(original: Any, x: Any, gate: Any) -> Any:
    """Call the stock forward with the arity it declares."""
    return original(x) if gate is None else original(x, gate)


def _module_root(model: Any) -> tuple[Any | None, str]:
    """Resolve the ``nn.Module`` tree behind a ``DeepPot``/``DeepEval`` wrapper.

    The public inference object is not an ``nn.Module``: the legacy ``pt``
    backend keeps its torch module at ``deep_eval.dp`` (a ``ModelWrapper``) and
    the ``pt_expt`` backend keeps the deserialized graph at
    ``deep_eval._dpmodel``. A caller may also hand in a module directly.
    """
    if callable(getattr(model, "modules", None)):
        return model, "model"
    deep_eval = getattr(model, "deep_eval", None)
    if deep_eval is None:
        return None, ""
    for attribute in ("_dpmodel", "dp", "model"):
        candidate = getattr(deep_eval, attribute, None)
        if candidate is not None and callable(getattr(candidate, "modules", None)):
            return candidate, f"deep_eval.{attribute}"
    if callable(getattr(deep_eval, "modules", None)):
        return deep_eval, "deep_eval"
    return None, ""


def describe_module(module: Any, *, max_depth: int = 2, max_rows: int = 40) -> list[str]:
    """Render the module tree and the patch targets as printable lines.

    The tree stops at ``max_depth`` so a real descriptor does not flood the
    console; the patch targets are listed in full because those are the rows
    the caller is about to replace.
    """
    lines: list[str] = []
    root, path = _module_root(module)
    if root is None:
        lines.append(
            f"module: {type(module).__name__} (no nn.Module tree found -- "
            "expected a module, or a DeepPot/DeepEval wrapper with "
            "deep_eval.dp / deep_eval._dpmodel)"
        )
        lines.append("patch targets: 0 (nothing can be bound)")
        return lines
    named = list(root.named_modules()) if hasattr(root, "named_modules") else []
    origin = "" if path == "model" else f" -> {path}"
    lines.append(
        f"module: {type(module).__name__}{origin} "
        f"({type(root).__name__}, {len(named)} submodules)"
    )
    for name, child in named[:max_rows]:
        depth = 0 if not name else name.count(".") + 1
        if depth > max_depth:
            continue
        lines.append(f"  {'  ' * depth}{name or '<root>'}  {type(child).__name__}")
    if len(named) > max_rows:
        lines.append(f"  ... {len(named) - max_rows} more submodules")
    targets = _blocks(root)
    lines.append(f"patch targets: {len(targets)} SO(2) block(s)")
    for index, block in enumerate(targets):
        lines.append(
            f"  [{index}] {type(block).__name__}  "
            f"n_atten_head={getattr(block, 'n_atten_head', '?')}  "
            f"seams: forward_envelope -> segment_sum, "
            f"attention softmax -> segment_softmax"
        )
    return lines


def deepmd_patch(module: Any, *, ops: Any = None, softmax: bool = True,
                 envelope: bool = True, force: bool = True,
                 aggregate: bool = True, pack: bool = True, slot_ops: Any = None,
                 blocks: Iterable[Any] | None = None,
                 namespaces: Iterable[Any] | None = None,
                 force_targets: Any = None,
                 verbose: bool = True) -> FusedStats:
    """Bind the fused operators on ``model`` (a ``DeepPot`` or an ``nn.Module``).

    ``ops`` may be an operator bundle exposing ``segment_sum`` and
    ``segment_softmax``; it defaults to :mod:`kernels.segment` and the install
    is a no-op when that extension cannot run on this host. ``namespaces``
    overrides the DeepMD modules that carry the softmax name (tests use it to
    avoid importing DeepMD). Returns the counters that :func:`status` formats.
    """
    global _SESSION
    stats = FusedStats()
    if verbose:
        for line in describe_module(module):
            print(f"[deepmd_patch] {line}", flush=True)
    uninstall(module)
    root, path = _module_root(module)
    if blocks is not None:
        module_list = list(blocks)
    elif root is None:
        module_list = []
    else:
        module_list = list(_blocks(root))
    if verbose and root is not None and path != "model":
        print(f"[deepmd_patch] resolved module tree: {path} "
              f"({type(root).__name__})", flush=True)
    stats.blocks = len(module_list)
    if ops is None:
        ops = _load_ops()
    if verbose:
        source = "kernels.segment" if ops is not None else "unavailable"
        print(
            f"[deepmd_patch] operators: {source}; "
            f"softmax={softmax} envelope={envelope}",
            flush=True,
        )
    if ops is None:
        if verbose:
            print("[deepmd_patch] no fused extension on this host -> nothing replaced", flush=True)
        return stats

    namespace_list: list[Any] = []
    reference = None
    if softmax:
        namespace_list = list(namespaces) if namespaces is not None else _softmax_namespaces()
        if not namespace_list:
            raise ValueError("deepmd_patch requires one module namespace to patch")
        reference = namespace_list[0].segment_envelope_gated_softmax
    _SESSION = _Session(
        ops=ops,
        stats=stats,
        reference_softmax=reference,
        slot_ops=slot_ops if slot_ops is not None else _load_slot_ops(),
    )

    if softmax:
        def bound_softmax(logits, edge_env, dst, n_nodes, z_bias_raw, eps, src_weight=None):
            session = _SESSION
            if session is None:  # uninstalled between calls
                return reference(logits, edge_env, dst, n_nodes, z_bias_raw, eps, src_weight)
            return session.softmax(logits, edge_env, dst, n_nodes, z_bias_raw, eps, src_weight)

        for namespace in namespace_list:
            name = "segment_envelope_gated_softmax"
            if hasattr(namespace, name):
                _SAVED[f"{namespace.__name__}.{name}"] = (namespace, getattr(namespace, name))
                setattr(namespace, name, bound_softmax)
                if verbose:
                    print(
                        f"[deepmd_patch] replace {namespace.__name__}.{name} "
                        f"-> model.fused_deepmd.bound_softmax",
                        flush=True,
                    )
        stats.bound_softmax = True

    if envelope:
        for block in module_list:
            original = block.forward_envelope
            _SAVED[f"block:{id(block)}"] = (block, original)

            def patched(self, x, edge_cache, radial_feat, _original=original):
                session = _SESSION
                if session is None:
                    return _original(x, edge_cache, radial_feat)
                return session.envelope(_original, self, x, edge_cache, radial_feat)

            block.forward_envelope = MethodType(patched, block)
        stats.bound_envelope = bool(module_list)
        if verbose:
            print(
                f"[deepmd_patch] replace SO2Convolution.forward_envelope on "
                f"{len(module_list)} block(s) -> model.fused_deepmd.patched",
                flush=True,
            )
    if force:
        targets = force_targets if force_targets is not None else _force_targets()
        if targets is None:
            if verbose:
                print(
                    "[deepmd_patch] force seam unavailable (no DeepMD force_assembly "
                    "module on this host) -> left on DeepMD's path",
                    flush=True,
                )
        else:
            transform_module, assembly_module = targets
            cutile = getattr(transform_module, "use_cutile_infer", None)
            if callable(cutile) and cutile():
                if verbose:
                    print(
                        "[deepmd_patch] cuTile force path is enabled -> force seam "
                        "skipped",
                        flush=True,
                    )
            else:
                original_level = transform_module.triton_infer_level
                original_assembly = assembly_module.edge_force_assembly

                def bound_force(g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr):
                    session = _SESSION
                    if session is None:
                        raise RuntimeError(
                            "the fused force seam was called after uninstall(); "
                            "call deepmd_patch() again or drop fused=('force',)"
                        )
                    return session.force(
                        g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr
                    )

                _SAVED[f"{transform_module.__name__}.triton_infer_level"] = (
                    transform_module,
                    original_level,
                )
                _SAVED[f"{assembly_module.__name__}.edge_force_assembly"] = (
                    assembly_module,
                    original_assembly,
                )
                transform_module.triton_infer_level = lambda: max(int(original_level()), 1)
                assembly_module.edge_force_assembly = bound_force
                stats.bound_force = True
                if verbose:
                    print(
                        "[deepmd_patch] replace "
                        f"{assembly_module.__name__}.edge_force_assembly "
                        "-> model.fused_deepmd.bound_force "
                        "(and force triton_infer_level() >= 1)",
                        flush=True,
                    )

    if pack and _SESSION.slot_ops is not None:
        activations = _activations(root)
        patched = 0
        for activation in activations:
            original_forward = getattr(activation, "forward", None)
            if not callable(original_forward):
                continue
            _SAVED[f"activation:{id(activation)}"] = (activation, original_forward)

            def patched_forward(self, x, gate=None, _original=original_forward):
                session = _SESSION
                if session is None:
                    return _original(x, gate)
                return session.gated_activation(self, x, gate, _original)

            activation.forward = MethodType(patched_forward, activation)
            patched += 1
        stats.bound_pack = bool(patched)
        if verbose and patched:
            print(
                f"[deepmd_patch] replace GatedActivation.forward on {patched} "
                "module(s) -> slot_pack_scale",
                flush=True,
            )

    if aggregate:
        patched = 0
        for block in module_list:
            original_dense = getattr(block, "forward_attention_dense", None)
            if not callable(original_dense):
                continue
            _SAVED[f"block-dense:{id(block)}"] = (block, original_dense)

            def patched_dense(
                self, x, edge_cache, radial_feat, x_l0_node, _original=original_dense
            ):
                session = _SESSION
                if session is None:
                    return _original(x, edge_cache, radial_feat, x_l0_node)
                return session.attention_dense(
                    self, x, edge_cache, radial_feat, x_l0_node, _original
                )

            block.forward_attention_dense = MethodType(patched_dense, block)
            patched += 1
        stats.bound_aggregate = bool(patched)
        if verbose and patched:
            print(
                f"[deepmd_patch] replace SO2Convolution.forward_attention_dense on "
                f"{patched} block(s) -> segment_sum",
                flush=True,
            )

    if verbose:
        print(
            f"[deepmd_patch] installed: softmax={stats.bound_softmax} "
            f"envelope={stats.bound_envelope} force={stats.bound_force} "
            f"aggregate={stats.bound_aggregate} pack={stats.bound_pack} "
            f"blocks={stats.blocks}",
            flush=True,
        )
    return stats


def uninstall(module: Any = None) -> None:
    """Restore every rebound name; safe to call when nothing is installed."""
    global _SESSION
    for key, (owner, original) in list(_SAVED.items()):
        if key.startswith("block:"):
            # Drop the instance attribute so the class method shows through
            # again, exactly as before the patch.
            if "forward_envelope" in getattr(owner, "__dict__", {}):
                del owner.__dict__["forward_envelope"]
        elif key.startswith("block-dense:"):
            if "forward_attention_dense" in getattr(owner, "__dict__", {}):
                del owner.__dict__["forward_attention_dense"]
        elif key.startswith("activation:"):
            if "forward" in getattr(owner, "__dict__", {}):
                del owner.__dict__["forward"]
        else:
            setattr(owner, key.rsplit(".", 1)[-1], original)
    _SAVED.clear()
    _SESSION = None


def status(stats: FusedStats | None) -> dict[str, Any]:
    """Report the counters of a patch (``None`` when it never ran)."""
    return {} if stats is None else stats.as_dict()


__all__ = ["FusedStats", "deepmd_patch", "describe_module", "status", "uninstall"]
