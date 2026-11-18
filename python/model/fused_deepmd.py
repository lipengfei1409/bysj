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
see ``docs/deepmd-fused-dispatch.md`` for what is left and why.

Usage::

    from model.fused_deepmd import install, uninstall, status

    stats = install(deep_pot)          # or install(deep_pot, softmax=False)
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
    bound_softmax: bool = False
    bound_envelope: bool = False
    blocks: int = 0
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "softmax_hits": self.softmax_hits,
            "softmax_fallbacks": self.softmax_fallbacks,
            "envelope_hits": self.envelope_hits,
            "envelope_fallbacks": self.envelope_fallbacks,
            "bound_softmax": self.bound_softmax,
            "bound_envelope": self.bound_envelope,
            "blocks": self.blocks,
            "last_error": self.last_error,
        }


def _usable_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and (value.is_cuda or _ACCEPT_ANY_DEVICE)
        and value.dtype == dtype
        and value.is_contiguous()
    )


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
            alpha = self.ops.segment_softmax(
                logits.reshape(n_edges, n_channel),
                edge_env.reshape(n_edges),
                offsets,
                null_logit.contiguous(),
                order=order,
            )
        except Exception as error:  # noqa: BLE001 - fall back, never break inference
            self._record(error)
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
            return original(block, x, edge_cache, radial_feat)
        n_node = x.shape[0]
        n_edges = dst.shape[0]
        if (
            x.ndim != 3
            or inverse.numel() != n_node
            or edge_env.numel() != n_edges
            or not _usable_tensor(dst, dtype=torch.int64)
        ):
            stats.envelope_fallbacks += 1
            return original(block, x, edge_cache, radial_feat)
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
            self._record(error)
            stats.envelope_fallbacks += 1
            return original(block, x, edge_cache, radial_feat)
        stats.envelope_hits += 1
        return out.reshape(x.shape)

    def _record(self, error: Exception) -> None:
        self.stats.last_error = f"{type(error).__name__}: {error}"
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


def install(model: Any, *, ops: Any = None, softmax: bool = True,
            envelope: bool = True, blocks: Iterable[Any] | None = None,
            namespaces: Iterable[Any] | None = None) -> FusedStats:
    """Bind the fused operators on ``model`` (a ``DeepPot`` or an ``nn.Module``).

    ``ops`` may be an operator bundle exposing ``segment_sum`` and
    ``segment_softmax``; it defaults to :mod:`kernels.segment` and the install
    is a no-op when that extension cannot run on this host. ``namespaces``
    overrides the DeepMD modules that carry the softmax name (tests use it to
    avoid importing DeepMD). Returns the counters that :func:`status` formats.
    """
    global _SESSION
    stats = FusedStats()
    uninstall(model)
    module_list = list(blocks) if blocks is not None else list(_blocks(model))
    stats.blocks = len(module_list)
    if ops is None:
        ops = _load_ops()
    if ops is None:
        return stats

    namespace_list: list[Any] = []
    reference = None
    if softmax:
        namespace_list = list(namespaces) if namespaces is not None else _softmax_namespaces()
        if not namespace_list:
            raise ValueError("install requires at least one module namespace to patch")
        reference = namespace_list[0].segment_envelope_gated_softmax
    _SESSION = _Session(ops=ops, stats=stats, reference_softmax=reference)

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
        stats.bound_softmax = True

    if envelope:
        for block in module_list:
            original = block.forward_envelope
            _SAVED[f"block:{id(block)}"] = (block, original)

            def patched(self, x, edge_cache, radial_feat, _original=original):
                session = _SESSION
                if session is None:
                    return _original(self, x, edge_cache, radial_feat)
                return session.envelope(_original, self, x, edge_cache, radial_feat)

            block.forward_envelope = MethodType(patched, block)
        stats.bound_envelope = bool(module_list)
    return stats


def uninstall(model: Any = None) -> None:
    """Restore every rebound name; safe to call when nothing is installed."""
    global _SESSION
    for key, (owner, original) in list(_SAVED.items()):
        if key.startswith("block:"):
            # Drop the instance attribute so the class method shows through
            # again, exactly as before the patch.
            if "forward_envelope" in getattr(owner, "__dict__", {}):
                del owner.__dict__["forward_envelope"]
        else:
            setattr(owner, key.rsplit(".", 1)[-1], original)
    _SAVED.clear()
    _SESSION = None


def status(stats: FusedStats | None) -> dict[str, Any]:
    """Report the counters of an install (``None`` when it never ran)."""
    return {} if stats is None else stats.as_dict()


__all__ = ["FusedStats", "install", "status", "uninstall"]
