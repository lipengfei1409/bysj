"""DeepMD-backed inference adapter using native GIE and RMSNorm by default.

This module delegates DPA4 architecture, checkpoint loading, neighbor-list
handling, and force/virial evaluation to DeepMD-kit. It patches matching GIE
and RMSNorm modules per-instance with this project's HIP/DCU operators, and can
additionally bind the fused segmented operators (``fused=``). TileLang dispatch
was removed from this project.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterable, Optional

_NATIVE_OPS = frozenset(("gie", "rmsnorm"))


def _normalize_native_ops(native_ops: Optional[Iterable[str]]) -> tuple[str, ...]:
    """Normalize the explicitly selected native DPA4 operator names."""
    if native_ops is None:
        return ("gie", "rmsnorm")
    values = (native_ops,) if isinstance(native_ops, str) else native_ops
    selected = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("native_ops entries must be strings")
        name = value.strip().lower()
        if name not in _NATIVE_OPS:
            raise ValueError(
                f"unknown native DPA4 operator {value!r}; choose from: "
                f"{', '.join(sorted(_NATIVE_OPS))}"
            )
        if name not in selected:
            selected.append(name)
    return tuple(selected)


_FUSED_SEAMS = ("softmax", "envelope")


def _normalize_fused(fused: Any) -> dict[str, bool] | None:
    """Translate the ``fused`` switch into the per-seam switches."""
    if not fused:
        return None
    if fused is True:
        return {name: True for name in _FUSED_SEAMS}
    if isinstance(fused, str):
        fused = (fused,)
    if isinstance(fused, Iterable) and not isinstance(fused, dict):
        selected = {}
        for value in fused:
            if not isinstance(value, str) or value.strip().lower() not in _FUSED_SEAMS:
                raise ValueError(
                    f"unknown fused seam {value!r}; choose from: "
                    f"{', '.join(_FUSED_SEAMS)}"
                )
            selected[value.strip().lower()] = True
        return selected
    if isinstance(fused, dict):
        unknown = set(fused) - set(_FUSED_SEAMS)
        if unknown:
            raise ValueError(
                f"unknown fused seams: {', '.join(sorted(map(str, unknown)))}; "
                f"choose from: {', '.join(_FUSED_SEAMS)}"
            )
        return {name: bool(fused.get(name, False)) for name in _FUSED_SEAMS}
    raise TypeError("fused must be a bool, a seam name, a sequence of names or a mapping")


def _create_deep_pot(model_file: str, **kwargs: Any) -> Any:
    """Load DeepMD lazily so importing this adapter has no DeepMD dependency."""
    try:
        from deepmd.infer import DeepPot
    except ImportError as error:
        raise RuntimeError(
            "DPA4Model requires an installed DeepMD-kit with the PyTorch "
            "inference backend."
        ) from error
    return DeepPot(model_file, **kwargs)


def _backend_name(deep_pot: Any) -> str:
    backend = getattr(deep_pot, "deep_eval", None)
    module_name = type(backend).__module__
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "deepmd":
        return parts[1]
    return "unknown"


def _load_native_operator(
    name: str,
) -> tuple[Callable[[], bool], Callable[..., Any]]:
    """Load one project's native wrapper lazily, only when selected."""
    if name == "gie":
        from kernels.gie import gie_available, zonal_scatter

        return gie_available, zonal_scatter
    if name == "rmsnorm":
        from kernels.rmsnorm import rmsnorm, rmsnorm_available

        return rmsnorm_available, rmsnorm
    raise ValueError(f"unknown native DPA4 operator {name!r}")


def _native_device_tensor(value: Any) -> bool:
    """Whether ``value`` is a tensor on the project's HIP device."""
    try:
        import torch
    except ImportError:
        return False
    return (
        isinstance(value, torch.Tensor)
        and value.device.type == "cuda"
        and torch.version.hip is not None
    )


def _native_fp32_tensor(value: Any) -> bool:
    """Whether ``value`` is an FP32 tensor on the project's HIP device."""
    try:
        import torch
    except ImportError:
        return False
    return _native_device_tensor(value) and value.dtype == torch.float32


def _native_dcu_tensors(*values: Any) -> bool:
    """Check that the tensors share a supported DCU, not just a CUDA device."""
    if not all(_native_device_tensor(value) for value in values):
        return False
    try:
        from kernels.native import require_dcu

        require_dcu(*values)
    except (ImportError, OSError, RuntimeError):
        return False
    return True


def _is_plain_deepmd_module(module: Any, *, class_name: str, module_name: str) -> bool:
    """Match the exact dpmodel base in the pt_expt module's MRO."""
    return any(
        cls.__name__ == class_name and cls.__module__ == module_name
        for cls in type(module).__mro__
    )


def _is_dpa4_gie(module: Any) -> bool:
    return (
        _is_plain_deepmd_module(
            module,
            class_name="GeometricInitialEmbedding",
            module_name="deepmd.dpmodel.descriptor.dpa4_nn.embedding",
        )
        and callable(getattr(module, "call", None))
        and hasattr(module, "radial_slot_index_for_row")
        and hasattr(module, "lmax")
    )


def _is_plain_rmsnorm(module: Any) -> bool:
    return (
        _is_plain_deepmd_module(
            module,
            class_name="RMSNorm",
            module_name="deepmd.dpmodel.descriptor.dpa4_nn.norm",
        )
        and callable(getattr(module, "call", None))
        and hasattr(module, "adam_scale")
        and hasattr(module, "eps")
        and hasattr(module, "precision")
    )


def _is_fp32_compute_precision(module: Any) -> bool:
    """Check DeepMD's configured compute dtype without hard-coding aliases."""
    try:
        import torch
        from deepmd.dpmodel.common import get_xp_precision

        return get_xp_precision(torch, module.precision) is torch.float32
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        return False


def _make_gie_call(module: Any, native_scatter: Callable[..., Any]) -> Any:
    original_call = module.call

    def patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        # DPA4's descriptor supplies these as keywords. Preserve all other call
        # forms, including inputs this primitive cannot represent.
        if args or not all(
            name in kwargs
            for name in ("edge_cache", "radial_feat", "zonal_coupling")
        ):
            return original_call(*args, **kwargs)

        edge_cache = kwargs["edge_cache"]
        radial = kwargs["radial_feat"]
        zonal = kwargs["zonal_coupling"]
        spin_l1_message = kwargs.get("spin_l1_message")
        inverse = getattr(edge_cache, "inv_sqrt_deg", None)
        dst = getattr(edge_cache, "dst", None)
        if (
            self.lmax <= 0
            or bool(getattr(self, "training", False))
            or spin_l1_message is not None
            or getattr(edge_cache, "edge_src_gate", None) is not None
            or getattr(edge_cache, "edge_mask", None) is not None
            or not _native_fp32_tensor(zonal)
            or not _native_fp32_tensor(radial)
            or not _native_fp32_tensor(inverse)
            or not _native_device_tensor(dst)
            or not _native_dcu_tensors(zonal, radial, dst, inverse)
        ):
            return original_call(*args, **kwargs)

        import torch

        radial_slot = torch.as_tensor(
            self.radial_slot_index_for_row,
            device=zonal.device,
            dtype=torch.int64,
        ).contiguous()
        n_nodes = kwargs.get("n_nodes")
        if (
            dst.dtype != torch.int64
            or zonal.ndim != 2
            or radial.ndim != 3
            or radial.shape[0] != zonal.shape[0]
            or radial.shape[2] != self.channels
            or (zonal.shape[1] > 0 and radial.shape[1] == 0)
            or dst.shape != (zonal.shape[0],)
            or radial_slot.shape != (zonal.shape[1],)
            or inverse.ndim not in (1, 3)
            or (inverse.ndim == 3 and inverse.shape[1:] != (1, 1))
            or n_nodes is None
            or inverse.shape[0] != n_nodes
            or zonal.device != radial.device
            or zonal.device != inverse.device
            or zonal.device != dst.device
        ):
            return original_call(*args, **kwargs)

        return native_scatter(
            zonal.contiguous(),
            radial.contiguous(),
            dst.contiguous(),
            radial_slot,
            inverse.reshape(-1, 1, 1).contiguous(),
        )

    return MethodType(patched_call, module)


def _make_rmsnorm_call(module: Any, native_rmsnorm: Callable[..., Any]) -> Any:
    original_call = module.call

    def patched_call(self: Any, x: Any) -> Any:
        scale = getattr(self, "adam_scale", None)
        if (
            not _native_fp32_tensor(x)
            or x.ndim < 1
            or x.shape[-1] <= 0
            or not x.is_contiguous()
            or not _is_fp32_compute_precision(self)
            or not _native_fp32_tensor(scale)
            or not _native_dcu_tensors(x, scale)
            or not scale.is_contiguous()
            or scale.device != x.device
            or scale.shape != (x.shape[-1],)
        ):
            return original_call(x)
        return native_rmsnorm(x, scale, eps=float(self.eps))

    return MethodType(patched_call, module)


def _install_native_ops(deep_pot: Any, native_ops: tuple[str, ...]) -> dict[str, int]:
    """Patch selected operators on one eager pt_expt model instance only."""
    if not native_ops:
        return {}
    if _backend_name(deep_pot) != "pt_expt":
        raise RuntimeError(
            "native DPA4 operators require DeepMD's `pt_expt` backend; "
            f"loaded backend is {_backend_name(deep_pot)!r}"
        )

    deep_eval = getattr(deep_pot, "deep_eval", None)
    model = getattr(deep_eval, "_dpmodel", None)
    modules_method = getattr(model, "modules", None)
    if not callable(modules_method):
        raise RuntimeError(
            "native DPA4 operators require an eager pt_expt model exposing "
            "deep_eval._dpmodel; use a `.pt` checkpoint"
        )
    modules = list(modules_method())
    targets: dict[str, list[Any]] = {
        "gie": [module for module in modules if _is_dpa4_gie(module)],
        "rmsnorm": [module for module in modules if _is_plain_rmsnorm(module)],
    }
    for name in native_ops:
        if not targets[name]:
            raise RuntimeError(
                f"native DPA4 operator {name!r} was requested, but no supported "
                "module instance was found in the loaded model"
            )
    implementations: dict[str, Callable[..., Any]] = {}
    for name in native_ops:
        try:
            available, implementation = _load_native_operator(name)
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError(
                f"native DPA4 operator {name!r} could not be loaded; build the "
                "project HIP extension on a supported DCU host"
            ) from error
        if not available():
            raise RuntimeError(
                f"native DPA4 operator {name!r} is unavailable; build the "
                "project HIP extension and use a supported DCU"
            )
        implementations[name] = implementation

    patches: list[tuple[Any, str, Any]] = []
    counts: dict[str, int] = {}
    if "gie" in native_ops:
        patches.extend(
            (module, "call", _make_gie_call(module, implementations["gie"]))
            for module in targets["gie"]
        )
        counts["gie"] = len(targets["gie"])
    if "rmsnorm" in native_ops:
        patches.extend(
            (module, "call", _make_rmsnorm_call(module, implementations["rmsnorm"]))
            for module in targets["rmsnorm"]
        )
        counts["rmsnorm"] = len(targets["rmsnorm"])
    originals = [
        (module, method_name, getattr(module, method_name))
        for module, method_name, _ in patches
    ]
    try:
        for module, method_name, patched_method in patches:
            setattr(module, method_name, patched_method)
    except Exception:
        for module, method_name, original_method in originals:
            setattr(module, method_name, original_method)
        raise
    return counts


class DPA4Model:
    """DeepMD DPA4/SeZM facade with native inference operator patches.

    DeepMD-kit owns model loading, neighbor-list handling, energy, force, and
    virial evaluation. By default, the matching GIE and plain RMSNorm modules
    in an eager ``pt_expt`` ``.pt`` model are patched per instance to use this
    project's HIP/DCU operators. Pass ``native_ops=()`` to run the DeepMD
    baseline, or select one native operator explicitly. Unsupported individual
    calls continue through the original DeepMD implementation. The destination
    reduction inside attention-free SO(2) convolution is available through the
    fused ``envelope`` seam (``fused=("envelope",)``), not through this
    selector.

    TileLang dispatch was removed from this project: the legacy ``tilelang_ops``
    keyword now raises instead of silently doing nothing, so an old call site
    fails loudly rather than appearing to select an operator that no longer
    exists. The adapter does not mutate environment variables.

    Parameters
    ----------
    model_file
        A model artifact supported by ``deepmd.infer.DeepPot``. Native patches
        require an eager ``pt_expt`` checkpoint ending in ``.pt``.
    fused
        Bind this project's fused segmented HIP operators into the loaded model;
        see :mod:`model.fused_deepmd`. ``True`` installs every seam, a string or
        sequence selects seams, a mapping sets each one, ``None``/``False``
        leaves DeepMD untouched.
    native_ops
        Project HIP/DCU operators to patch. Defaults to ``("gie", "rmsnorm")``;
        pass an empty iterable to opt out, or select one of them. The project HIP
        extension and a supported DCU must be available when non-empty.
    **deepmd_kwargs
        Keyword arguments forwarded to :class:`deepmd.infer.DeepPot`.

    Notes
    -----
    This project's "CUDA" tensors use PyTorch's ``device="cuda"`` API on a
    HIP/DCU backend; the operators are not NVIDIA CUDA kernels. Native patching
    attaches to DeepMD's live private ``_dpmodel`` tree and therefore cannot
    target legacy ``pt`` models or exported ``.pt2``/``.pte`` artifacts. An
    explicit baseline selection does not rewrite operators already baked into
    an exported graph.
    """

    def __init__(
        self,
        model_file: str | os.PathLike[str],
        *,
        native_ops: Optional[Iterable[str]] = None,
        fused: Any = None,
        **deepmd_kwargs: Any,
    ) -> None:
        self.model_file = str(Path(model_file).expanduser())
        if "tilelang_ops" in deepmd_kwargs:
            raise RuntimeError(
                "TileLang dispatch was removed from this project; drop the "
                "tilelang_ops argument and use native_ops / fused instead"
            )
        selected_native = _normalize_native_ops(native_ops)
        artifact_suffix = Path(self.model_file).suffix.lower()
        if selected_native and artifact_suffix != ".pt":
            raise RuntimeError(
                "native DPA4 operators can only patch eager pt_expt `.pt` "
                "checkpoints; `.pt2` and `.pte` artifacts are not patchable "
                "after export"
            )

        self._deep_pot = _create_deep_pot(self.model_file, **deepmd_kwargs)

        self.fused_seams = _normalize_fused(fused)
        self.fused_ops: dict[str, Any] = {}
        if self.fused_seams:
            from .fused_deepmd import deepmd_patch as _deepmd_patch
            from .fused_deepmd import status as _fused_status

            self.fused_ops = _fused_status(
                _deepmd_patch(self._deep_pot, **self.fused_seams)
            )

        self.backend = _backend_name(self._deep_pot)
        self.native_ops = selected_native
        self.uses_native_ops = bool(selected_native)
        self.native_ops_patched: dict[str, int] = {}
        if self.uses_native_ops:
            try:
                self.native_ops_patched = _install_native_ops(
                    self._deep_pot, self.native_ops
                )
            except Exception:
                try:
                    self.close()
                except Exception:
                    pass
                raise

    def eval(
        self,
        coords: Any,
        cells: Any,
        atom_types: Any,
        *,
        atomic: bool = False,
        fparam: Any = None,
        aparam: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Evaluate energy, forces and virial using DeepMD's public API."""
        return self._deep_pot.eval(
            coords,
            cells,
            atom_types,
            atomic=atomic,
            fparam=fparam,
            aparam=aparam,
            **kwargs,
        )

    predict = eval
    __call__ = eval

    def get_rcut(self) -> float:
        return self._deep_pot.get_rcut()

    def get_type_map(self) -> list[str]:
        return self._deep_pot.get_type_map()

    def get_ntypes(self) -> int:
        return self._deep_pot.get_ntypes()

    def get_model_def_script(self) -> Any:
        return self._deep_pot.get_model_def_script()

    def close(self) -> None:
        if self.fused_ops:
            from .fused_deepmd import uninstall as _uninstall_fused

            _uninstall_fused(self._deep_pot)
            self.fused_ops = {}
        close = getattr(self._deep_pot, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "DPA4Model":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"DPA4Model(model_file={self.model_file!r}, backend={self.backend!r}, "
            f"native_ops={self.native_ops!r}, fused_ops={self.fused_ops!r})"
        )


__all__ = ["DPA4Model"]
