"""DeepMD-backed inference adapter with opt-in project operators.

This module deliberately delegates DPA4 architecture, checkpoint loading,
neighbor-list handling, and force/virial evaluation to DeepMD-kit. Selected
TileLang operators dispatch during model construction via DeepMD's existing
``DP_TILELANG_*`` gates. The standalone HIP GIE and RMSNorm operators can be
attached per-instance to matching eager ``pt_expt`` modules with ``native_ops``.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from types import MethodType
from typing import Any, Callable, Iterable, Iterator, Optional

_TRUE = frozenset(("1", "true", "yes", "on"))
_TILELANG_ENV_LOCK = RLock()

_TILELANG_ALIASES = {
    "dpa4": "DP_TILELANG_DPA4",
    "radial_basis": "DP_TILELANG_RADIAL_BASIS",
    "c3_envelope": "DP_TILELANG_C3_ENVELOPE",
    "c3_cutoff_envelope": "DP_TILELANG_C3_ENVELOPE",
    "edge_geometry": "DP_TILELANG_EDGE_GEOMETRY",
    "edge_degree": "DP_TILELANG_EDGE_DEGREE",
    "edge_quaternion": "DP_TILELANG_EDGE_QUATERNION",
    "edge_type_features": "DP_TILELANG_EDGE_TYPE_FEATURES",
    "wigner_l1": "DP_TILELANG_WIGNER_L1",
    "zonal_scatter": "DP_TILELANG_ZONAL_SCATTER",
    "env_seed_matrix": "DP_TILELANG_ENV_SEED_MATRIX",
    "environment_matrix": "DP_TILELANG_ENV_SEED_MATRIX",
    "env_seed_scatter": "DP_TILELANG_ENV_SEED_SCATTER",
    "env_seed_gram": "DP_TILELANG_ENV_SEED_GRAM",
    "so2_rotation_l1": "DP_TILELANG_SO2_ROTATION_L1",
    "so2_radial_scale": "DP_TILELANG_SO2_RADIAL_SCALE",
    "so2_linear_l1": "DP_TILELANG_SO2_LINEAR_L1",
    "attention_softmax": "DP_TILELANG_ATTENTION_SOFTMAX",
    "attention_aggregate": "DP_TILELANG_ATTENTION_AGGREGATE",
    "grid_product": "DP_TILELANG_GRID_PRODUCT",
}
_TILELANG_FLAGS = tuple(dict.fromkeys(_TILELANG_ALIASES.values()))
_FLAG_ALIASES = {flag.lower(): flag for flag in _TILELANG_FLAGS}
_NATIVE_OPS = frozenset(("gie", "rmsnorm"))


def _normalize_tilelang_ops(
    tilelang_ops: Optional[Iterable[str]],
) -> Optional[tuple[str, ...]]:
    """Normalize operator aliases to their DeepMD environment flag names.

    ``None`` preserves the caller's existing environment. An empty iterable
    explicitly disables every known TileLang DPA4 selector for this model.
    """
    if tilelang_ops is None:
        return None
    values = (tilelang_ops,) if isinstance(tilelang_ops, str) else tilelang_ops
    flags = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("tilelang_ops entries must be strings")
        key = value.strip().lower()
        if key in _TILELANG_ALIASES:
            flags.add(_TILELANG_ALIASES[key])
        elif key in _FLAG_ALIASES:
            flags.add(_FLAG_ALIASES[key])
        else:
            choices = ", ".join(sorted(_TILELANG_ALIASES))
            raise ValueError(
                f"unknown TileLang DPA4 operator {value!r}; choose from: {choices}"
            )
    return tuple(sorted(flags))


def _normalize_native_ops(native_ops: Optional[Iterable[str]]) -> tuple[str, ...]:
    """Normalize the explicitly selected native DPA4 operator names."""
    if native_ops is None:
        return ()
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


def _active_tilelang_flags() -> tuple[str, ...]:
    return tuple(
        flag
        for flag in _TILELANG_FLAGS
        if os.environ.get(flag, "0").strip().lower() in _TRUE
    )


@contextmanager
def _tilelang_environment(
    selected_flags: Optional[Iterable[str]],
) -> Iterator[tuple[str, ...]]:
    """Lock and scope TileLang selectors while DeepPot captures them."""
    with _TILELANG_ENV_LOCK:
        previous = {flag: os.environ.get(flag) for flag in _TILELANG_FLAGS}
        try:
            if selected_flags is not None:
                selected = set(selected_flags)
                for flag in _TILELANG_FLAGS:
                    os.environ[flag] = "1" if flag in selected else "0"
            yield _active_tilelang_flags()
        finally:
            for flag, value in previous.items():
                if value is None:
                    os.environ.pop(flag, None)
                else:
                    os.environ[flag] = value


def _ensure_project_tilelang_ops() -> None:
    """Ensure DeepMD's lazy ``kernels.tilelang_ops`` import resolves here."""
    try:
        module = importlib.import_module("kernels.tilelang_ops")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TileLang operators were requested, but this project's "
            "kernels.tilelang_ops package is not importable. Install myproject "
            "with `pip install -e .` or put its python/ directory first on "
            "PYTHONPATH."
        ) from error

    expected = Path(__file__).resolve().parents[1] / "kernels" / "tilelang_ops"
    actual_file = getattr(module, "__file__", None)
    if actual_file is None or Path(actual_file).resolve().parent != expected.resolve():
        raise RuntimeError(
            "DeepMD's `kernels.tilelang_ops` import does not resolve to this "
            f"project (expected {expected}, got {actual_file!r}); install or "
            "reorder myproject/python on PYTHONPATH."
        )


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
    if "gie" in native_ops and any(
        bool(getattr(module, "_tilelang_scatter_enabled", False))
        for module in targets["gie"]
    ):
        raise RuntimeError(
            "native GIE and DP_TILELANG_ZONAL_SCATTER cannot be selected for "
            "the same model"
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

    patches: list[tuple[Any, Any]] = []
    counts: dict[str, int] = {}
    if "gie" in native_ops:
        patches.extend(
            (module, _make_gie_call(module, implementations["gie"]))
            for module in targets["gie"]
        )
        counts["gie"] = len(targets["gie"])
    if "rmsnorm" in native_ops:
        patches.extend(
            (module, _make_rmsnorm_call(module, implementations["rmsnorm"]))
            for module in targets["rmsnorm"]
        )
        counts["rmsnorm"] = len(targets["rmsnorm"])

    originals = [(module, module.call) for module, _ in patches]
    try:
        for module, patched_call in patches:
            module.call = patched_call
    except Exception:
        for module, original_call in originals:
            module.call = original_call
        raise
    return counts


class DPA4Model:
    """DeepMD DPA4/SeZM inference facade.

    DeepMD-kit remains responsible for loading the selected DPA4 model
    scenario and evaluating its energy, force, and virial. Set ``tilelang_ops``
    to a sequence of operator aliases to route just those stages through this
    project's TileLang kernels. Use ``"dpa4"`` only for DeepMD's guarded
    single-focus water fast path. An empty sequence forces the PyTorch baseline;
    ``None`` leaves existing ``DP_TILELANG_*`` environment settings unchanged.
    Set ``native_ops`` to ``("gie", "rmsnorm")`` to patch the matching eager
    ``pt_expt`` modules with this project's HIP/DCU operators. Native patches
    are disabled by default and are applied only to that loaded model instance.

    Parameters
    ----------
    model_file
        Any model artifact supported by ``deepmd.infer.DeepPot``.
    tilelang_ops
        Optional aliases such as ``"radial_basis"`` and ``"edge_geometry"``.
        Individual operators can be selected for non-water DPA4 configurations;
        DeepMD raises its normal error if a selected operator does not support a
        model's configuration.
    native_ops
        Optional project HIP/DCU operators to patch into an eager ``pt_expt``
        checkpoint: ``"gie"`` and/or ``"rmsnorm"``. Defaults to no native
        patches. The project extension must be built and a supported DCU visible.
    **deepmd_kwargs
        Keyword arguments forwarded to :class:`deepmd.infer.DeepPot`.

    Notes
    -----
    Runtime TileLang dispatch is available while DeepMD constructs an eager
    ``pt_expt`` model, normally from a ``pt_expt`` training checkpoint ending
    in ``.pt``. A regular legacy ``pt`` checkpoint can still be evaluated by
    DeepPot, but requesting TileLang for it raises instead of silently running
    the non-TileLang backend. Exported ``.pt2``/``.pte`` archives can be
    evaluated normally, but their operator choices were fixed at export time;
    pass no TileLang selectors when loading them. Native patches likewise
    require an eager ``pt_expt`` ``.pt`` model because they attach to DeepMD's
    live private ``_dpmodel`` tree; unsupported inputs continue through the
    original DeepMD call.
    """

    def __init__(
        self,
        model_file: str | os.PathLike[str],
        *,
        tilelang_ops: Optional[Iterable[str]] = None,
        native_ops: Optional[Iterable[str]] = None,
        **deepmd_kwargs: Any,
    ) -> None:
        self.model_file = str(Path(model_file).expanduser())
        selected = _normalize_tilelang_ops(tilelang_ops)
        selected_native = _normalize_native_ops(native_ops)
        artifact_suffix = Path(self.model_file).suffix.lower()
        if selected_native and artifact_suffix != ".pt":
            raise RuntimeError(
                "native DPA4 operators can only patch eager pt_expt `.pt` "
                "checkpoints; `.pt2` and `.pte` artifacts are not patchable "
                "after export"
            )

        # DeepMD captures selectors while constructing the descriptor. Keep
        # construction serialized against other DPA4Model instances that may
        # select a different operator set in this process.
        with _tilelang_environment(selected) as active_flags:
            if active_flags and artifact_suffix in (".pt2", ".pte"):
                raise RuntimeError(
                    f"TileLang selectors cannot be changed while loading an exported "
                    f"{artifact_suffix} artifact; its kernels were fixed at export "
                    "time. Use an eager pt_expt training checkpoint to select "
                    "operators at load time, or clear tilelang_ops for this artifact."
                )
            if active_flags:
                _ensure_project_tilelang_ops()
            self._deep_pot = _create_deep_pot(self.model_file, **deepmd_kwargs)

        self.backend = _backend_name(self._deep_pot)
        self.tilelang_ops = tuple(active_flags)
        self.uses_tilelang = bool(active_flags)
        self.native_ops = selected_native
        self.uses_native_ops = bool(selected_native)
        self.native_ops_patched: dict[str, int] = {}
        if self.uses_tilelang and self.backend != "pt_expt":
            self.close()
            raise RuntimeError(
                "TileLang DPA4 operators require DeepMD's `pt_expt` backend, "
                f"but this model loaded with backend {self.backend!r}. The "
                "regular `pt` checkpoint backend does not dispatch this "
                "project's TileLang ops; use a pt_expt-compatible checkpoint "
                "or pass tilelang_ops=() for the DeepMD baseline."
            )
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
            f"tilelang_ops={self.tilelang_ops!r}, native_ops={self.native_ops!r})"
        )


__all__ = ["DPA4Model"]
