"""DeepMD-backed inference adapter with opt-in project TileLang operators.

This module deliberately delegates DPA4 architecture, checkpoint loading,
neighbor-list handling, and force/virial evaluation to DeepMD-kit. The selected
TileLang operators are the standalone implementations in
``kernels.tilelang_ops``; DeepMD's ``pt_expt`` DPA4 modules dispatch to them via
the existing ``DP_TILELANG_*`` environment gates during model construction.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Iterator, Optional

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


class DPA4Model:
    """DeepMD DPA4/SeZM inference facade.

    DeepMD-kit remains responsible for loading the selected DPA4 model
    scenario and evaluating its energy, force, and virial. Set ``tilelang_ops``
    to a sequence of operator aliases to route just those stages through this
    project's TileLang kernels. Use ``"dpa4"`` only for DeepMD's guarded
    single-focus water fast path. An empty sequence forces the PyTorch baseline;
    ``None`` leaves existing ``DP_TILELANG_*`` environment settings unchanged.

    Parameters
    ----------
    model_file
        Any model artifact supported by ``deepmd.infer.DeepPot``.
    tilelang_ops
        Optional aliases such as ``"radial_basis"`` and ``"edge_geometry"``.
        Individual operators can be selected for non-water DPA4 configurations;
        DeepMD raises its normal error if a selected operator does not support a
        model's configuration.
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
    pass no TileLang selectors when loading them.
    """

    def __init__(
        self,
        model_file: str | os.PathLike[str],
        *,
        tilelang_ops: Optional[Iterable[str]] = None,
        **deepmd_kwargs: Any,
    ) -> None:
        self.model_file = str(Path(model_file).expanduser())
        selected = _normalize_tilelang_ops(tilelang_ops)
        artifact_suffix = Path(self.model_file).suffix.lower()

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
        if self.uses_tilelang and self.backend != "pt_expt":
            self.close()
            raise RuntimeError(
                "TileLang DPA4 operators require DeepMD's `pt_expt` backend, "
                f"but this model loaded with backend {self.backend!r}. The "
                "regular `pt` checkpoint backend does not dispatch this "
                "project's TileLang ops; use a pt_expt-compatible checkpoint "
                "or pass tilelang_ops=() for the DeepMD baseline."
            )

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
            f"tilelang_ops={self.tilelang_ops!r})"
        )


__all__ = ["DPA4Model"]
