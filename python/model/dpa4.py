"""DeepMD-backed inference facade with this project's fused operator seams.

DeepMD-kit owns the DPA4/SeZM model graph, checkpoint loading, neighbour lists,
energy, forces and virial; this module only adds two things on top of
``deepmd.infer.DeepPot``:

* ``fused=`` installs the fused segmented HIP operators through
  :mod:`model.fused_deepmd`, which binds them into DeepMD's own dispatch seams
  and keeps a per-call fallback to the stock implementation;
* a ``close()``/context-manager lifecycle that removes those bindings before the
  underlying ``DeepPot`` is closed.

The earlier per-instance GIE/RMSNorm patcher lived here too. It required an
eager ``pt_expt`` checkpoint exposing ``deep_eval._dpmodel``, so it could not run
on the legacy ``pt`` checkpoints this project profiles, and it duplicated
``model.fused_deepmd``'s registration/fallback design with a second, independent
patch registry. It was removed; ``csrc/gie.cpp`` and ``csrc/rmsnorm.cpp`` remain
as tested primitives under :mod:`kernels`, and the destination reduction of
attention-free SO(2) convolution is served by the ``envelope`` fused seam.
TileLang dispatch was removed from the project as well.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_FUSED_SEAMS = ("softmax", "envelope", "force", "aggregate", "pack")


def _normalize_fused(fused: Any) -> dict[str, bool] | None:
    """Translate the ``fused`` switch into the per-seam switches."""
    if not fused:
        return None
    if fused is True:
        return {name: True for name in _FUSED_SEAMS}
    if isinstance(fused, str):
        fused = (fused,)
    if isinstance(fused, (list, tuple, set, frozenset)):
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


class DPA4Model:
    """``DeepPot`` facade that can bind the fused segmented operators.

    Parameters
    ----------
    model_file
        A model artifact supported by ``deepmd.infer.DeepPot``.
    fused
        Install this project's fused operators into the loaded model:
        ``True`` installs every seam, a string or sequence selects seams
        (``"softmax"``, ``"envelope"``, ``"force"``, ``"aggregate"``, ``"pack"``), a
        mapping sets each one, and
        ``None``/``False`` leaves DeepMD untouched. See
        :mod:`model.fused_deepmd` for the seams and their fallback rules.
    **deepmd_kwargs
        Keyword arguments forwarded to :class:`deepmd.infer.DeepPot`.

    Notes
    -----
    This project's "CUDA" tensors use PyTorch's ``device="cuda"`` API on a
    HIP/DCU backend; they are not NVIDIA CUDA kernels.
    """

    def __init__(
        self,
        model_file: str | os.PathLike[str],
        *,
        fused: Any = None,
        so2_conv: str = "off",
        **deepmd_kwargs: Any,
    ) -> None:
        self.model_file = str(Path(model_file).expanduser())
        for removed, hint in (
            ("tilelang_ops", "drop the argument; TileLang dispatch was removed from this project"),
            ("native_ops", "drop the argument and use fused=(\"softmax\", \"envelope\") instead"),
        ):
            if removed in deepmd_kwargs:
                raise RuntimeError(
                    f"{removed} was removed from DPA4Model: {hint}; "
                    "an old call site should fail loudly instead of being "
                    "forwarded to DeepPot"
                )

        # Normalise the switch before loading anything, so a typo fails fast
        # instead of after a multi-GB checkpoint has been read.
        self.fused_seams = _normalize_fused(fused)
        # The translated (hipified) SO(2) convolution was removed: it measured
        # 0.80x the dense path, so it is no longer built or routed.  The argument
        # stays, and only "off" is accepted, so an old call site fails with a
        # sentence instead of a TypeError on an unexpected keyword.
        if so2_conv != "off":
            raise ValueError(
                f"so2_conv={so2_conv!r} is no longer supported: the translated "
                "SO(2) convolution was removed from this project (measured 0.80x "
                "the dense path)"
            )
        self.so2_conv = so2_conv
        self._deep_pot = _create_deep_pot(self.model_file, **deepmd_kwargs)
        self.backend = _backend_name(self._deep_pot)
        self.fused_ops: dict[str, Any] = {}
        if self.fused_seams:
            from .fused_deepmd import deepmd_patch, status

            self.fused_ops = dict(status(deepmd_patch(self._deep_pot, **self.fused_seams)))

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
            from .fused_deepmd import uninstall

            uninstall(self._deep_pot)
        self.fused_ops = {}
        close = getattr(self._deep_pot, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> DPA4Model:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"DPA4Model(model_file={self.model_file!r}, backend={self.backend!r}, "
            f"fused_ops={self.fused_ops!r})"
        )


__all__ = ["DPA4Model"]
