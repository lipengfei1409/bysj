"""Compatibility imports for the pre-``model.dpa4`` package path."""
from __future__ import annotations

import importlib
import sys

_modeling = importlib.import_module("model.dpa4.modeling")

__all__ = list(_modeling.__all__)
for _name in __all__:
    globals()[_name] = getattr(_modeling, _name)

# Alias submodules so patches against legacy paths affect the canonical module.
for _name in (
    "backend", "components", "config", "dpa4", "loading", "numerical",
    "so3_grid", "types", "wigner",
):
    _module = importlib.import_module(f"model.dpa4.modeling.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
