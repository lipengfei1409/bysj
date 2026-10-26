"""Public model definitions; execution and runners live in :mod:`runtime`."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DPA4Config": ("model.dpa4.modeling.config", "DPA4Config"),
    "DPA4Model": ("model.dpa4.modeling.dpa4", "DPA4Model"),
    "AtomicBatch": ("model.dpa4.modeling.types", "AtomicBatch"),
    "EdgeGraph": ("model.dpa4.modeling.types", "EdgeGraph"),
    "EdgeFeatures": ("model.dpa4.modeling.types", "EdgeFeatures"),
    "ModelOutput": ("model.dpa4.modeling.types", "ModelOutput"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name == "dpa4":
        value = import_module("model.dpa4")
        globals()[name] = value
        return value
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
