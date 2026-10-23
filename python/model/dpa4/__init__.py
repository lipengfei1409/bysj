"""Canonical, lazily loaded DPA4 Python model package."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CheckpointSpec": ("model.dpa4.checkpoint", "CheckpointSpec"),
    "load_checkpoint_spec": ("model.dpa4.checkpoint", "load_checkpoint_spec"),
    "DPA4Config": ("model.dpa4.modeling.config", "DPA4Config"),
    "DPA4Model": ("model.dpa4.modeling.dpa4", "DPA4Model"),
    "AtomicBatch": ("model.dpa4.modeling.types", "AtomicBatch"),
    "EdgeGraph": ("model.dpa4.modeling.types", "EdgeGraph"),
    "EdgeFeatures": ("model.dpa4.modeling.types", "EdgeFeatures"),
    "ModelOutput": ("model.dpa4.modeling.types", "ModelOutput"),
    "ShapeBackend": ("model.dpa4.modeling.backend", "ShapeBackend"),
    "PlaceholderBackend": ("model.dpa4.modeling.backend", "PlaceholderBackend"),
    "NumericalBackend": ("model.dpa4.modeling.numerical", "NumericalBackend"),
    "RadialBasis": ("model.dpa4.modeling.components", "RadialBasis"),
    "RadialMLP": ("model.dpa4.modeling.components", "RadialMLP"),
    "TypeEmbedding": ("model.dpa4.modeling.components", "TypeEmbedding"),
    "DPA4Runner": ("model.dpa4.runner", "DPA4Runner"),
    "Prediction": ("model.dpa4.runner", "Prediction"),
    "compute_metrics": ("model.dpa4.runner", "compute_metrics"),
    "configure_device": ("model.dpa4.runner", "configure_device"),
    "DeepmdSystem": ("model.dpa4.system", "DeepmdSystem"),
    "load_deepmd_system": ("model.dpa4.system", "load_deepmd_system"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name == "modeling":
        value = import_module("model.dpa4.modeling")
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
