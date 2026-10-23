"""Lazy model-family registry for optional inference adapters."""
from __future__ import annotations

from importlib import import_module
from threading import RLock

from .runner import InferenceRunner


_BUILTINS = {
    ("dpa4", "energy-force-virial"): "model.dpa4.inference:create_adapter",
    ("qwen3", "text-generation"): "model.inference.adapters.transformers:create_text_generation_adapter",
    ("qwen3.5", "text-generation"): "model.inference.adapters.transformers:create_text_generation_adapter",
}
_FACTORIES = {}
_LOCK = RLock()


def register_adapter(family, task, factory, *, replace=False):
    """Register a lazy factory or callable without importing the model backend."""
    key = (str(family).strip().lower(), str(task).strip().lower())
    if not all(key):
        raise ValueError("model family and task must be nonempty")
    if not callable(factory) and not isinstance(factory, str):
        raise TypeError("factory must be callable or a 'module:attribute' entrypoint")
    with _LOCK:
        if (key in _FACTORIES or key in _BUILTINS) and not replace:
            raise ValueError(f"adapter {key[0]!r}/{key[1]!r} is already registered")
        _FACTORIES[key] = factory


def registered_adapters():
    with _LOCK:
        return tuple(sorted(set(_BUILTINS) | set(_FACTORIES)))


def _resolve_factory(factory):
    if callable(factory):
        return factory
    if ":" not in factory:
        raise ValueError("adapter entrypoints must use 'module:attribute' syntax")
    module_name, attribute = factory.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("adapter entrypoints must use 'module:attribute' syntax")
    module = import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(f"adapter entrypoint {factory!r} does not exist") from exc


def create_runner(family, task, **config):
    """Load one registered adapter only when requested by the caller."""
    key = (str(family).strip().lower(), str(task).strip().lower())
    with _LOCK:
        factory = _FACTORIES.get(key, _BUILTINS.get(key))
    if factory is None:
        supported = ", ".join(f"{name}/{kind}" for name, kind in registered_adapters())
        raise KeyError(f"unsupported model adapter {key[0]!r}/{key[1]!r}; registered: {supported}")
    adapter = _resolve_factory(factory)(**config)
    return InferenceRunner(key[0], key[1], adapter)
