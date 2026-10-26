"""Lifecycle wrapper around a family-specific inference adapter."""
from __future__ import annotations

from typing import Any


class InferenceRunner:
    """Common lifecycle; prediction payloads remain owned by each adapter."""

    def __init__(self, family: str, task: str, adapter: Any):
        if not family or not task:
            raise ValueError("model family and task must be nonempty")
        if not callable(getattr(adapter, "predict", None)):
            raise TypeError("model adapter must provide predict(request, **options)")
        self.family = family
        self.task = task
        self.adapter = adapter
        self._closed = False

    def predict(self, request, **options):
        if self._closed:
            raise RuntimeError("inference runner is closed")
        return self.adapter.predict(request, **options)

    def close(self):
        if not self._closed:
            close = getattr(self.adapter, "close", None)
            if callable(close):
                close()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("inference runner is closed")
        return self

    def __exit__(self, *exc):
        self.close()
