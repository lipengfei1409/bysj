"""Typed adapter contract shared by heterogeneous model families."""
from __future__ import annotations

from typing import Any, Protocol, TypeVar

RequestT = TypeVar("RequestT", contravariant=True)
ResponseT = TypeVar("ResponseT", covariant=True)


class ModelAdapter(Protocol[RequestT, ResponseT]):
    """Model-family adapter; request and response schemas remain family-owned."""

    def predict(self, request: RequestT, **options: Any) -> ResponseT: ...

    def close(self) -> None: ...
