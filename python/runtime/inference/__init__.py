"""Generic, lazy model inference framework."""
from .protocols import ModelAdapter
from .registry import create_runner, register_adapter, registered_adapters
from .runner import InferenceRunner
from .types import TextGenerationRequest, TextGenerationResponse

__all__ = [
    "InferenceRunner", "ModelAdapter", "TextGenerationRequest",
    "TextGenerationResponse", "create_runner", "register_adapter",
    "registered_adapters",
]
