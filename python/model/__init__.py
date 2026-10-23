"""Generic model-inference registry; family-specific packages load lazily."""
from .inference import (
    InferenceRunner,
    ModelAdapter,
    TextGenerationRequest,
    TextGenerationResponse,
    create_runner,
    register_adapter,
    registered_adapters,
)

__all__ = [
    "InferenceRunner", "ModelAdapter", "TextGenerationRequest",
    "TextGenerationResponse", "create_runner", "register_adapter",
    "registered_adapters",
]
