"""Request/response types for the built-in text-generation adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TextGenerationRequest:
    prompt: str | None = None
    messages: Sequence[Mapping[str, Any]] | None = None
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self):
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("provide exactly one of prompt or messages")
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if self.messages is not None:
            if not self.messages:
                raise ValueError("messages must not be empty")
            for message in self.messages:
                if not isinstance(message, Mapping):
                    raise TypeError("each chat message must be a mapping")
                if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
                    raise ValueError("chat messages require string role and content fields")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0,1]")


@dataclass(frozen=True)
class TextGenerationResponse:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    generated_tokens: int
