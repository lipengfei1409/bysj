"""Optional HuggingFace text-generation adapter (Transformers loads lazily)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model.inference.types import TextGenerationRequest, TextGenerationResponse


class TransformersTextAdapter:
    """Causal-LM text generation through a compatible Transformers auto class."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        model_kwargs: Mapping[str, Any] | None = None,
    ):
        if not model_id:
            raise ValueError("model_id must be a HuggingFace id or local model directory")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "text generation requires the optional Transformers dependency; "
                "install the qwen extra"
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
        self.model_id = model_id
        self.device = device
        self._torch = torch
        load_options: dict[str, Any] = {"trust_remote_code": bool(trust_remote_code)}
        if dtype == "auto":
            load_options["torch_dtype"] = "auto"
        else:
            dtypes = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if dtype not in dtypes:
                raise ValueError("dtype must be auto, float16, bfloat16, or float32")
            load_options["torch_dtype"] = dtypes[dtype]
        load_options.update(dict(model_kwargs or {}))
        # Remote model code is an explicit opt-in, never enabled by extra kwargs.
        load_options["trust_remote_code"] = bool(trust_remote_code)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=bool(trust_remote_code)
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_options)
        except (KeyError, ValueError, ImportError) as exc:
            raise RuntimeError(
                f"Transformers has no compatible causal-LM implementation for {model_id!r}; "
                "this adapter supports text generation, while Qwen3.5 multimodal variants "
                "may need a processor-specific adapter"
            ) from exc
        self.model.to(device)
        self.model.eval()
        self._closed = False

    def predict(self, request, **options):
        if self._closed:
            raise RuntimeError("text-generation adapter is closed")
        if isinstance(request, str):
            request = TextGenerationRequest(prompt=request)
        elif isinstance(request, Mapping):
            request = TextGenerationRequest(**request)
        if not isinstance(request, TextGenerationRequest):
            raise TypeError(
                "text-generation requests must be strings, mappings, or TextGenerationRequest"
            )

        if request.prompt is not None:
            encoded = self.tokenizer(request.prompt, return_tensors="pt")
        else:
            messages = list(request.messages or ())
            if any(not isinstance(message.get("content"), str) for message in messages):
                raise NotImplementedError(
                    "this adapter supports text chat only; multimodal requests need a processor adapter"
                )
            if not callable(getattr(self.tokenizer, "apply_chat_template", None)):
                raise RuntimeError("the selected tokenizer does not provide a chat template")
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except TypeError:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            if not isinstance(encoded, Mapping):
                encoded = {"input_ids": encoded}

        encoded = {
            name: value.to(self.device)
            for name, value in encoded.items()
            if hasattr(value, "to")
        }
        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("tokenizer must return input_ids with shape [batch,sequence]")
        prompt_length = input_ids.shape[-1]
        generation = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
        }
        if request.do_sample:
            generation.update(temperature=request.temperature, top_p=request.top_p)
        generation.update(options)
        with self._torch.inference_mode():
            output = self.model.generate(**encoded, **generation)
        generated_ids = output[0, prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return TextGenerationResponse(
            text=text,
            token_ids=tuple(int(token) for token in generated_ids.tolist()),
            prompt_tokens=int(prompt_length),
            generated_tokens=int(generated_ids.numel()),
        )

    def close(self):
        self.model = None
        self.tokenizer = None
        self._closed = True


def create_text_generation_adapter(**config):
    return TransformersTextAdapter(**config)
