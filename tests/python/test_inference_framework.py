import importlib
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch

from model.inference import create_runner, register_adapter, registered_adapters
from model.inference.adapters.transformers import TransformersTextAdapter
from model.inference.types import TextGenerationRequest, TextGenerationResponse


class _EchoAdapter:
    def __init__(self, suffix="!"):
        self.suffix = suffix
        self.closed = False

    def predict(self, request, **options):
        return request + self.suffix

    def close(self):
        self.closed = True


class InferenceFrameworkTests(unittest.TestCase):
    def test_model_import_is_lazy(self):
        root = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=f"{root / 'python'}:{root}")
        code = (
            "import sys, model; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'deepmd' not in sys.modules; "
            "assert callable(model.create_runner)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_registry_dispatches_factory_and_closes_adapter(self):
        family = "unit-test-echo"
        register_adapter(family, "text", lambda suffix="!": _EchoAdapter(suffix))
        self.assertIn((family, "text"), registered_adapters())
        with create_runner(family, "text", suffix="?") as runner:
            self.assertEqual(runner.predict("hello"), "hello?")
            adapter = runner.adapter
        self.assertTrue(adapter.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runner.predict("again")

    def test_legacy_dpa4_import_paths_alias_canonical_modules(self):
        self.assertIs(
            importlib.import_module("model.modeling.numerical"),
            importlib.import_module("model.dpa4.modeling.numerical"),
        )
        self.assertIs(
            importlib.import_module("model.runner"),
            importlib.import_module("model.dpa4.runner"),
        )
        self.assertIs(
            importlib.import_module("model.checkpoint"),
            importlib.import_module("model.dpa4.checkpoint"),
        )

    def test_dpa4_adapter_delegates_to_deepmd_runner(self):
        from model.dpa4.system import DeepmdSystem
        from model.dpa4.inference import DPA4DeepMDAdapter

        class FakeRunner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.closed = False

            def eval(self, system, **kwargs):
                return system, kwargs

            def close(self):
                self.closed = True

        system = DeepmdSystem(
            coords=torch.zeros(1, 1, 3).numpy(), cells=None,
            atype=torch.tensor([0]).numpy(), type_map=("H",), energy=None, force=None,
        )
        with patch("model.dpa4.runner.DPA4Runner", FakeRunner):
            adapter = DPA4DeepMDAdapter(checkpoint="local.pt", device="cpu")
            self.assertEqual(adapter.predict(system, max_frames=1), (system, {"max_frames": 1}))
            runner = adapter._runner
            adapter.close()
        self.assertTrue(runner.closed)

    def test_transformers_adapter_uses_chat_template_without_download(self):
        class FakeTokenizer:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                tokenizer = FakeTokenizer()
                tokenizer.model_id = model_id
                tokenizer.load_kwargs = kwargs
                return tokenizer

            def apply_chat_template(self, messages, **kwargs):
                return torch.tensor([[1, 2, 3]], dtype=torch.int64)

            def decode(self, token_ids, **kwargs):
                return "ok"

        class FakeModel:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                model = FakeModel()
                model.model_id = model_id
                model.load_kwargs = kwargs
                return model

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.is_eval = True
                return self

            def generate(self, input_ids, **kwargs):
                suffix = input_ids.new_tensor([[7, 8]])
                return torch.cat((input_ids, suffix), dim=1)

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = FakeTokenizer
        transformers.AutoModelForCausalLM = FakeModel
        with patch.dict(sys.modules, {"transformers": transformers}):
            runner = create_runner(
                "qwen3", "text-generation", model_id="local-test-model",
                device="cpu", dtype="float32",
            )
            result = runner.predict(TextGenerationRequest(
                messages=({"role": "user", "content": "hello"},), max_new_tokens=2,
            ))
        self.assertIsInstance(result, TextGenerationResponse)
        self.assertEqual(result.text, "ok")
        self.assertEqual(result.prompt_tokens, 3)
        self.assertEqual(result.generated_tokens, 2)
        self.assertFalse(runner.adapter.model.load_kwargs["trust_remote_code"])

    def test_transformers_adapter_fails_clearly_for_unregistered_model_class(self):
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: object()
        )
        def unsupported(*args, **kwargs):
            raise ValueError("unregistered config")
        transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=unsupported)
        with patch.dict(sys.modules, {"transformers": transformers}):
            with self.assertRaisesRegex(RuntimeError, "no compatible causal-LM"):
                TransformersTextAdapter(model_id="unsupported-qwen", device="cpu")

    def test_example_help_does_not_load_optional_model_backends(self):
        root = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=f"{root / 'python'}:{root}")
        for script in (root / "examples/dpa4/infer.py", root / "examples/qwen3/infer.py"):
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
