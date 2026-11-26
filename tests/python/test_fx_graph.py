"""Host tests for the FX graph entry point's target registry.

Tracing needs DeepMD and a checkpoint; the registry, the listing and the
argument handling are testable anywhere.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tests" / "deepmd" / "profile" / "fx_graph.py"


def _module():
    spec = importlib.util.spec_from_file_location("fx_graph", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegistryTests(unittest.TestCase):
    def test_the_two_required_targets_exist(self) -> None:
        module = _module()
        self.assertEqual(sorted(module.TARGETS), ["model", "so2_conv"])

    def test_every_target_has_a_description_and_a_builder(self) -> None:
        module = _module()
        for name, (description, builder) in module.TARGETS.items():
            self.assertTrue(description, name)
            self.assertIn("SO2" if name == "so2_conv" else "DPA4", description)
            self.assertTrue(callable(builder), name)
            # builders take (model, sample, traced, options)
            self.assertEqual(builder.__code__.co_argcount, 4, name)

    def test_list_targets_names_both_and_exits_zero(self) -> None:
        module = _module()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = module.main(["--list-targets"])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        self.assertIn("model", text)
        self.assertIn("so2_conv", text)

    def test_trace_device_defaults_to_cpu_and_accepts_cuda(self) -> None:
        # The default must stay CPU: the artifact is readable anywhere, and a
        # DCU trace is an opt-in because it needs the device.
        module = _module()
        parser = None
        captured = {}
        real_parse = module.argparse.ArgumentParser.parse_args

        def spy(self, args=None, namespace=None):
            result = real_parse(self, args, namespace)
            captured["trace_device"] = getattr(result, "trace_device", "<missing>")
            captured["device"] = getattr(result, "device", "<missing>")
            raise SystemExit(0)

        module.argparse.ArgumentParser.parse_args = spy
        try:
            for argv, expected in (
                (["--model-file", "x.pt"], "cpu"),
                (["--model-file", "x.pt", "--trace-device", "cuda"], "cuda"),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        module.main(argv)
                self.assertEqual(captured["trace_device"], expected)
        finally:
            module.argparse.ArgumentParser.parse_args = real_parse

    def test_a_missing_model_file_is_a_usage_error(self) -> None:
        module = _module()
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                module.main(["--target", "model"])


if __name__ == "__main__":
    unittest.main()
