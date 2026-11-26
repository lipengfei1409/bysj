"""Host tests for the FX graph printer.

Tracing needs DeepMD and a checkpoint; what is testable anywhere is the part
that turns a graph into what a report quotes -- op naming, family bucketing,
shape extraction, rendering -- plus the target registry and its arguments.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tests" / "deepmd" / "profile" / "fx_graph.py"


def _module():
    spec = importlib.util.spec_from_file_location("fx_graph", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _traced(fn, *args):
    from torch.fx.experimental.proxy_tensor import make_fx

    return make_fx(fn, tracing_mode="real")(*args)


class NamingTests(unittest.TestCase):
    def test_aten_overloads_collapse_to_the_base_name(self) -> None:
        module = _module()
        self.assertEqual(module.op_name(torch.ops.aten.bmm.default), "aten.bmm")
        self.assertEqual(module.op_name(torch.ops.aten.sum.dim_IntList), "aten.sum")
        self.assertEqual(module.op_name(torch.ops.aten.mul.Tensor), "aten.mul")

    def test_plain_callables_name_themselves(self) -> None:
        import operator

        module = _module()
        self.assertEqual(module.op_name(operator.mul), "_operator.mul")
        self.assertEqual(module.op_name(torch.relu), "torch.relu")

    def test_families_follow_the_base_name(self) -> None:
        module = _module()
        self.assertEqual(module.family_of(torch.ops.aten.bmm.default), "gemm")
        self.assertEqual(module.family_of(torch.ops.aten.index_add_.default), "index")
        self.assertEqual(module.family_of(torch.ops.aten._softmax.default), "reduce")
        self.assertEqual(module.family_of(torch.ops.aten.copy_.default), "copy")
        self.assertEqual(module.family_of(torch.ops.aten.view.default), "layout")
        self.assertEqual(module.family_of(torch.ops.aten.relu.default), "other")


class GraphReadingTests(unittest.TestCase):
    def test_a_missing_module_stack_reads_as_empty(self) -> None:
        module = _module()
        traced = _traced(lambda x: x * 2.0, torch.randn(3, 4))
        for node in traced.graph.nodes:
            self.assertEqual(module.module_stack(node), [])

    def test_shapes_come_from_tensor_meta(self) -> None:
        module = _module()
        weight = torch.ones(4, 4)  # built outside: real tracing traces the factory too
        traced = _traced(lambda x: x @ weight, torch.randn(2, 4, 4))
        shapes = [module.shape_of(n) for n in traced.graph.nodes if n.op == "call_function"]
        self.assertIn((2, 4, 4), shapes)

    def test_render_prints_one_line_per_node_with_shape(self) -> None:
        module = _module()
        traced = _traced(lambda x: torch.relu(x) + 1.0, torch.randn(2, 3))
        calls = [n for n in traced.graph.nodes if n.op == "call_function"]
        text = module.render(calls)
        self.assertEqual(len(text.splitlines()), len(calls))
        self.assertIn("aten.relu", text)
        self.assertIn("(2, 3)", text)

    def test_counts_cover_every_operation_and_bucket_them(self) -> None:
        module = _module()

        class Toy(torch.nn.Module):
            def forward(self, x):
                weight = torch.ones(1, 4, 4)
                return torch.sum(torch.bmm(x, weight) * 2.0, dim=1)

        traced = torch.fx.symbolic_trace(Toy())
        node = type("Opts", (), {"raw": False})()
        ops, families, text = module._describe(traced.graph, node)
        self.assertEqual(sum(ops.values()), sum(families.values()))
        for expected in ("gemm", "elementwise", "reduce"):
            self.assertIn(expected, families)
        self.assertTrue(text)


class StubTests(unittest.TestCase):
    def test_stub_lib_is_a_noop_when_deepmd_is_absent(self) -> None:
        # Without DeepMD importable the helper must not pretend to work: the
        # caller passes --stub-lib only on a host that has the *source*.
        module = _module()
        try:
            import deepmd  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaises(ModuleNotFoundError):
                module.install_stub_lib()
        else:
            self.skipTest("this host can import DeepMD")


class RegistryTests(unittest.TestCase):
    def test_the_two_required_targets_exist(self) -> None:
        module = _module()
        self.assertEqual(sorted(module.TARGETS), ["model", "so2_conv"])

    def test_every_target_has_a_description_and_a_four_argument_builder(self) -> None:
        module = _module()
        for name, (description, builder) in module.TARGETS.items():
            self.assertTrue(description, name)
            self.assertEqual(builder.__code__.co_argcount, 4, name)

    def test_list_targets_names_both_and_exits_zero(self) -> None:
        module = _module()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = module.main(["--list-targets"])
        self.assertEqual(code, 0)
        self.assertIn("so2_conv", buffer.getvalue())

    def test_trace_device_defaults_to_cpu_and_accepts_cuda(self) -> None:
        # The artifact must be readable anywhere by default; a DCU trace is opt-in
        # because it needs the device.
        module = _module()
        captured = {}
        real_parse = module.argparse.ArgumentParser.parse_args

        def spy(self, args=None, namespace=None):
            result = real_parse(self, args, namespace)
            captured["trace_device"] = getattr(result, "trace_device", "<missing>")
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
