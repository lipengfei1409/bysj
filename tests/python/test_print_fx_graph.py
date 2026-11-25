"""Host tests for the FX graph printer's bookkeeping.

The printer itself needs DeepMD and a checkpoint; what can be tested anywhere is
the part that turns a graph into the tables a report quotes: op naming, family
bucketing and the summary counts.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tests" / "deepmd" / "profile" / "print_fx_graph.py"


def _module():
    spec = importlib.util.spec_from_file_location("print_fx_graph", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NamingTests(unittest.TestCase):
    def test_aten_overloads_lose_their_default_suffix(self) -> None:
        module = _module()
        self.assertEqual(module.op_name(torch.ops.aten.bmm.default), "aten.bmm")
        self.assertEqual(module.op_name(torch.ops.aten.mul.Tensor), "aten.mul")

    def test_every_overload_collapses_to_the_base_name(self) -> None:
        # The trace prints the resolved overload (``aten.sum.dim_IntList``);
        # the family table must see ``sum`` either way.
        module = _module()
        self.assertEqual(module.op_name(torch.ops.aten.sum.dim_IntList), "aten.sum")
        self.assertEqual(module.op_name(torch.ops.aten.mul.Tensor), "aten.mul")
        self.assertEqual(module.family_of(torch.ops.aten.sum.dim_IntList), "reduce")

    def test_families_follow_the_base_name(self) -> None:
        module = _module()
        self.assertEqual(module.family_of(torch.ops.aten.bmm.default), "gemm")
        self.assertEqual(module.family_of(torch.ops.aten.index_add_.default), "index")
        self.assertEqual(module.family_of(torch.ops.aten._softmax.default), "reduce")
        self.assertEqual(module.family_of(torch.ops.aten.copy_.default), "copy")
        self.assertEqual(module.family_of(torch.ops.aten.relu.default), "other")


class SummariseTests(unittest.TestCase):
    def _graph(self):
        class Toy(torch.nn.Module):
            def forward(self, x):
                weight = torch.ones(4, 4)
                y = torch.bmm(x, weight)
                z = y * 2.0
                return torch.sum(z, dim=1)

        traced = torch.fx.symbolic_trace(Toy())
        return traced.graph

    def test_counts_cover_every_call_function(self) -> None:
        module = _module()
        report = module.summarise(self._graph())
        self.assertEqual(report["call_function"], sum(report["ops"].values()))
        self.assertEqual(report["call_function"], sum(report["families"].values()))
        self.assertGreaterEqual(report["nodes"], report["call_function"])

    def test_families_are_bucketed_not_dropped(self) -> None:
        module = _module()
        report = module.summarise(self._graph())
        self.assertIn("gemm", report["families"])
        self.assertIn("elementwise", report["families"])
        self.assertIn("reduce", report["families"])

    def test_summary_prints_a_percentage_per_family(self) -> None:
        module = _module()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            module.print_summary(module.summarise(self._graph()))
        text = buffer.getvalue()
        self.assertIn("nodes:", text)
        self.assertIn("gemm", text)
        self.assertIn("%", text)


if __name__ == "__main__":
    unittest.main()
