"""Host tests for the FX subgraph extractor's bookkeeping.

The extractor needs DeepMD and a checkpoint to do anything useful; what is
testable anywhere is the part that decides what a node is and how it renders:
module-scope lookup, shape extraction, and the one-line-per-node form.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tests" / "deepmd" / "profile" / "extract_fx_subgraph.py"


def _module():
    spec = importlib.util.spec_from_file_location("extract_fx_subgraph", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HelperTests(unittest.TestCase):
    def _traced(self, module, *args):
        from torch.fx.experimental.proxy_tensor import make_fx

        return make_fx(module, tracing_mode="real")(*args)

    def test_a_missing_module_stack_reads_as_empty(self) -> None:
        extracted = _module()
        traced = self._traced(lambda x: x * 2.0, torch.randn(3, 4))
        for node in traced.graph.nodes:
            self.assertEqual(extracted.module_stack(node), [])

    def test_shapes_come_from_tensor_meta(self) -> None:
        extracted = _module()
        weight = torch.ones(4, 4)  # built outside: real tracing would trace the factory too
        traced = self._traced(lambda x: x @ weight, torch.randn(2, 4, 4))
        shapes = [extracted.shape_of(n) for n in traced.graph.nodes if n.op == "call_function"]
        self.assertIn((2, 4, 4), shapes)

    def test_render_prints_one_line_per_node_with_args(self) -> None:
        extracted = _module()
        traced = self._traced(lambda x: torch.relu(x) + 1.0, torch.randn(2, 3))
        calls = [n for n in traced.graph.nodes if n.op == "call_function"]
        text = extracted.render(calls)
        self.assertEqual(len(text.splitlines()), len(calls))
        self.assertIn("aten.relu", text)
        self.assertIn("(2, 3)", text)

    def test_anchor_slice_keeps_only_the_input_closure(self) -> None:
        extracted = _module()
        traced = self._traced(lambda x: torch.sum(torch.sigmoid(x) * 2.0, dim=0), torch.randn(4, 3))
        selected, note = extracted.slice_by_anchor(traced.graph, __import__("re").compile("_softmax|sigmoid"))
        self.assertTrue(selected)
        self.assertIn("anchor", note)
        # Everything kept must be on the path to the anchor.
        kept = {n.name for n in selected}
        self.assertTrue(any("sigmoid" in n.name for n in selected))


if __name__ == "__main__":
    unittest.main()
