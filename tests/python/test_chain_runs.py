"""Host test pinning the C6 audit's conclusion.

The audit says a chain-fusion kernel has almost no material on this graph.  That
claim rests on three numbers, so they are pinned here: if a future trace or a
change in the audit's definition moves them, the conclusion has to be revisited
rather than quietly inherited.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tests" / "deepmd" / "profile" / "chain_runs.py"
TRACE = ROOT.parent / "dpa4_inference_trace.json"


def _tool():
    spec = importlib.util.spec_from_file_location("chain_runs", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChainRunsTests(unittest.TestCase):
    def test_op_sets_do_not_drift(self) -> None:
        module = _tool()
        for op in ("mul", "add_", "copy_", "fill_", "sigmoid_backward"):
            self.assertIn(op, module.ELEMENTWISE)
        for op in ("add_", "index_add_", "scatter_add_"):
            self.assertIn(op, module.ACCUMULATE)
        self.assertNotIn("bmm", module.ELEMENTWISE)

    @unittest.skipUnless(TRACE.is_file(), f"needs the reference trace at {TRACE}")
    def test_reference_trace_has_no_chain_material(self) -> None:
        report = _tool().analyse(TRACE, top=3)
        runs = report["runs"]
        self.assertEqual(report["launches"], 2614)
        self.assertAlmostEqual(runs["total_ms"], 100.7, delta=1.0)
        self.assertAlmostEqual(runs["at_least_3_ms"], 16.2, delta=0.5)
        self.assertAlmostEqual(runs["at_least_4_ms"], 15.0, delta=0.5)
        self.assertEqual(report["fill"]["pairs"], 0)
        self.assertAlmostEqual(report["fill"]["total_ms"], 75.4, delta=1.0)
        # The material has to stay a small fraction of the elementwise family,
        # or the audit's conclusion ("chain fusion has nothing to work on") is
        # no longer true.
        self.assertLess(runs["at_least_3_ms"], 0.05 * report["device_ms"])


if __name__ == "__main__":
    unittest.main()
