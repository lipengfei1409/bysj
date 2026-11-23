"""Tests for the module-scoped operator time-share report.

The subtle part of ``op_time_share.py`` is the *scope key*: a kernel belongs to
the model only when its **launch** timestamp falls inside the module interval.
Device timestamps lag by up to a second on a real trace, so the fixture below
gives every kernel a device time well outside its launch time and asserts that
the launch-keyed scoping still counts it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import op_time_share  # noqa: E402

MODULE = "nn.Module: ModelWrapper_0"
FORWARD_REGION = "sezm.py(1470): forward_with_edges"
BACKWARD_REGION = "transform_output.py(214): edge_energy_deriv"


def _scope(ts: float, dur: float, name: str) -> dict:
    return {"ph": "X", "cat": "python_function", "name": name, "ts": ts, "dur": dur}


def _launch(external: int, ts: float, dur_us: float, dims) -> list[dict]:
    """A launch + its kernel; the kernel's device time is far from ``ts``."""
    return [
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::mul",
            "ts": ts,
            "dur": 1.0,
            "args": {"External id": external, "Input Dims": dims,
                     "Input type": ["float"]},
        },
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "hipLaunchKernel",
            "ts": ts,
            "dur": 2.0,
            "args": {"External id": external, "correlation": external, "cid": 0},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "void at::native::elementwise_kernel",
            "ts": ts + 5000.0,
            "dur": dur_us,
            "pid": 0,
            "tid": 0,
            "args": {"correlation": external, "device": 0, "stream": 0},
        },
    ]


class OpTimeShareTests(unittest.TestCase):
    def setUp(self) -> None:
        dims = [[1, 181838, 10, 64]]
        events = [
            _scope(1000.0, 100.0, MODULE),
            _scope(1010.0, 30.0, FORWARD_REGION),
            _scope(1060.0, 30.0, BACKWARD_REGION),
            # inside the module but outside both regions (graph replay / casts)
            *_launch(1, 1005.0, 3.0, dims),
            *_launch(2, 1020.0, 7.0, dims),
            *_launch(3, 1070.0, 11.0, dims),
            # outside the module scope: must be dropped
            *_launch(4, 2000.0, 100.0, dims),
        ]
        self.tmp = tempfile.TemporaryDirectory()
        self.trace = Path(self.tmp.name) / "trace.json"
        self.trace.write_text(json.dumps({"traceEvents": events}))
        self.report = op_time_share.analyze(self.trace, module=MODULE, top=5)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scope_uses_launch_time_not_device_time(self) -> None:
        coverage = self.report["coverage"]
        self.assertEqual(coverage["kernels_in_trace"], 4)
        self.assertEqual(coverage["kernels_inside_module"], 3)
        self.assertAlmostEqual(coverage["module_total_ms"], 0.021, places=6)
        self.assertAlmostEqual(coverage["share_of_trace"], 0.021 / 0.121, places=6)

    def test_forward_backward_split(self) -> None:
        regions = self.report["regions"]
        self.assertEqual(regions["forward_calls"], 1)
        self.assertEqual(regions["backward_calls"], 1)
        self.assertAlmostEqual(regions["forward_ms"], 0.007, places=6)
        self.assertAlmostEqual(regions["backward_ms"], 0.011, places=6)
        self.assertAlmostEqual(regions["other_ms"], 0.003, places=6)

    def test_operator_rows_carry_region_and_shape(self) -> None:
        rows = self.report["ops"]
        self.assertEqual([row["op"] for row in rows], ["mul"])
        row = rows[0]
        self.assertEqual(row["family"], "elementwise")
        self.assertEqual(row["calls"], 3)
        self.assertAlmostEqual(row["ms"], 0.021, places=6)
        self.assertAlmostEqual(row["share"], 1.0, places=6)
        self.assertIn("181838, 10, 64", row["top_shape"])

    def test_families_cover_the_scope(self) -> None:
        families = self.report["families"]
        self.assertEqual([row["family"] for row in families], ["elementwise"])
        self.assertAlmostEqual(sum(row["share"] for row in families), 1.0, places=6)

    def test_rows_are_kept_beyond_the_print_limit(self) -> None:
        # The JSON must not be truncated by --top: a caller looking up one
        # operator cannot be defeated by the console table length.
        self.assertEqual(len(self.report["ops"]), 1)
        self.assertEqual(len(self.report["shapes"]), 1)

    def test_unknown_module_lists_available_scopes(self) -> None:
        with self.assertRaises(KeyError) as caught:
            op_time_share.analyze(self.trace, module="nn.Module: Nope_0")
        self.assertIn(MODULE, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
