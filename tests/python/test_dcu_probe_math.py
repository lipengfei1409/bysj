"""Host tests for the FFMA probe's projection arithmetic.

The probe itself needs a DCU, but the decision it feeds (does the fused
convolution's float32 arithmetic fit inside the dense segment's device time?)
is arithmetic, and a wrong constant here would quietly green-light a 24-file
port.  ``tests/dcu/fma_probe.py`` imports torch lazily, so the module loads on
any host.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests" / "dcu" / "fma_probe.py"

_spec = importlib.util.spec_from_file_location("dcu_fma_probe", PROBE)
assert _spec is not None and _spec.loader is not None
fma_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fma_probe)


class ProjectionTests(unittest.TestCase):
    def test_air_shape_constants_match_the_loaded_model(self) -> None:
        # Read off the checkpoint: lmax=3, Cf=64, n_focus=1, mixing_layers=4,
        # 3 interaction blocks.  GATE = 192 = lmax*Cf, M0 = 256, M1 = 384.
        self.assertEqual(fma_probe.AIR_PER_LAYER_FMA, 225_280)
        self.assertEqual(fma_probe.AIR_EDGES, 181838)
        self.assertEqual(fma_probe.AIR_LAYERS, 4)
        self.assertEqual(fma_probe.AIR_BLOCKS, 3)
        self.assertEqual(fma_probe.AIR_N_FOCUS, 1)

    def test_per_edge_components_add_up(self) -> None:
        components = fma_probe.air_per_edge_fma()
        self.assertEqual(components["gated_layers"], 3 * 225_280)
        self.assertEqual(components["identity_layer"], 212_992)
        self.assertEqual(components["rotations"], 20_480)
        self.assertEqual(components["radial_mix"], 6_400)
        self.assertEqual(components["total"], 915_712)

    def test_gemm_only_flop_matches_the_trace(self) -> None:
        # 24 bmm calls of 384x384 and 24 of 256x256 in the trace, over 3 blocks
        # and 4 layers: 2*E*(384^2+256^2)*4*3 = 9.30e11 FLOP.  The full per-edge
        # total (with gate, rotations, radial) must sit within 10% of it.
        gemm_flop = 2 * 181838 * (384**2 + 256**2) * 4 * 3
        total_flop = fma_probe.projection(18.1)["flop"]
        self.assertAlmostEqual(total_flop / gemm_flop, 1.0, delta=0.10)
        self.assertAlmostEqual(fma_probe.DENSE_FORWARD_MS, 163.8)
        self.assertAlmostEqual(fma_probe.DENSE_BACKWARD_MS, 416.2)

    def test_projection_is_linear_in_the_measured_rate(self) -> None:
        slow = fma_probe.projection(10.0)
        fast = fma_probe.projection(20.0)
        self.assertAlmostEqual(slow["projected_ms"] / fast["projected_ms"], 2.0, places=6)
        # Twice the rate halves the projected time, which doubles the speedup.
        self.assertAlmostEqual(fast["speedup_over_dense"],
                               2.0 * slow["speedup_over_dense"], places=6)

    def test_flop_count(self) -> None:
        report = fma_probe.projection(15.0)
        expected = 2.0 * 915_712 * 181838 * 3
        self.assertAlmostEqual(report["flop"], expected, places=0)
        self.assertAlmostEqual(
            report["projected_ms"], expected / (15.0e12) * 1e3, places=6
        )

    def test_ridge_and_budget_decline_the_air_shape(self) -> None:
        # 15 TFLOP/s over 630 GB/s is a ridge of ~24 FLOP/byte, a third of the
        # calibration part's 73.2, so the admissible layer shrinks accordingly.
        report = fma_probe.projection(15.0)
        self.assertAlmostEqual(report["ridge_flop_per_byte"], 15e12 / 630e9, places=6)
        self.assertLess(report["profitable_budget_fma"], 65536)
        self.assertGreater(report["per_layer_fma"], report["profitable_budget_fma"])

    def test_rejects_a_nonpositive_rate(self) -> None:
        with self.assertRaises(ValueError):
            fma_probe.projection(0.0)


if __name__ == "__main__":
    unittest.main()
