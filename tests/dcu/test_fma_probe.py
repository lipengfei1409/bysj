"""Measure the DCU's sustained FFMA rate and project the fused SO(2) convolution.

Run on a DCU with::

    python tests/dcu/test_fma_probe.py -v

The number this prints decides whether porting ``deepmd::dpa4_so2_conv`` to HIP
is worth its 24 translation units: the fused operator replaces the whole
per-edge span (163.8 ms forward / 416.2 ms backward in our trace) but pays for
it in float32 SIMT multiply-adds, and 225.3 kFMA per edge per layer is what
upstream's own guard declines.  See docs/execution-model.md 4.5.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

import fma_probe  # noqa: E402


@unittest.skipUnless(
    fma_probe.available(),
    "the FFMA probe needs HIP-enabled PyTorch and a visible DCU",
)
class FmaProbeTests(unittest.TestCase):
    def test_sustained_ffma_rate_and_projection(self) -> None:
        best = 0.0
        for nacc in (8, 16):
            rate = fma_probe.measure(nacc=nacc)
            best = max(best, rate)
            print(f"nacc={nacc:2d}: {rate:.1f} TFLOP/s")
        self.assertGreater(best, 0.1, "the probe measured no throughput at all")

        report = fma_probe.projection(best)
        print(
            f"\nprojected fused conv fwd: {report['projected_ms']:.1f} ms "
            f"vs dense {report['dense_ms']:.1f} ms "
            f"-> {report['speedup_over_dense']:.2f}x"
        )
        print(
            f"ridge: {report['ridge_flop_per_byte']:.1f} FLOP/byte; "
            f"_profitable budget {report['profitable_budget_fma']:.0f} FMA "
            f"vs {report['per_layer_fma']} required"
        )
        if report["projected_ms"] < report["dense_ms"]:
            print("VERDICT: arithmetic fits -- port one specialization and A/B it")
        else:
            print("VERDICT: arithmetic does not fit -- do not port, take the C6 route")


if __name__ == "__main__":
    unittest.main()
