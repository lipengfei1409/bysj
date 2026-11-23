"""Host tests for the A/B gain reader."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ab_report  # noqa: E402


def _write(root: Path, tag: str, *, device_ms: float, conv_ms: float,
           launches: int = 1860, enabled: bool = True) -> None:
    run = root / tag / "dpa4_1700000000"
    run.mkdir(parents=True)
    summary = {
        "top_profiler_events": [
            {"name": "dpa4/inference.single_energy_force", "self_device_us": device_ms * 1e3}
        ],
        "device_event_count": launches,
        "trace_analysis": {
            "top_device_kernels": [
                {"name": "void dpa4::so2_conv_fwd_kernel<3, 64, ...>", "duration_us": conv_ms * 5e2},
                {"name": "void dpa4::so2_conv_bwd_kernel<3, 64, ...>", "duration_us": conv_ms * 5e2},
                {"name": "vectorized_elementwise_kernel", "duration_us": 14000.0},
            ]
        },
        "so2_conv": {"enabled": enabled, "reason": "" if enabled else "no visible DCU"},
    }
    (run / "dpa4_inference_profile_summary.json").write_text(json.dumps(summary))


class AbReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_net_win_is_reported_with_the_factor(self) -> None:
        _write(self.root, "base", device_ms=957.0, conv_ms=0.0)
        _write(self.root, "kc3", device_ms=880.0, conv_ms=620.0)
        lines = ab_report.verdict(ab_report.report(self.root, ["base", "kc3"]))
        self.assertTrue(any("net win" in line for line in lines), lines)
        self.assertFalse(any("cannot pay for itself" in line for line in lines), lines)

    def test_a_fast_conv_that_does_not_move_the_total_is_flagged(self) -> None:
        _write(self.root, "base", device_ms=957.0, conv_ms=0.0)
        _write(self.root, "kc3", device_ms=956.0, conv_ms=700.0)
        lines = ab_report.verdict(ab_report.report(self.root, ["base", "kc3"]))
        self.assertTrue(any("noise floor" in line for line in lines), lines)
        self.assertTrue(any("cannot pay for itself" in line for line in lines), lines)

    def test_routing_off_is_not_read_as_no_gain(self) -> None:
        _write(self.root, "base", device_ms=957.0, conv_ms=0.0)
        _write(self.root, "kc3", device_ms=957.0, conv_ms=0.0, enabled=False)
        lines = ab_report.verdict(ab_report.report(self.root, ["base", "kc3"]))
        self.assertTrue(any("routed OFF" in line for line in lines), lines)

    def test_missing_run_is_an_error_and_names_the_expected_path(self) -> None:
        import contextlib
        import io

        _write(self.root, "base", device_ms=957.0, conv_ms=0.0)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = ab_report.main([str(self.root), "base", "nope"])
        self.assertEqual(code, 1)
        self.assertIn("nope/dpa4_*/dpa4_inference_profile_summary.json", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
