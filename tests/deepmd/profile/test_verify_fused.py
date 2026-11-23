"""Tests for the fused-operator validation ladder runner.

Everything here runs on a machine with no DCU: the plan is pure construction and
the verdict takes two summaries, so a broken unittest pattern or an inverted
criterion fails here instead of on the cluster.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_fused  # noqa: E402


def _summary(*, device_us: float, launches: int, hits: dict | None = None,
             fallbacks: dict | None = None, last_error: str = "") -> dict:
    return {
        "device_event_count": launches,
        "top_profiler_events": [
            {"name": "dpa4/inference.single_energy_force", "self_device_us": device_us},
        ],
        "fused": {
            **(hits if hits is not None else {"pack_hits": 9}),
            **(fallbacks if fallbacks is not None else {"pack_fallbacks": 0}),
            "last_error": last_error,
        },
    }


class PlanTests(unittest.TestCase):
    def test_host_stage_runs_the_python_contracts(self) -> None:
        plan = verify_fused.build_plan()
        (command, extra, _label), = plan["host"]
        self.assertIn("tests/python", command)
        self.assertEqual(extra, {})

    def test_latency_stage_sets_the_opt_in_gates(self) -> None:
        plan = verify_fused.build_plan(out_dir=Path("/tmp/v"))
        environ = [extra for _command, extra, _label in plan["lat"]]
        self.assertIn("DPA4_RUN_SEGMENT_LATENCY", environ[0])
        self.assertIn("DPA4_RUN_SLOT_LATENCY", environ[1])
        self.assertEqual(environ[0]["DPA4_SEGMENT_LATENCY_JSON"], "/tmp/v/seg_latency.json")

    def test_end_to_end_needs_a_model_and_an_input(self) -> None:
        self.assertEqual(verify_fused.build_plan()["e2e"], [])
        plan = verify_fused.build_plan(model_file="m.pt", input_npz="i.npz")
        baseline, fused = plan["e2e"]
        self.assertIn("--fused", fused[0])
        self.assertNotIn("--fused", baseline[0])
        self.assertEqual(baseline[0][baseline[0].index("--output-dir") + 1], "/root/private_data/lipengfei/test_result/verify/off")

    def test_dry_run_prints_without_spawning(self) -> None:
        calls: list[list[str]] = []

        def explode(command, **kwargs):  # pragma: no cover - must not run
            calls.append(command)
            raise AssertionError("dry run must not spawn a process")

        original = verify_fused.subprocess.call
        verify_fused.subprocess.call = explode
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                code = verify_fused.main(
                    ["--dry-run", "--stage", "acc", "--stage", "par"]
                )
        finally:
            verify_fused.subprocess.call = original
        self.assertIn("test_segment_ops*", buffer.getvalue())
        self.assertIn("DP_RUN_FUSED_DEEPMD_PARITY=1", buffer.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])


class JudgeTests(unittest.TestCase):
    def test_every_criterion_met_passes(self) -> None:
        ok, messages = verify_fused.judge(
            _summary(device_us=1_218_600.0, launches=2614),
            _summary(device_us=1_150_000.0, launches=2400,
                     hits={"pack_hits": 18, "aggregate_hits": 3}),
        )
        self.assertTrue(ok, messages)
        self.assertTrue(any(m.startswith("ok   device ") for m in messages))

    def test_no_seam_dispatched_fails(self) -> None:
        ok, messages = verify_fused.judge(
            _summary(device_us=1_218_600.0, launches=2614),
            _summary(device_us=1_100_000.0, launches=2400, hits={"pack_hits": 0}),
        )
        self.assertFalse(ok)
        self.assertTrue(any("no seam dispatched" in m for m in messages))

    def test_fallback_fails_even_when_time_drops(self) -> None:
        ok, messages = verify_fused.judge(
            _summary(device_us=1_218_600.0, launches=2614),
            _summary(device_us=1_100_000.0, launches=2400,
                     fallbacks={"pack_fallbacks": 4}),
        )
        self.assertFalse(ok)
        self.assertTrue(any("fallbacks" in m for m in messages))

    def test_unchanged_device_time_fails(self) -> None:
        # The B2 lesson: fewer launches and real hits, but the pool was empty.
        ok, messages = verify_fused.judge(
            _summary(device_us=964_400.0, launches=2614),
            _summary(device_us=963_800.0, launches=2464),
        )
        self.assertTrue(ok, messages)  # 0.6 ms is still a drop
        ok, messages = verify_fused.judge(
            _summary(device_us=964_400.0, launches=2614),
            _summary(device_us=964_400.0, launches=2464),
        )
        self.assertFalse(ok)
        self.assertTrue(any("device time did not drop" in m for m in messages))

    def test_missing_device_total_fails(self) -> None:
        ok, messages = verify_fused.judge(
            {"device_event_count": 1, "fused": {"pack_hits": 1}},
            _summary(device_us=1.0, launches=1),
        )
        self.assertFalse(ok)
        self.assertTrue(any("no device total" in m for m in messages))


if __name__ == "__main__":
    unittest.main()
