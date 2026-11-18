"""Host tests for the fused-run comparison tool used by the 3000-atom profile."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "deepmd" / "profile" / "compare_fused_runs.py"
_spec = importlib.util.spec_from_file_location("compare_fused_runs", _MODULE_PATH)
compare_fused_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_fused_runs)


def _summary(device_ms, launches, kernels=None, fused=None):
    return {
        "model_file": "/models/dpa4.pt",
        "backend": "pt",
        "device_name": "K500SM_AI",
        "atom_count": 3000,
        "input_source": "/data/water3000.npz",
        "fused": fused or {},
        "top_profiler_events": [
            {"name": "dpa4/inference.single_energy_force", "calls": 1,
             "self_device_us": device_ms * 1000.0}
        ],
        "trace_analysis": {
            "device_kernel_calls": launches,
            "top_device_kernels": [
                {"name": name, "calls": 1, "duration_us": duration_ms * 1000.0}
                for name, duration_ms in (kernels or {}).items()
            ],
        },
    }


class CompareTests(unittest.TestCase):
    def test_reports_speedup_launches_and_kernel_deltas(self):
        baseline = _summary(
            1218.584,
            2614,
            {"at::native::indexFuncSmallIndex": 41.0, "at::native::vectorized_elementwise_kernel": 144.5},
        )
        patched = _summary(
            1100.0,
            2400,
            {"at::native::indexFuncSmallIndex": 5.0, "at::native::vectorized_elementwise_kernel": 144.5},
            fused={"softmax_hits": 3, "softmax_fallbacks": 0},
        )
        report = compare_fused_runs.compare(baseline, patched)
        self.assertAlmostEqual(report["baseline_ms"], 1218.584)
        self.assertAlmostEqual(report["patched_ms"], 1100.0)
        self.assertAlmostEqual(report["speedup"], 1218.584 / 1100.0)
        self.assertEqual(report["baseline_launches"], 2614)
        self.assertEqual(report["patched_launches"], 2400)
        self.assertEqual(report["kernel_deltas"][0]["delta_ms"], -36.0)
        self.assertTrue(all(report["config_matches"].values()))

    def test_device_time_falls_back_to_the_kernel_sum(self):
        summary = _summary(0.0, 10, {"a": 2.0, "b": 3.0})
        summary.pop("top_profiler_events")
        self.assertAlmostEqual(compare_fused_runs._device_ms(summary), 5.0)

    def test_render_names_the_seams_and_flags_a_mismatch(self):
        baseline = _summary(1218.0, 2614, {"k": 10.0})
        patched = _summary(1100.0, 2400, {"k": 5.0}, fused={"softmax_hits": 3})
        patched["atom_count"] = 4500
        text = compare_fused_runs.render(compare_fused_runs.compare(baseline, patched))
        self.assertIn("1.107x", text)
        self.assertIn("softmax=3", text)
        self.assertIn("NOT comparable", text)

    def test_cli_writes_the_json_report(self):
        baseline = _summary(1218.0, 2614, {"k": 10.0})
        patched = _summary(1100.0, 2400, {"k": 4.0}, fused={"softmax_hits": 3})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, patched_path = root / "base.json", root / "patched.json"
            base_path.write_text(json.dumps(baseline))
            patched_path.write_text(json.dumps(patched))
            out_path = root / "compare.json"
            code = compare_fused_runs.main(
                ["--baseline", str(base_path), "--patched", str(patched_path), "--json", str(out_path)]
            )
            self.assertEqual(code, 0)
            report = json.loads(out_path.read_text())
            self.assertAlmostEqual(report["speedup"], 1218.0 / 1100.0)


if __name__ == "__main__":
    unittest.main()
