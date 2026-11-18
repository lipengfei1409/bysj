"""Host-side tests for the trace attribution helper."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "dpa4_analyze_trace", Path(__file__).with_name("analyze_trace.py")
)
trace_analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(trace_analysis)


def _write_trace(tmp: Path, events) -> Path:
    path = tmp / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return path


def _kernel(name, dur, cat="kernel"):
    return {"ph": "X", "cat": cat, "name": name, "dur": dur}


class TraceAnalysis(unittest.TestCase):
    def test_families_separate_triton_from_aten_and_gemm(self):
        events = [
            _kernel("_stack_gate_kernel", 300.0),
            _kernel(
                "void at::native::vectorized_elementwise_kernel<4, ...>(int)", 500.0
            ),
            _kernel("Cijk_Ailk_Bljk_SB_MT128x64x16_ISA928", 700.0),
            _kernel("rocprim::detail::radix_sort_block_sort_kernel", 40.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = trace_analysis.analyze(
                _write_trace(Path(tmp), events), use_registry=False
            )
        self.assertEqual(report["families"]["triton"]["calls"], 1)
        self.assertEqual(report["families"]["gemm"]["calls"], 1)
        self.assertEqual(report["families"]["aten"]["calls"], 2)
        self.assertEqual(report["triton_kernel_calls"], 1)
        self.assertEqual(report["device_kernel_calls"], 4)

    def test_aten_global_kernel_is_not_reported_as_triton(self):
        # ``sinc_vectorized4_kernel`` is an ATen global whose name ends in
        # ``_kernel``; the classifier must not count it as a Triton kernel.
        events = [_kernel("sinc_vectorized4_kernel", 12.0)]
        with tempfile.TemporaryDirectory() as tmp:
            report = trace_analysis.analyze(
                _write_trace(Path(tmp), events), use_registry=False
            )
        self.assertEqual(report["triton_kernel_calls"], 0)
        self.assertEqual(report["families"]["other"]["calls"], 1)

    def test_registry_names_win_over_name_shape(self):
        # ``activation`` is a Triton kernel without a ``_kernel`` suffix, so the
        # name-shape fallback cannot see it; the DeepMD registry can.
        events = [_kernel("activation", 20.0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_trace(Path(tmp), events)
            self.assertEqual(
                trace_analysis.analyze(path, use_registry=False)["triton_kernel_calls"],
                0,
            )
            known = {"activation"}
            report = trace_analysis.analyze(path, use_registry=False)
            self.assertEqual(report["families"]["other"]["calls"], 1)
        self.assertEqual(trace_analysis._classify("activation", known), "triton")
        self.assertEqual(trace_analysis._classify("activation", set()), "other")

    def test_roofline_columns_follow_the_kernel_family(self):
        events = [
            _kernel("_stack_gate_kernel", 1000.0),
            _kernel("Cijk_Ailk_Bljk_SB_MT128x64x16", 1000.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = trace_analysis.analyze(
                _write_trace(Path(tmp), events),
                use_registry=False,
                bandwidth_gbps=1000.0,
                peak_tflops=10.0,
            )
        rows = {row["family"]: row for row in report["top_kernels"]}
        self.assertAlmostEqual(rows["triton"]["equiv_mb"], 1000.0, places=3)
        self.assertAlmostEqual(rows["gemm"]["equiv_gflop"], 10.0, places=3)
        self.assertNotIn("equiv_mb", rows["gemm"])

    def test_cast_report_groups_by_dtype_and_dimensions(self):
        def cast(dur, dtype, dims):
            return {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::to",
                "dur": dur,
                "args": {"Input type": [dtype, "Scalar"], "Input Dims": [dims, []]},
            }

        events = [
            cast(10.0, "double", [1, 3000, 1, 1]),
            cast(20.0, "double", [1, 3000, 1, 1]),
            cast(5.0, "float", [181838, 64]),
            _kernel("_stack_gate_kernel", 100.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = trace_analysis.analyze(
                _write_trace(Path(tmp), events), use_registry=False
            )
        top = report["cast_report"][0]
        self.assertEqual(top["calls"], 2)
        self.assertAlmostEqual(top["duration_us"], 30.0)
        self.assertAlmostEqual(top["max_us"], 20.0)
        self.assertIn("double", top["signature"])

    def test_trace_without_device_kernels_is_rejected(self):
        events = [{"ph": "X", "cat": "cpu_op", "name": "aten::to", "dur": 1.0}]
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(RuntimeError, "no device kernel"),
        ):
            trace_analysis.analyze(_write_trace(Path(tmp), events), use_registry=False)

    def test_cli_writes_json_and_reports_missing_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = _write_trace(Path(tmp), [_kernel("_stack_gate_kernel", 42.0)])
            out = Path(tmp) / "report.json"
            self.assertEqual(
                trace_analysis.main([str(trace), "--json", str(out), "--no-registry"]),
                0,
            )
            self.assertEqual(json.loads(out.read_text())["device_kernel_calls"], 1)
            self.assertEqual(trace_analysis.main([str(Path(tmp) / "missing.json")]), 2)


if __name__ == "__main__":
    unittest.main()
