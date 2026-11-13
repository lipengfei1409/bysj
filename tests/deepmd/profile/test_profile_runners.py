"""Exercise orchestration on the host; mock only device/runtime boundaries."""
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

if __package__:
    from .test_profile_contracts import cli
else:
    from test_profile_contracts import cli


def write_trace(path):
    Path(path).write_text(json.dumps({"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "test_kernel", "dur": 10},
    ]}))


class ProfileRunners(unittest.TestCase):
    def test_direct_records_one_eval_after_five_warmups(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            model_path = root / "model.pt"
            model_path.touch()
            np.savez(root / "system.npz", coords=np.zeros((2, 3)), atom_types=np.zeros(2, dtype=np.int32))
            args = cli.build_parser().parse_args([
                "dpa4", "--model-file", str(model_path), "--input-npz", str(root / "system.npz"),
                "--output-dir", str(root / "out"), "--fixed-input",
            ])
            args.cli_command = ["test"]
            model = Mock()
            backend_type = type("DeepEval", (), {"__module__": "deepmd.pt.infer.deep_eval"})
            model.deep_eval = backend_type()
            model.get_type_map.return_value = ["O", "H"]
            model.eval.return_value = (np.zeros(1), np.zeros((2, 3)), np.zeros(9))
            profiler = MagicMock()
            profiler.__enter__.return_value = profiler
            profiler.key_averages.return_value = [SimpleNamespace(
                key="dpa4/inference.single_energy_force", count=1, self_device_time_total=10,
                self_cpu_time_total=10, cpu_time_total=10,
            )]
            profiler.export_chrome_trace.side_effect = write_trace
            stack.enter_context(patch.object(cli, "_construct_deepmd_model", return_value=model))
            stack.enter_context(patch.object(cli.torch.profiler, "profile", return_value=profiler))
            stack.enter_context(patch.object(cli.torch.version, "hip", "test"))
            for method, value in {"is_available": True, "device_count": 1, "get_device_name": "K500SM_AI", "memory_allocated": 0, "max_memory_allocated": 1024}.items():
                stack.enter_context(patch.object(cli.torch.cuda, method, return_value=value))
            for method in ("reset_peak_memory_stats", "synchronize", "set_device"):
                stack.enter_context(patch.object(cli.torch.cuda, method))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(cli._run_dpa4(args), 0)
            self.assertEqual(model.eval.call_count, 6)
            model.close.assert_called_once()
            report = json.loads(next((root / "out").rglob("*summary.json")).read_text())
            self.assertEqual(report["backend"], "pt")
            self.assertEqual(report["operator_source"], "original DeepMD")
            self.assertNotIn("native_ops", report)
            self.assertEqual(report["profiled_inference_calls"], 1)
            self.assertEqual(report["warmup_calls"], 5)
            self.assertEqual(report["atom_count"], 2)
            self.assertEqual(report["trace_analysis"]["device_kernel_calls"], 1)

    def test_lammps_single_and_multi_rank_collect_complete_windows(self):
        for ranks in (1, 2):
            with self.subTest(ranks=ranks), tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
                root = Path(tmp)
                model = root / "model.pt2"
                with zipfile.ZipFile(model, "w") as archive:
                    archive.writestr("metadata.json", '{"has_comm_artifact": true}')
                (root / "water.data").write_text("2 atoms\n")
                executable = root / "lmp"
                executable.touch()
                args = cli.build_parser().parse_args([
                    "lammps", "--model-file", str(model), "--data-file", str(root / "water.data"),
                    "--lammps-input", str(Path(__file__).with_name("dpa4_water_profile.template")),
                    "--lammps-exe", str(executable), "--mpi-exe", str(executable),
                    "--gpu-ids", ",".join(map(str, range(ranks))), "--world-size", str(ranks),
                    "--output-dir", str(root / "out"),
                ])
                args.cli_command = ["test"]
                def simulate(command, *, env, cwd, **kwargs):
                    self.assertEqual(env["WORLD_SIZE"], str(ranks))
                    self.assertEqual(env["DP_PROFILER_SKIP"], "5")
                    self.assertEqual(env["DP_PROFILER_ACTIVE"], "10")
                    self.assertTrue((Path(cwd) / "profile.in").is_file())
                    for rank in range(ranks):
                        suffix = f"_rank{rank}" if ranks > 1 else ""
                        trace = Path(env["DPA4_DP_PROFILER_STEM"] + suffix + f"_gpu{rank}.json")
                        write_trace(trace)
                        Path(str(trace) + ".summary.json").write_text(json.dumps({
                            "complete": True, "aborted": False, "captured_calls": 10,
                            "skip_calls": 5, "active_calls": 10,
                        }))
                    return SimpleNamespace(returncode=0)
                stack.enter_context(patch.object(cli.subprocess, "run", side_effect=simulate))
                stack.enter_context(patch.object(cli.torch.version, "hip", "test"))
                stack.enter_context(patch.object(cli.torch.cuda, "is_available", return_value=True))
                stack.enter_context(patch.object(cli.torch.cuda, "device_count", return_value=ranks))
                stack.enter_context(patch.dict(cli.os.environ, {}, clear=True))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                self.assertEqual(cli._run_lammps(args), 0)
                report = json.loads(next((root / "out").rglob("profile_summary.json")).read_text())
                self.assertTrue(report["profile_valid"])
                self.assertEqual(report["world_size"], ranks)
                self.assertEqual(len(report["rank_profiles"]), ranks)
