"""Host-side regression tests for the three profiling scenarios."""
import importlib.util
import json
import tempfile
import unittest
import sys
from types import ModuleType
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "dpa4_profile_cli", Path(__file__).with_name("test_dpa4_inference_profile.py")
)
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


class ProfileContracts(unittest.TestCase):
    def test_cli_defaults_select_fixed_input_and_warmup(self):
        args = cli.build_parser().parse_args([
            "dpa4", "--model-file", "m.pt", "--input-npz", "s.npz",
            "--output-dir", "out",
        ])
        self.assertTrue(getattr(args, "auto_size", False))
        self.assertEqual(args.memory_fraction, 0.70)
        self.assertEqual(getattr(args, "warmup", None), 5)
        self.assertEqual(cli._PROJECT_ROOT, Path(__file__).resolve().parents[3])

    def test_fixed_input_never_runs_sizing(self):
        source = (np.zeros((1, 2, 3)), np.zeros((1, 2), dtype=np.int32), np.eye(3).reshape(1, 9))
        args = SimpleNamespace(auto_size=False, max_atoms=0)
        with patch.object(cli, "_choose_large_system", side_effect=AssertionError("sizing")):
            result, replicas, report = cli._prepare_direct_system(None, source, args)
        self.assertIs(result, source)
        self.assertEqual(replicas, (1, 1, 1))
        self.assertEqual(report["selected_atoms"], 2)
        args.max_atoms = 1
        with self.assertRaisesRegex(ValueError, "max-atoms"):
            cli._prepare_direct_system(None, source, args)

    def test_explicit_warmup_finishes_before_capture(self):
        model = Mock()
        model.eval.return_value = (np.zeros(1), np.zeros((1, 3)), np.zeros(9))
        system = ("coords", "types", "cell")
        with patch.object(cli.torch.cuda, "synchronize") as sync:
            cli._warmup_model(model, system, "cuda:0", 5)
        self.assertEqual(model.eval.call_count, 5)
        model.eval.assert_called_with("coords", "cell", "types")
        sync.assert_called_once_with("cuda:0")

    def test_trace_requires_actual_kernels_and_summarizes_by_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "rank0.json"
            trace.write_text(json.dumps({"traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "aten::mul", "dur": 20},
                {"ph": "X", "cat": "kernel", "name": "original_deepmd_kernel", "dur": 4},
                {"ph": "X", "cat": "kernel", "name": "original_deepmd_kernel", "dur": 6},
            ]}))
            result = cli._summarize_trace(trace)
            self.assertEqual(result["device_kernel_calls"], 2)
            self.assertEqual(result["top_device_kernels"][0]["duration_us"], 10)
            self.assertNotIn("native_execution", result)
            trace.write_text('{"traceEvents": []}')
            with self.assertRaisesRegex(RuntimeError, "device kernel"):
                cli._summarize_trace(trace)

    def test_lammps_checks_each_rank_and_completed_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "profile"
            trace = Path(f"{stem}_rank0_gpu0.json")
            trace.write_text(json.dumps({"traceEvents": [
                {"ph": "X", "cat": "kernel", "name": "compute", "dur": 10}
            ]}))
            Path(str(trace) + ".summary.json").write_text(json.dumps({
                "complete": True, "aborted": False, "captured_calls": 10,
                "skip_calls": 5, "active_calls": 10,
            }))
            result = cli._collect_lammps_profiles(stem, 1, 5, 10)
            self.assertEqual(len(result), 1)
            with self.assertRaisesRegex(RuntimeError, "rank 1"):
                cli._collect_lammps_profiles(stem, 2, 5, 10)
            Path(str(trace) + ".summary.json").write_text('{"complete": false}')
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                cli._collect_lammps_profiles(stem, 1, 5, 10)

    def test_lammps_defaults_are_real_md_with_bounded_capture(self):
        args = cli.build_parser().parse_args([
            "lammps", "--model-file", "m.pt2", "--data-file", "s.data",
            "--lammps-input", "in", "--lammps-exe", "lmp", "--gpu-ids", "0",
            "--output-dir", "out",
        ])
        self.assertEqual(args.steps, 20)
        self.assertEqual(getattr(args, "profile_skip", None), 5)
        self.assertEqual(getattr(args, "profile_calls", None), 10)
        template = Path(__file__).with_name("dpa4_water_profile.template").read_text()
        self.assertIn("fix integrate all nve", template)
        self.assertIn("timestep 0.25", template)
        self.assertIn("velocity all create 300.0 20261113", template)
        self.assertIn("dump trajectory all custom 1 trajectory.lammpstrj id type x y z", template)

    def test_geometry_styles_and_image_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "system.data"
            for style, row in [
                ("atomic", "1 2 0.2 0.3 0.4 1 -1 2"),
                ("charge", "1 2 -0.8 0.2 0.3 0.4 1 -1 2"),
                ("full", "1 7 2 -0.8 0.2 0.3 0.4 1 -1 2"),
            ]:
                with self.subTest(style=style):
                    path.write_text(f"1 atoms\n0 3 xlo xhi\n0 4 ylo yhi\n0 5 zlo zhi\n\nAtoms # {style}\n\n{row}\n")
                    coords, types, _ = cli._read_lammps_system(path)
                    np.testing.assert_allclose(coords[0, 0], [0.2, 0.3, 0.4])
                    self.assertEqual(types[0, 0], 1)
            path.write_text("1 atoms\n0 3 xlo xhi\n0 4 ylo yhi\n0 5 zlo zhi\n\nAtoms # sphere\n\n1 2 1 1 0 0 0\n")
            with self.assertRaisesRegex(ValueError, "atom style"):
                cli._read_lammps_system(path)

    def test_world_size_matches_gpu_count(self):
        args = cli.build_parser().parse_args([
            "lammps", "--model-file", "m.pt2", "--data-file", "s.data",
            "--lammps-input", "in", "--lammps-exe", "lmp", "--gpu-ids", "0,1",
            "--world-size", "2", "--output-dir", "out",
        ])
        self.assertEqual(cli._lammps_rank_count(args), 2)
        args.world_size = 4
        with self.assertRaisesRegex(ValueError, "world-size"):
            cli._lammps_rank_count(args)

    def test_gpu_selection_preserves_existing_allocation(self):
        self.assertEqual(cli._gpu_visibility((0, 1), {"HIP_VISIBLE_DEVICES": "4,5"}), "4,5")
        self.assertEqual(cli._gpu_visibility((1,), {"CUDA_VISIBLE_DEVICES": "4,5"}), "5")
        self.assertEqual(cli._gpu_visibility((0, 2), {}), "0,2")
        with self.assertRaisesRegex(ValueError, "visibility"):
            cli._gpu_visibility((0,), {"HIP_VISIBLE_DEVICES": "4,5", "CUDA_VISIBLE_DEVICES": "0,1"})

    def test_autosizing_rejects_successful_probe_over_memory_budget(self):
        source = (np.zeros((1, 1, 3)), np.zeros((1, 1), dtype=np.int32), np.eye(3).reshape(1, 9))
        args = SimpleNamespace(device="cuda:0", max_atoms=10, memory_fraction=0.70)
        with patch.object(cli.torch.cuda, "mem_get_info", return_value=(1000000, 1000000)), \
             patch.object(cli.torch.cuda, "empty_cache"), \
             patch.object(cli, "_memory_delta_for_inference", side_effect=[65536, 900000, 500000]), \
             patch.object(cli, "_MAX_SIZING_PROBES", 2):
            selected, _, report = cli._choose_large_system(Mock(), source, args)
        self.assertEqual(selected[0].shape[1], 7)
        self.assertLessEqual(report["selected_peak_increment_bytes"], report["memory_budget_bytes"])
        self.assertEqual(report["sizing_probes"][1]["status"], "over_budget")

    def test_autosizing_does_not_hide_operator_errors(self):
        source = (np.zeros((1, 1, 3)), np.zeros((1, 1), dtype=np.int32), np.eye(3).reshape(1, 9))
        args = SimpleNamespace(device="cuda:0", max_atoms=10, memory_fraction=0.70)
        with patch.object(cli.torch.cuda, "mem_get_info", return_value=(1000000, 1000000)), \
             patch.object(cli.torch.cuda, "empty_cache"), \
             patch.object(cli, "_memory_delta_for_inference", side_effect=[65536, ValueError("operator bug")]):
            with self.assertRaisesRegex(ValueError, "operator bug"):
                cli._choose_large_system(Mock(), source, args)

    def test_direct_constructor_uses_original_deepmd_for_both_backends(self):
        args = SimpleNamespace(device="cuda:1", model_file=Path("model.pt"))
        for backend in ("pt", "pt_expt"):
            with self.subTest(backend=backend):
                backend_type = type("DeepEval", (), {"__module__": f"deepmd.{backend}.infer.deep_eval"})
                expected = SimpleNamespace(deep_eval=backend_type())
                def create(path, **kwargs):
                    self.assertEqual(path, "model.pt")
                    self.assertEqual(kwargs, {"auto_batch_size": False})
                    self.assertEqual(cli.os.environ["DEVICE"], "cuda")
                    self.assertEqual(cli.os.environ["LOCAL_RANK"], "1")
                    return expected
                package = ModuleType("deepmd")
                infer = ModuleType("deepmd.infer")
                infer.DeepPot = Mock(side_effect=create)
                with patch.dict(sys.modules, {"deepmd": package, "deepmd.infer": infer}), \
                     patch.object(cli, "_loaded_deepmd_device_modules", return_value=[]), \
                     patch.dict(cli.os.environ, {"DEVICE": "cpu", "LOCAL_RANK": "7"}):
                    actual = cli._construct_deepmd_model(args)
                    self.assertIs(actual, expected)
                    self.assertEqual(cli._deepmd_backend_name(actual), backend)
                    self.assertEqual(cli.os.environ["DEVICE"], "cpu")
                    self.assertEqual(cli.os.environ["LOCAL_RANK"], "7")
                infer.DeepPot.assert_called_once()

    def test_direct_cli_does_not_offer_native_patches(self):
        args = cli.build_parser().parse_args([
            "dpa4", "--model-file", "m.pt", "--input-npz", "s.npz",
            "--output-dir", "out",
        ])
        self.assertFalse(hasattr(args, "native_ops"))

    def test_water_metadata_remaps_types_for_multi_element_model(self):
        model = Mock()
        model.get_type_map.return_value = ["H", "C", "N", "O"]
        source = (np.zeros((1, 3, 3)), np.asarray([[0, 1, 1]], dtype=np.int32), np.eye(3).reshape(1, 9))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "water.npz"
            np.savez(path, coords=source[0], atom_types=source[1], cells=source[2])
            path.with_suffix(".metadata.json").write_text('{"type_map": ["O", "H"]}')
            args = SimpleNamespace(input_npz=path, input_type_map=None)
            result, report = cli._remap_input_types(model, source, args)
            np.testing.assert_array_equal(result[1], [[3, 0, 0]])
            np.testing.assert_array_equal(source[1], [[0, 1, 1]])
            self.assertEqual(report["input_type_map"], ["O", "H"])
            self.assertEqual(report["model_type_map"], ["H", "C", "N", "O"])
            model.get_type_map.return_value = ["H", "C"]
            with self.assertRaisesRegex(ValueError, "O"):
                cli._remap_input_types(model, source, args)

    def test_explicit_input_types_override_metadata_and_check_indices(self):
        model = Mock()
        model.get_type_map.return_value = ["H", "O"]
        source = (np.zeros((1, 2, 3)), np.asarray([[0, 1]], dtype=np.int32), None)
        args = SimpleNamespace(input_npz=None, input_type_map="O,H")
        result, _ = cli._remap_input_types(model, source, args)
        np.testing.assert_array_equal(result[1], [[1, 0]])
        args.input_type_map = None
        result, report = cli._remap_input_types(model, source, args)
        np.testing.assert_array_equal(result[1], source[1])
        self.assertEqual(report["source"], "model type indices")
        source[1][0, 0] = 2
        with self.assertRaisesRegex(ValueError, "type indices"):
            cli._remap_input_types(model, source, args)

    def test_generate_data_preserves_water_element_labels_for_profiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expanded.npz"
            self.assertEqual(cli.main([
                "generate-data", "--input-npz", str(Path(__file__).with_name("water_spce_4500.npz")),
                "--replicate", "2,1,1", "--output-npz", str(path),
            ]), 0)
            with np.load(path) as sample:
                self.assertIn("type_map", sample.files)
                np.testing.assert_array_equal(sample["type_map"], ["O", "H"])
            model = Mock()
            model.get_type_map.return_value = ["H", "C", "N", "O"]
            remapped, report = cli._remap_input_types(model, cli._load_npz_system(path), SimpleNamespace(input_npz=path, input_type_map=None))
            self.assertEqual(np.count_nonzero(remapped[1] == 3), 3000)
            self.assertEqual(np.count_nonzero(remapped[1] == 0), 6000)
            self.assertEqual(report["input_type_map"], ["O", "H"])
