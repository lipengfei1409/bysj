"""Contracts for the DeepMD-backed DPA4 adapter."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from model import dpa4
from model.dpa4 import DPA4Model


class _FakePtExptBackend:
    __module__ = "deepmd.pt_expt.infer.deep_eval"


class _FakePtBackend:
    __module__ = "deepmd.pt.infer.deep_eval"


class _FakeDeepPot:
    def __init__(self, backend_type=_FakePtExptBackend):
        self.deep_eval = backend_type()
        self.closed = False
        self.result = (np.array([[1.25]]), np.zeros((1, 2, 3)), np.zeros((1, 9)))

    def eval(self, *args, **kwargs):
        self.eval_args = args
        self.eval_kwargs = kwargs
        return self.result

    def get_rcut(self):
        return 6.0

    def get_type_map(self):
        return ["O", "H"]

    def get_ntypes(self):
        return 2

    def get_model_def_script(self):
        return {"model": {"type": "dpa4"}}

    def close(self):
        self.closed = True


class DPA4ModelAdapterTests(unittest.TestCase):
    def test_project_tilelang_package_resolves_without_importing_tilelang(self):
        dpa4._ensure_project_tilelang_ops()

    def test_selected_ops_are_scoped_to_model_construction(self):
        previous = {
            "DP_TILELANG_DPA4": "caller-global",
            "DP_TILELANG_RADIAL_BASIS": "caller-radial",
            "DP_TILELANG_EDGE_GEOMETRY": "caller-edge",
        }
        seen = {}
        fake_model = _FakeDeepPot()

        def create_deep_pot(model_file, **kwargs):
            seen["flags"] = {
                name: os.environ.get(name)
                for name in (
                    "DP_TILELANG_DPA4",
                    "DP_TILELANG_RADIAL_BASIS",
                    "DP_TILELANG_EDGE_GEOMETRY",
                    "DP_TILELANG_GRID_PRODUCT",
                )
            }
            seen["model_file"] = model_file
            seen["kwargs"] = kwargs
            return fake_model

        with mock.patch.dict(os.environ, previous, clear=False):
            with mock.patch.object(
                dpa4, "_create_deep_pot", side_effect=create_deep_pot
            ):
                model = DPA4Model(
                    "water_pt_expt.pt",
                    tilelang_ops=("radial_basis", "edge_geometry"),
                    nlist_backend="native",
                )
            self.assertEqual(os.environ.get("DP_TILELANG_DPA4"), "caller-global")
            self.assertEqual(
                os.environ.get("DP_TILELANG_RADIAL_BASIS"), "caller-radial"
            )
            self.assertEqual(os.environ.get("DP_TILELANG_EDGE_GEOMETRY"), "caller-edge")

        self.assertEqual(
            seen["flags"],
            {
                "DP_TILELANG_DPA4": "0",
                "DP_TILELANG_RADIAL_BASIS": "1",
                "DP_TILELANG_EDGE_GEOMETRY": "1",
                "DP_TILELANG_GRID_PRODUCT": "0",
            },
        )
        self.assertEqual(seen["model_file"], "water_pt_expt.pt")
        self.assertEqual(seen["kwargs"], {"nlist_backend": "native"})
        self.assertEqual(model.backend, "pt_expt")
        self.assertTrue(model.uses_tilelang)
        self.assertEqual(model.get_rcut(), 6.0)
        self.assertEqual(model.get_type_map(), ["O", "H"])
        coords = np.zeros((1, 2, 3), dtype=np.float32)
        cells = np.eye(3, dtype=np.float32).reshape(1, 9)
        atom_types = np.array([[0, 1]], dtype=np.int32)
        self.assertIs(model.eval(coords, cells, atom_types), fake_model.result)
        self.assertEqual(fake_model.eval_args, (coords, cells, atom_types))
        model.close()
        self.assertTrue(fake_model.closed)

    def test_none_preserves_caller_tilelang_environment(self):
        fake_model = _FakeDeepPot()
        seen = {}

        def create_deep_pot(model_file, **kwargs):
            seen["radial"] = os.environ.get("DP_TILELANG_RADIAL_BASIS")
            seen["global"] = os.environ.get("DP_TILELANG_DPA4")
            return fake_model

        with mock.patch.dict(
            os.environ,
            {"DP_TILELANG_DPA4": "0", "DP_TILELANG_RADIAL_BASIS": "1"},
            clear=False,
        ):
            with mock.patch.object(
                dpa4, "_create_deep_pot", side_effect=create_deep_pot
            ):
                model = DPA4Model("water_pt_expt.pt")
        self.assertEqual(seen, {"radial": "1", "global": "0"})
        self.assertEqual(model.tilelang_ops, ("DP_TILELANG_RADIAL_BASIS",))
        self.assertTrue(model.uses_tilelang)

    def test_empty_selection_forces_baseline_and_restores_environment(self):
        fake_model = _FakeDeepPot(backend_type=_FakePtBackend)
        seen = {}

        def create_deep_pot(model_file, **kwargs):
            seen["global"] = os.environ.get("DP_TILELANG_DPA4")
            seen["radial"] = os.environ.get("DP_TILELANG_RADIAL_BASIS")
            return fake_model

        with mock.patch.dict(
            os.environ,
            {"DP_TILELANG_DPA4": "1", "DP_TILELANG_RADIAL_BASIS": "1"},
            clear=False,
        ):
            with mock.patch.object(
                dpa4, "_create_deep_pot", side_effect=create_deep_pot
            ):
                model = DPA4Model("legacy.pt", tilelang_ops=())
            self.assertEqual(os.environ["DP_TILELANG_DPA4"], "1")
            self.assertEqual(os.environ["DP_TILELANG_RADIAL_BASIS"], "1")
        self.assertEqual(seen, {"global": "0", "radial": "0"})
        self.assertEqual(model.backend, "pt")
        self.assertFalse(model.uses_tilelang)

    def test_rejects_selector_changes_for_exported_artifacts(self):
        with mock.patch.object(dpa4, "_create_deep_pot") as create:
            with self.assertRaisesRegex(RuntimeError, "fixed at export time"):
                DPA4Model("water.pt2", tilelang_ops=("radial_basis",))
        create.assert_not_called()

    def test_rejects_tilelang_for_legacy_pt_backend(self):
        fake_model = _FakeDeepPot(backend_type=_FakePtBackend)
        with mock.patch.object(dpa4, "_create_deep_pot", return_value=fake_model):
            with self.assertRaisesRegex(RuntimeError, "pt_expt.*backend"):
                DPA4Model("legacy.pt", tilelang_ops=("radial_basis",))
        self.assertTrue(fake_model.closed)

    def test_unknown_operator_is_rejected_before_loading(self):
        with mock.patch.object(dpa4, "_create_deep_pot") as create:
            with self.assertRaisesRegex(ValueError, "unknown TileLang DPA4 operator"):
                DPA4Model("water.pt2", tilelang_ops=("not-an-op",))
        create.assert_not_called()


_RUN_INTEGRATION = os.environ.get("DP_RUN_DPA4_MODEL_INTEGRATION", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@unittest.skipUnless(
    _RUN_INTEGRATION,
    "set DP_RUN_DPA4_MODEL_INTEGRATION=1 and model/input paths "
    "to run DeepMD integration",
)
class DPA4ModelIntegrationTests(unittest.TestCase):
    def test_deepmd_dpa4_energy_and_force(self):
        model_path = os.environ.get("DP_DPA4_MODEL_FILE")
        input_path = os.environ.get("DP_DPA4_INPUT_NPZ")
        if not model_path or not input_path:
            self.fail("set DP_DPA4_MODEL_FILE and DP_DPA4_INPUT_NPZ")
        if not os.path.isfile(model_path) or not os.path.isfile(input_path):
            self.fail("DP_DPA4_MODEL_FILE and DP_DPA4_INPUT_NPZ must exist")

        with np.load(input_path) as sample:
            coords = sample["coords"]
            atom_types = sample["atom_types"]
            cells = sample["cells"] if "cells" in sample else None
        raw_ops = os.environ.get("DP_DPA4_MODEL_TILELANG_OPS")
        ops = () if raw_ops is None else tuple(
            item.strip() for item in raw_ops.split(",") if item.strip()
        )
        with DPA4Model(model_path, tilelang_ops=ops) as model:
            outputs = model.eval(coords, cells, atom_types)

        if isinstance(outputs, dict):
            energy = outputs.get("energy", outputs.get("energy_redu"))
            force = outputs.get("force", outputs.get("energy_derv_r"))
        else:
            energy, force = outputs[:2]
        self.assertIsNotNone(energy, "DeepMD evaluation did not return energy")
        self.assertIsNotNone(force, "DeepMD evaluation did not return force")
        self.assertTrue(np.isfinite(np.asarray(energy)).all())
        self.assertTrue(np.isfinite(np.asarray(force)).all())
