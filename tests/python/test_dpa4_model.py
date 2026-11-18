"""Contracts for the DeepMD-backed DPA4 facade and its fused seams.

The facade is thin on purpose: DeepMD owns the graph, this module owns the
``fused`` switch and the close/uninstall lifecycle. The operator-level tests
live next to the operators (``tests/csrc``) and the seam tests next to the
adapter (``tests/python/test_fused_deepmd.py``); here we only check that the
facade wires them correctly and that the two removed selectors fail loudly.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from model import dpa4
from model.dpa4 import DPA4Model


class _FakeBackend:
    def __init__(self, module_name):
        self.__class__ = type("Backend", (_FakeBackend,), {"__module__": module_name})


class _FakeDeepPot:
    def __init__(self, backend_module="deepmd.pt.infer.deep_eval"):
        self.deep_eval = _FakeBackend(backend_module)
        self.calls = []
        self.closed = False
        self.rcut = 6.0

    def eval(self, coords, cells, atom_types, **kwargs):
        self.calls.append((coords, cells, atom_types, kwargs))
        return {"energy": 1.0, "force": 2.0, "virial": 3.0}

    def get_rcut(self):
        return self.rcut

    def get_type_map(self):
        return ["O", "H"]

    def get_ntypes(self):
        return 2

    def get_model_def_script(self):
        return {"type": "dpa4"}

    def close(self):
        self.closed = True


def _model(fake_pot, **kwargs):
    with mock.patch.object(dpa4, "_create_deep_pot", return_value=fake_pot):
        return DPA4Model("water.pt", **kwargs)


class DPA4ModelFacadeTests(unittest.TestCase):
    def test_eval_forwards_every_argument(self):
        fake = _FakeDeepPot()
        model = _model(fake)
        coords, cells, types = np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), np.zeros((1, 3), dtype=int)
        output = model.eval(coords, cells, types, atomic=True, fparam=[0.5], aparam=[0.1])
        self.assertEqual(output["energy"], 1.0)
        forwarded = fake.calls[0]
        self.assertIs(forwarded[0], coords)
        self.assertIs(forwarded[1], cells)
        self.assertIs(forwarded[2], types)
        self.assertEqual(forwarded[3], {"atomic": True, "fparam": [0.5], "aparam": [0.1]})
        self.assertIs(model.predict.__func__, model.eval.__func__)
        self.assertEqual(model.get_rcut(), 6.0)
        self.assertEqual(model.get_type_map(), ["O", "H"])
        self.assertEqual(model.get_ntypes(), 2)
        self.assertEqual(model.get_model_def_script(), {"type": "dpa4"})

    def test_backend_name_comes_from_the_evaluator_module(self):
        self.assertEqual(_model(_FakeDeepPot()).backend, "pt")
        self.assertEqual(
            _model(_FakeDeepPot("deepmd.pt_expt.infer.deep_eval")).backend, "pt_expt"
        )
        self.assertEqual(_model(_FakeDeepPot("torch")).backend, "unknown")

    def test_fused_switch_installs_and_records_the_report(self):
        fake = _FakeDeepPot()
        report = {"softmax_hits": 3, "bound_softmax": True}
        with mock.patch.object(
            dpa4, "_create_deep_pot", return_value=fake
        ), mock.patch("model.fused_deepmd.deepmd_patch") as patch, mock.patch(
            "model.fused_deepmd.status", return_value=report
        ):
            model = DPA4Model("water.pt", fused=True)
        patch.assert_called_once_with(fake, softmax=True, envelope=True)
        self.assertEqual(model.fused_ops, report)
        self.assertEqual(model.fused_seams, {"softmax": True, "envelope": True})

    def test_no_fused_switch_leaves_deepmd_untouched(self):
        with mock.patch("model.fused_deepmd.deepmd_patch") as patch:
            model = _model(_FakeDeepPot())
        patch.assert_not_called()
        self.assertEqual(model.fused_ops, {})
        self.assertIsNone(model.fused_seams)

    def test_unknown_seam_is_rejected_before_loading(self):
        with mock.patch.object(dpa4, "_create_deep_pot") as create:
            with self.assertRaisesRegex(ValueError, "unknown fused seam"):
                DPA4Model("water.pt", fused="flash")
        create.assert_not_called()

    def test_close_uninstalls_once_and_closes_the_pot(self):
        fake = _FakeDeepPot()
        with mock.patch.object(dpa4, "_create_deep_pot", return_value=fake), mock.patch(
            "model.fused_deepmd.deepmd_patch", return_value=SimpleNamespace(as_dict=lambda: {"bound": True})
        ), mock.patch("model.fused_deepmd.status", return_value={"bound": True}), mock.patch(
            "model.fused_deepmd.uninstall"
        ) as uninstall:
            model = DPA4Model("water.pt", fused=("softmax",))
            model.close()
            self.assertEqual(model.fused_ops, {})
            model.close()
        uninstall.assert_called_once_with(fake)
        self.assertTrue(fake.closed)

    def test_context_manager_closes_and_uninstalls(self):
        fake = _FakeDeepPot()
        with mock.patch.object(dpa4, "_create_deep_pot", return_value=fake), mock.patch(
            "model.fused_deepmd.deepmd_patch"
        ), mock.patch("model.fused_deepmd.status", return_value={}), mock.patch(
            "model.fused_deepmd.uninstall"
        ):
            with DPA4Model("water.pt", fused=True) as model:
                self.assertIsInstance(model, DPA4Model)
        self.assertTrue(fake.closed)

    def test_repr_names_the_fused_report(self):
        fake = _FakeDeepPot()
        with mock.patch.object(dpa4, "_create_deep_pot", return_value=fake), mock.patch(
            "model.fused_deepmd.deepmd_patch"
        ), mock.patch("model.fused_deepmd.status", return_value={"softmax_hits": 1}):
            text = repr(DPA4Model("water.pt", fused=("softmax",)))
        self.assertIn("backend='pt'", text)
        self.assertIn("fused_ops={'softmax_hits': 1}", text)

    def test_removed_selectors_fail_loudly_before_loading(self):
        for keyword in ("tilelang_ops", "native_ops"):
            with self.subTest(keyword=keyword):
                with mock.patch.object(dpa4, "_create_deep_pot") as create:
                    with self.assertRaisesRegex(RuntimeError, f"{keyword} was removed"):
                        DPA4Model("water.pt", **{keyword: ("gie",)})
                create.assert_not_called()


@unittest.skipUnless(
    os.environ.get("DP_RUN_DPA4_MODEL_INTEGRATION") == "1",
    "set DP_RUN_DPA4_MODEL_INTEGRATION=1 with a DCU and DeepMD",
)
class DPA4ModelIntegrationTests(unittest.TestCase):
    def test_fused_seams_match_the_deepmd_baseline(self):
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

        with DPA4Model(model_path) as baseline:
            expected = baseline.eval(coords, cells, atom_types)
        with DPA4Model(model_path, fused=True) as patched:
            actual = patched.eval(coords, cells, atom_types)
            self.assertTrue(patched.fused_ops.get("bound_softmax") or
                            patched.fused_ops.get("bound_envelope"),
                            f"no fused seam was bound: {patched.fused_ops}")
            self.assertEqual(patched.fused_ops.get("softmax_fallbacks", 0) > 0, False)

        for label, expected_value, actual_value in zip(
            ("energy", "force", "virial"),
            _outputs(expected),
            _outputs(actual),
        ):
            with self.subTest(output=label):
                if expected_value is None or actual_value is None:
                    self.assertIs(expected_value, actual_value)
                else:
                    np.testing.assert_allclose(
                        np.asarray(actual_value),
                        np.asarray(expected_value),
                        rtol=2.0e-4,
                        atol=2.0e-5,
                        err_msg=f"the fused seams changed {label}",
                    )


def _outputs(outputs):
    """Normalise DeepMD's dict or tuple return into ``(energy, force, virial)``."""
    if isinstance(outputs, dict):
        return (
            outputs.get("energy", outputs.get("energy_redu")),
            outputs.get("force", outputs.get("energy_derv_r")),
            outputs.get("virial", outputs.get("energy_derv_c")),
        )
    energy, force = outputs[:2]
    virial = outputs[2] if len(outputs) > 2 else None
    return energy, force, virial


if __name__ == "__main__":
    unittest.main()
