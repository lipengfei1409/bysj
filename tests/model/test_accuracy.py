"""Opt-in full-model E/F/V parity against the same DeePMD checkpoint."""
import os
from dataclasses import replace
import unittest

import numpy as np
import torch

from model.modeling.dpa4 import DPA4Model
from .case import make_water_accuracy_case, resolve_checkpoint


_RUN = os.environ.get("DPA4_RUN_MODEL_ACCURACY_TESTS") == "1"


@unittest.skipUnless(_RUN, "set DPA4_RUN_MODEL_ACCURACY_TESTS=1 for model oracle accuracy")
class DPA4NeoModelAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = resolve_checkpoint()
        # Configure the reference runtime before constructing DeePMD's backend.
        from model.runner import DPA4Runner, configure_device

        configure_device("cpu")
        cls.model = DPA4Model.from_checkpoint(
            cls.checkpoint, primitive_backend="reference", linear_backend="reference",
        ).eval()
        cls.system, cls.batch, cls.graph = make_water_accuracy_case(cls.model.config)
        cls.runner = DPA4Runner(cls.checkpoint, device="cpu", neighbor_backend="native")
        cls.reference = cls.runner.eval(cls.system)
        with torch.no_grad():
            cls.actual = cls.model(cls.batch, cls.graph, compute_forces=True)

    @classmethod
    def tearDownClass(cls):
        runner = getattr(cls, "runner", None)
        if runner is not None:
            runner.close()

    def _compare(self, name, actual, expected, *, atol):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        self.assertTrue(np.isfinite(actual).all(), f"{name} contains non-finite values")
        self.assertTrue(np.isfinite(expected).all(), f"reference {name} contains non-finite values")
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        details = (
            f"{name}: rmse={np.sqrt(np.mean(delta * delta)):.6g}, "
            f"max_abs={np.max(np.abs(delta)):.6g}, checkpoint={self.checkpoint}"
        )
        np.testing.assert_allclose(
            actual, expected,
            rtol=float(os.environ.get("DPA4_MODEL_RTOL", "2e-4")),
            atol=atol,
            err_msg=details,
        )

    def test_frame_energy_matches_deepmd(self):
        self._compare(
            "energy_eV", self.actual.energy.detach().cpu().numpy(),
            self.reference.energy, atol=float(os.environ.get("DPA4_MODEL_ENERGY_ATOL", "1e-4")),
        )
        atom_sum = self.actual.atom_energy.detach().sum().cpu().item()
        frame_sum = self.actual.energy.detach().sum().cpu().item()
        self.assertAlmostEqual(atom_sum, frame_sum, delta=1e-8)

    def test_force_matches_deepmd(self):
        self.assertIsNotNone(self.actual.force, "model did not return forces")
        force = self.actual.force.detach().reshape_as(
            torch.as_tensor(self.reference.force, dtype=self.actual.force.dtype)
        ).cpu().numpy()
        self._compare(
            "force_eV_per_A", force, self.reference.force,
            atol=float(os.environ.get("DPA4_MODEL_FORCE_ATOL", "2e-4")),
        )

    def test_virial_matches_deepmd(self):
        self.assertIsNotNone(self.actual.virial, "model did not return virial")
        self._compare(
            "virial_eV", self.actual.virial.detach().cpu().numpy(),
            self.reference.virial,
            atol=float(os.environ.get("DPA4_MODEL_VIRIAL_ATOL", "2e-3")),
        )

    @unittest.skipUnless(
        os.environ.get("DPA4_RUN_MODEL_FINITE_DIFFERENCE") == "1",
        "set DPA4_RUN_MODEL_FINITE_DIFFERENCE=1 for the optional model force check",
    )
    def test_one_force_component_matches_energy_finite_difference(self):
        step = float(os.environ.get("DPA4_MODEL_FD_STEP", "1e-3"))
        atom, axis = 1, 0
        positions = self.batch.positions.detach().clone()
        plus = positions.clone()
        minus = positions.clone()
        plus[atom, axis] += step
        minus[atom, axis] -= step
        with torch.no_grad():
            e_plus = self.model(replace(self.batch, positions=plus), self.graph,
                                compute_forces=False).energy.sum().item()
            e_minus = self.model(replace(self.batch, positions=minus), self.graph,
                                 compute_forces=False).energy.sum().item()
        finite_difference_force = -(e_plus - e_minus) / (2.0 * step)
        predicted_force = self.actual.force[atom, axis].detach().cpu().item()
        self.assertAlmostEqual(
            predicted_force, finite_difference_force,
            delta=float(os.environ.get("DPA4_MODEL_FD_ATOL", "2e-2")),
        )


if __name__ == "__main__":
    unittest.main()
