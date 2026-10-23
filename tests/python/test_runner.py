import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.dpa4.runner import compute_metrics, configure_device
from model.dpa4.runner import DPA4Runner


class RunnerTests(unittest.TestCase):
    @unittest.skipIf(torch.version.hip and torch.cuda.is_available(), "requires unavailable DCU")
    def test_dcu_request_does_not_silently_run_cpu(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,"ROCm|DCU"):
                configure_device("dcu")
            self.assertNotIn("DP_COMPILE_INFER",os.environ)

    @unittest.skipUnless(sys.version_info<(3,10), "old interpreter diagnostic")
    def test_deepmd_integration_requires_supported_python(self):
        with self.assertRaisesRegex(RuntimeError,"Python.*3.10"):
            DPA4Runner("unused.pt")

    def test_unknown_operator_backend_is_rejected_before_checkpoint_load(self):
        with self.assertRaisesRegex(ValueError,"operator_backend"):
            DPA4Runner("missing-checkpoint.pt",operator_backend="auto")

    def test_metrics_report_energy_per_atom_and_force(self):
        predicted_energy = np.array([[3.0]])
        reference_energy = np.array([[1.0]])
        predicted_force = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        reference_force = np.zeros((1, 2, 3))

        metrics = compute_metrics(
            predicted_energy, predicted_force, reference_energy, reference_force
        )

        self.assertEqual(metrics["energy_rmse_ev"], 2.0)
        self.assertEqual(metrics["energy_rmse_per_atom_ev"], 1.0)
        self.assertAlmostEqual(metrics["force_rmse_ev_per_angstrom"], 1.0 / np.sqrt(6))

    def test_cpu_configuration_disables_compile_and_fused_paths(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_device("cpu")
            self.assertEqual(os.environ["DEVICE"], "cpu")
            self.assertEqual(os.environ["DP_COMPILE_INFER"], "0")
            self.assertEqual(os.environ["DP_TRITON_INFER"], "0")
            self.assertEqual(os.environ["DP_CUDA_INFER"], "0")
            for name in ("DP_CUTE_INFER","DP_CUTILE_INFER","DP_ACT_INFER","DP_AMP_INFER"):
                self.assertEqual(os.environ.get(name),"0")

    def test_dcu_configuration_rejects_unknown_device(self):
        with self.assertRaises(ValueError):
            configure_device("nvidia")


if __name__ == "__main__":
    unittest.main()
