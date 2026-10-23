"""Opt-in actual DeePMD + water checkpoint integration, never contract doubles.

With DPA4_RUN_DEEPMD_TESTS=1 a missing dependency is a FAILURE, not a skip.
Use in the existing DeePMD environment after installing this project.
"""
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("DPA4_RUN_DEEPMD_TESTS")=="1","opt in to actual installed DeePMD integration")
class DeePMDIntegrationTests(unittest.TestCase):
    def test_water_energy_force_virial_install_and_restore(self):
        from model.dpa4.runner import DPA4Runner,configure_device
        from model.dpa4.system import load_deepmd_system
        from dcu_dpa4.deepmd import install_operators

        root=Path(__file__).resolve().parents[3]
        checkpoint=os.environ.get("DPA4_TEST_CHECKPOINT",str(root/"deepmd-kit/examples/water/dpa4/lmp/pretrained.pt"))
        system_path=os.environ.get("DPA4_TEST_SYSTEM",str(root/"deepmd-kit/examples/water/data/data_0"))
        device=os.environ.get("DPA4_TEST_DEVICE","cpu")
        backend=os.environ.get("DPA4_TEST_OPERATOR_BACKEND","reference")
        neighbors=os.environ.get("DPA4_TEST_NEIGHBOR_BACKEND","native")
        configure_device(device)
        system=load_deepmd_system(system_path,max_frames=1)
        with DPA4Runner(checkpoint,neighbor_backend=neighbors) as runner:
            ref=runner.eval(system)
            model=runner.torch_model
            state_keys=tuple(model.state_dict())
            params=dict(model.named_parameters())
            with install_operators(model,backend=backend) as handle:
                actual=runner.eval(system)
                self.assertGreater(sum(r["calls"] for r in handle.report()),0,"installed but bypassed")
                self.assertEqual(tuple(model.state_dict()),state_keys)
                self.assertTrue(all(dict(model.named_parameters())[k] is v for k,v in params.items()))
                for name in ("energy","force","virial"):
                    # Native FP32 atomic/GEMM order differs; validate E/F/V explicitly.
                    tolerance=2e-3 if backend=="hip" else 2e-5
                    np.testing.assert_allclose(getattr(actual,name),getattr(ref,name),rtol=tolerance,atol=tolerance)
            restored=runner.eval(system)
            for name in ("energy","force","virial"):
                np.testing.assert_allclose(getattr(restored,name),getattr(ref,name),rtol=1e-7,atol=1e-7)
        # Also exercise the user-facing automatic attach/lifecycle path.
        with DPA4Runner(checkpoint,device=device,neighbor_backend=neighbors,operator_backend=backend) as runner:
            actual=runner.eval(system)
            self.assertGreater(sum(r["calls"] for r in runner.operator_report),0)
            for name in ("energy","force","virial"):
                tolerance=2e-3 if backend=="hip" else 2e-5
                np.testing.assert_allclose(getattr(actual,name),getattr(ref,name),rtol=tolerance,atol=tolerance)
