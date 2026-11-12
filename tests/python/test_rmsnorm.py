"""CPU contracts and reference checks for the native RMSNorm wrapper."""
import importlib
import math
import os
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from kernels.rmsnorm import rmsnorm
from reference.rmsnorm import rmsnorm as reference_rmsnorm

_rmsnorm_module = importlib.import_module("kernels.rmsnorm")
_native_module = importlib.import_module("kernels.native")


class RMSNormBuildConfigurationTests(unittest.TestCase):
    def _capture_native_extension(self, flags):
        root = Path(__file__).resolve().parents[2]
        extension_args = {}

        def fake_cuda_extension(name, **kwargs):
            extension_args["name"] = name
            extension_args.update(kwargs)
            return SimpleNamespace(name=name)

        with patch.dict(os.environ, flags, clear=False), patch.object(
            torch.version, "hip", "6.0.0"
        ), patch(
            "torch.utils.cpp_extension.CUDAExtension",
            side_effect=fake_cuda_extension,
        ), patch("torch.utils.cpp_extension.BuildExtension", return_value="build-ext"), patch(
            "setuptools.setup"
        ) as setup_call:
            runpy.run_path(str(root / "setup.py"), run_name="__main__")

        return extension_args, setup_call.call_args.kwargs

    def test_rmsnorm_only_build_does_not_require_or_include_dumma(self):
        extension, setup_kwargs = self._capture_native_extension(
            {
                "DPA4_BUILD_HIP": "1",
                "DPA4_ENABLE_DUMMA": "0",
                "DPA4_ENABLE_RMSNORM": "1",
                "DPA4_ENABLE_GIE": "0",
            }
        )
        self.assertEqual(extension["name"], "kernels._C")
        self.assertEqual(
            extension["sources"], ["csrc/bindings.cpp", "csrc/rmsnorm/rmsnorm.cpp"]
        )
        self.assertEqual(extension["define_macros"], [("DPA4_ENABLE_RMSNORM", "1")])
        self.assertEqual(set(setup_kwargs["cmdclass"]), {"build_ext"})

    def test_existing_dumma_build_selection_is_preserved(self):
        extension, _ = self._capture_native_extension(
            {
                "DPA4_BUILD_HIP": "1",
                "DPA4_ENABLE_DUMMA": "1",
                "DPA4_ENABLE_RMSNORM": "0",
                "DPA4_ENABLE_GIE": "0",
            }
        )
        self.assertEqual(
            extension["sources"], ["csrc/bindings.cpp", "csrc/gemm/gemm.cpp"]
        )
        self.assertEqual(extension["define_macros"], [("DPA4_ENABLE_DUMMA", "1")])


class RMSNormReferenceTests(unittest.TestCase):
    def test_reference_matches_formula_and_gradients(self):
        x = torch.randn(3, 2, 17, dtype=torch.float32, requires_grad=True)
        scale = torch.randn(17, dtype=torch.float32, requires_grad=True)
        expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        expected = expected * scale
        actual = reference_rmsnorm(x, scale, eps=1.0e-5)
        torch.testing.assert_close(actual, expected)

        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (x, scale), grad)
        expected_grads = torch.autograd.grad(expected, (x, scale), grad)
        torch.testing.assert_close(actual_grads[0], expected_grads[0])
        torch.testing.assert_close(actual_grads[1], expected_grads[1])

    def test_reference_without_scale_and_zero_input(self):
        x = torch.zeros(2, 7, dtype=torch.float32)
        output = reference_rmsnorm(x, eps=1.0e-6)
        self.assertTrue(torch.equal(output, x))
        self.assertTrue(torch.isfinite(output).all())

    def test_inference_mode_on_dpa4_representative_feature_widths(self):
        # Synthetic feature widths for primitive validation, not dimensions
        # asserted to belong to a particular external DeepMD checkpoint.
        with torch.inference_mode():
            for rows, width in ((1, 128), (8, 256), (32, 512)):
                with self.subTest(rows=rows, width=width):
                    x = torch.randn(rows, width, dtype=torch.float32)
                    scale = torch.ones(width, dtype=torch.float32)
                    output = reference_rmsnorm(x, scale)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(tuple(output.shape), (rows, width))

    def test_energy_force_style_input_gradient(self):
        coords = torch.randn(4, 19, dtype=torch.float32, requires_grad=True)
        scale = torch.randn(19, dtype=torch.float32)
        features = torch.sin(coords) + 0.25 * coords
        energy = reference_rmsnorm(features, scale).square().sum()
        force = -torch.autograd.grad(energy, coords)[0]

        coords_ref = coords.detach().clone().requires_grad_(True)
        features_ref = torch.sin(coords_ref) + 0.25 * coords_ref
        energy_ref = reference_rmsnorm(features_ref, scale).square().sum()
        expected_force = -torch.autograd.grad(energy_ref, coords_ref)[0]
        torch.testing.assert_close(force, expected_force)

    def test_reference_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(TypeError, "input must be a tensor"):
            reference_rmsnorm(None)
        with self.assertRaisesRegex(ValueError, "at least one dimension"):
            reference_rmsnorm(torch.tensor(1.0))
        with self.assertRaisesRegex(ValueError, "feature dimension"):
            reference_rmsnorm(torch.empty(2, 0))
        with self.assertRaisesRegex(TypeError, "float32"):
            reference_rmsnorm(torch.ones(2, 4, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            reference_rmsnorm(torch.ones(3, 4).t())
        with self.assertRaisesRegex(ValueError, "scale must have shape"):
            reference_rmsnorm(torch.ones(2, 4), torch.ones(3))
        for eps in (0, -1, math.inf, math.nan):
            with self.subTest(eps=eps):
                with self.assertRaisesRegex(ValueError, "eps"):
                    reference_rmsnorm(torch.ones(2, 4), eps=eps)


class RMSNormWrapperTests(unittest.TestCase):
    def test_invalid_contracts_fail_before_dcu_check(self):
        with self.assertRaisesRegex(TypeError, "input must be a tensor"):
            rmsnorm(None)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            rmsnorm(torch.ones(3, 5).t())
        with self.assertRaisesRegex(ValueError, "scale must have shape"):
            rmsnorm(torch.ones(2, 4), torch.ones(3))
        with self.assertRaisesRegex(ValueError, "eps"):
            rmsnorm(torch.ones(2, 4), eps=0.0)

    def test_cpu_tensor_reaches_dcu_guard_after_validation(self):
        with patch.object(
            _rmsnorm_module,
            "require_dcu",
            side_effect=RuntimeError("DCU unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "DCU unavailable"):
                rmsnorm(torch.ones(2, 4))

    def test_native_dispatch_and_input_scale_gradients(self):
        class FakeExtension:
            @staticmethod
            def rmsnorm_forward(x, scale, eps):
                weight = scale if scale.numel() else None
                output = reference_rmsnorm(x, weight, eps)
                inv = torch.rsqrt(x.square().mean(dim=-1) + eps).reshape(-1)
                return output, inv

            @staticmethod
            def rmsnorm_backward(grad_output, x, scale, inverse_rms, need_scale_grad):
                width = x.shape[-1]
                rows = x.reshape(-1, width)
                grad_rows = grad_output.reshape(-1, width)
                inv = inverse_rms.reshape(-1, 1)
                scaled_grad = grad_rows * (scale if scale.numel() else 1.0)
                grad_x = inv * (
                    scaled_grad
                    - rows * inv.square() * (rows * scaled_grad).mean(dim=1, keepdim=True)
                )
                grad_x = grad_x.reshape_as(x)
                grad_scale = (
                    (grad_rows * rows * inv).sum(dim=0)
                    if need_scale_grad
                    else x.new_empty((0,))
                )
                return grad_x, grad_scale

        x = torch.randn(2, 3, 11, dtype=torch.float32, requires_grad=True)
        scale = torch.randn(11, dtype=torch.float32, requires_grad=True)
        grad = torch.randn_like(x)
        expected = reference_rmsnorm(x, scale, eps=1.0e-5)
        expected_grads = torch.autograd.grad(expected, (x, scale), grad)
        extension = FakeExtension()
        with patch.object(_rmsnorm_module, "require_dcu"), patch.object(
            _rmsnorm_module, "_extension", return_value=extension
        ):
            actual = rmsnorm(x, scale, eps=1.0e-5)
            actual_grads = torch.autograd.grad(actual, (x, scale), grad)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grads[0], expected_grads[0])
        torch.testing.assert_close(actual_grads[1], expected_grads[1])

    def test_native_loader_accepts_rmsnorm_only_extension(self):
        extension = SimpleNamespace(rmsnorm_forward=object(), rmsnorm_backward=object())
        with patch.object(_native_module, "_require_hip_runtime"), patch.object(
            _native_module.importlib, "import_module", return_value=extension
        ):
            self.assertIs(_native_module._extension("rmsnorm_forward"), extension)

        with patch.object(_native_module, "_require_hip_runtime"), patch.object(
            _native_module.importlib, "import_module", return_value=SimpleNamespace()
        ):
            with self.assertRaisesRegex(RuntimeError, "no 'rmsnorm_forward' entry point"):
                _native_module._extension("rmsnorm_forward")

    def test_native_dispatch_without_scale(self):
        class FakeExtension:
            @staticmethod
            def rmsnorm_forward(x, scale, eps):
                self.assertEqual(scale.numel(), 0)
                return reference_rmsnorm(x, eps=eps), torch.rsqrt(
                    x.square().mean(dim=-1) + eps
                ).reshape(-1)

            @staticmethod
            def rmsnorm_backward(grad_output, x, scale, inverse_rms, need_scale_grad):
                rows = x.reshape(-1, x.shape[-1])
                grad_rows = grad_output.reshape_as(rows)
                inv = inverse_rms.reshape(-1, 1)
                grad_x = inv * (
                    grad_rows
                    - rows * inv.square() * (rows * grad_rows).mean(dim=1, keepdim=True)
                )
                return grad_x.reshape_as(x), x.new_empty((0,))

        x = torch.randn(3, 9, requires_grad=True)
        with patch.object(_rmsnorm_module, "require_dcu"), patch.object(
            _rmsnorm_module, "_extension", return_value=FakeExtension()
        ):
            output = rmsnorm(x)
        torch.testing.assert_close(output, reference_rmsnorm(x))
        self.assertEqual(tuple(output.shape), tuple(x.shape))


if __name__ == "__main__":
    unittest.main()
