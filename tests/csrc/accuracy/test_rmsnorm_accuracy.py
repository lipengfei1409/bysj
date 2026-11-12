"""Opt-in native RMSNorm accuracy and force-gradient tests for a supported DCU."""
import unittest

import torch

from kernels.rmsnorm import rmsnorm, rmsnorm_available
from reference.rmsnorm import rmsnorm as reference_rmsnorm


@unittest.skipUnless(
    rmsnorm_available() and torch.cuda.is_available(),
    "requires a built RMSNorm HIP extension and supported DCU",
)
class RMSNormDCUAccuracyTests(unittest.TestCase):
    def test_forward_tail_widths_and_empty_rows(self):
        for shape in ((1, 1), (7, 127), (2, 3, 257), (0, 65)):
            with self.subTest(shape=shape):
                x = torch.randn(*shape, device="cuda", dtype=torch.float32)
                width = shape[-1]
                scale = torch.randn(width, device="cuda", dtype=torch.float32)
                actual = rmsnorm(x, scale, eps=1.0e-5)
                expected = reference_rmsnorm(x, scale, eps=1.0e-5)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                self.assertEqual(tuple(actual.shape), shape)

    def test_input_and_scale_gradients(self):
        x = torch.randn(5, 3, 127, device="cuda", dtype=torch.float32, requires_grad=True)
        scale = torch.randn(127, device="cuda", dtype=torch.float32, requires_grad=True)
        upstream = torch.randn_like(x)
        actual = rmsnorm(x, scale, eps=1.0e-5)
        actual_grads = torch.autograd.grad(actual, (x, scale), upstream)

        x_ref = x.detach().clone().requires_grad_(True)
        scale_ref = scale.detach().clone().requires_grad_(True)
        expected = reference_rmsnorm(x_ref, scale_ref, eps=1.0e-5)
        expected_grads = torch.autograd.grad(expected, (x_ref, scale_ref), upstream)
        torch.testing.assert_close(actual_grads[0], expected_grads[0], rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(actual_grads[1], expected_grads[1], rtol=3e-5, atol=3e-5)

    def test_energy_force_style_gradient(self):
        """Exercise the input gradient required by energy/force differentiation."""
        coords = torch.randn(4, 129, device="cuda", dtype=torch.float32, requires_grad=True)
        scale = torch.randn(129, device="cuda", dtype=torch.float32)
        features = torch.sin(coords) + 0.25 * coords
        energy = rmsnorm(features, scale).square().sum()
        actual_force = -torch.autograd.grad(energy, coords)[0]

        coords_ref = coords.detach().clone().requires_grad_(True)
        features_ref = torch.sin(coords_ref) + 0.25 * coords_ref
        energy_ref = reference_rmsnorm(features_ref, scale).square().sum()
        expected_force = -torch.autograd.grad(energy_ref, coords_ref)[0]
        torch.testing.assert_close(actual_force, expected_force, rtol=4e-5, atol=4e-5)

    def test_inference_mode_representative_feature_widths(self):
        # These widths are DPA4-relevant synthetic stress points, not claims about
        # a particular external DeepMD checkpoint's hidden dimensions.
        with torch.inference_mode():
            for rows, width in ((4, 128), (16, 256), (8, 512)):
                with self.subTest(rows=rows, width=width):
                    x = torch.randn(rows, width, device="cuda", dtype=torch.float32)
                    scale = torch.ones(width, device="cuda", dtype=torch.float32)
                    output = rmsnorm(x, scale)
                    self.assertTrue(torch.isfinite(output).all())
                    torch.testing.assert_close(
                        output, reference_rmsnorm(x, scale), rtol=2e-5, atol=2e-5
                    )

    def test_current_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randn(9, 131, device="cuda", dtype=torch.float32)
            scale = torch.randn(131, device="cuda", dtype=torch.float32)
            actual = rmsnorm(x, scale)
            expected = reference_rmsnorm(x, scale)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
