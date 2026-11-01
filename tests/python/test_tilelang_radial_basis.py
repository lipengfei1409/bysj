"""Tests for the standalone TileLang Bessel radial-basis operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops.radial_basis import (
    bessel_radial_basis_reference,
    tilelang_bessel_radial_basis,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class BesselRadialBasisTests(unittest.TestCase):
    def test_reference_matches_torch_sinc_and_has_finite_zero_gradient(self):
        distance = torch.tensor([[0.0], [1.0e-7], [0.25], [5.5]], requires_grad=True)
        frequencies = torch.tensor([[0.5, 1.0, 1.5]])
        actual = bessel_radial_basis_reference(distance, frequencies)
        expected = frequencies * torch.sinc(distance * frequencies / torch.pi)
        torch.testing.assert_close(actual, expected)

        (gradient,) = torch.autograd.grad(actual.sum(), distance)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(gradient[0].item(), 0.0)

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_bessel_radial_basis(
                torch.zeros((2, 1), dtype=torch.float32),
                torch.ones((1, 3), dtype=torch.float32),
            )

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_distance_backward_match_reference(self):
        distance = torch.tensor(
            [[0.0], [1.0e-5], [0.25], [3.0]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        frequencies = torch.tensor(
            [[0.5, 1.0, 1.5, 2.0]], dtype=torch.float32, device="cuda"
        )
        upstream = torch.randn((distance.shape[0], frequencies.shape[1]), device="cuda")

        actual = tilelang_bessel_radial_basis(distance, frequencies)
        (actual_grad,) = torch.autograd.grad(actual, distance, upstream)

        reference_distance = distance.detach().clone().requires_grad_(True)
        expected = bessel_radial_basis_reference(reference_distance, frequencies)
        (expected_grad,) = torch.autograd.grad(expected, reference_distance, upstream)
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2.0e-4, atol=2.0e-4)


if __name__ == "__main__":
    unittest.main()
