"""Tests for the standalone TileLang C3 cutoff-envelope operator."""

import importlib.util
import math
import unittest

import torch
from kernels.tilelang_ops.cutoff_envelope import (
    c3_cutoff_envelope_reference,
    tilelang_c3_cutoff_envelope,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class C3CutoffEnvelopeTests(unittest.TestCase):
    def test_reference_matches_polynomial_and_boundary_gradients(self):
        rcut = 6.0
        for exponent in (1, 5, 7):
            distance = torch.tensor(
                [[-0.2], [0.0], [0.25], [rcut - 1.0e-5], [rcut], [rcut + 0.5]],
                dtype=torch.float64,
                requires_grad=True,
            )
            actual = c3_cutoff_envelope_reference(distance, rcut, exponent)
            x = (distance / rcut).clamp(0.0, 1.0)
            expected = (1.0 - x).pow(4) * sum(
                float(math.comb(k + 3, 3)) * x.pow(k) for k in range(exponent)
            )
            torch.testing.assert_close(actual, expected)
            (grad,) = torch.autograd.grad(actual.sum(), distance)
            self.assertTrue(torch.isfinite(grad).all())
            if exponent >= 2:
                self.assertEqual(grad[1].item(), 0.0)
            self.assertEqual(grad[-2].item(), 0.0)
            self.assertEqual(grad[-1].item(), 0.0)

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_c3_cutoff_envelope(torch.zeros((2, 1)), 6.0, 5)

    def test_invalid_inputs_are_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError, r"shape \(n_edges, 1\)"):
            tilelang_c3_cutoff_envelope(torch.zeros((2,)), 6.0, 5)
        with self.assertRaisesRegex(ValueError, "rcut must be positive"):
            tilelang_c3_cutoff_envelope(torch.zeros((2, 1)), 0.0, 5)

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_distance_backward_match_reference(self):
        rcut = 6.0
        distance = torch.tensor(
            [[0.0], [1.0e-5], [0.25], [rcut - 1.0e-4], [rcut], [7.0]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        upstream = torch.randn_like(distance)
        for exponent in (1, 5, 7):
            actual = tilelang_c3_cutoff_envelope(distance, rcut, exponent)
            (actual_grad,) = torch.autograd.grad(
                actual, distance, upstream, retain_graph=True
            )
            reference_distance = distance.detach().clone().requires_grad_(True)
            expected = c3_cutoff_envelope_reference(reference_distance, rcut, exponent)
            (expected_grad,) = torch.autograd.grad(
                expected, reference_distance, upstream
            )
            torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=3.0e-4, atol=2.0e-5
            )


if __name__ == "__main__":
    unittest.main()
