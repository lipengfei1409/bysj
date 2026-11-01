"""Tests for the standalone TileLang SO2 radial scaling operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops.so2_radial_scale import (
    so2_radial_scale_reference,
    tilelang_so2_radial_scale,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class SO2RadialScaleTests(unittest.TestCase):
    def test_reference_forward_and_gradients(self):
        values = torch.randn((3, 4, 5), dtype=torch.float64, requires_grad=True)
        radial = torch.randn((3, 4, 5), dtype=torch.float64, requires_grad=True)
        output = so2_radial_scale_reference(values, radial)
        expected = values * radial
        torch.testing.assert_close(output, expected)
        upstream = torch.randn_like(output)
        gradients = torch.autograd.grad(output, (values, radial), upstream)
        expected_gradients = torch.autograd.grad(expected, (values, radial), upstream)
        for grad, expected_grad in zip(gradients, expected_gradients):
            torch.testing.assert_close(grad, expected_grad)

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        values = torch.ones((2, 3, 4), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_so2_radial_scale(values, values)

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        values = torch.randn(
            (9, 4, 32), dtype=torch.float32, device="cuda", requires_grad=True
        )
        radial = torch.randn_like(values, requires_grad=True)
        upstream = torch.randn_like(values)
        actual = tilelang_so2_radial_scale(values, radial)
        actual_grads = torch.autograd.grad(actual, (values, radial), upstream)
        ref_values = values.detach().clone().requires_grad_(True)
        ref_radial = radial.detach().clone().requires_grad_(True)
        expected = so2_radial_scale_reference(ref_values, ref_radial)
        expected_grads = torch.autograd.grad(
            expected, (ref_values, ref_radial), upstream
        )
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
        for grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(grad, expected_grad, rtol=2.0e-5, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
