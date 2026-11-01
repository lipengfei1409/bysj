"""Tests for the standalone TileLang DPA4 environment-seed Gram operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops import (
    environment_seed_gram_reference,
    tilelang_environment_seed_gram,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class EnvironmentSeedGramTests(unittest.TestCase):
    def test_reference_matches_matrix_product_and_gradients(self):
        env_agg = torch.randn((3, 5, 7), dtype=torch.float64, requires_grad=True)
        actual = environment_seed_gram_reference(env_agg, axis_dim=4)
        expected = torch.matmul(env_agg.transpose(-1, -2), env_agg[..., :4])
        torch.testing.assert_close(actual, expected)
        (gradient,) = torch.autograd.grad(actual.square().sum(), env_agg)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_environment_seed_gram(torch.zeros((2, 4, 8)), axis_dim=3)

    def test_invalid_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "axis_dim must be"):
            tilelang_environment_seed_gram(torch.zeros((2, 4, 8)), axis_dim=9)

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        env_agg = torch.randn((3, 5, 7), dtype=torch.float32, device="cuda")
        env_agg.requires_grad_()
        upstream = torch.randn((3, 7, 4), dtype=torch.float32, device="cuda")
        actual = tilelang_environment_seed_gram(env_agg, axis_dim=4)
        (actual_grad,) = torch.autograd.grad(actual, env_agg, upstream)

        ref = env_agg.detach().clone().requires_grad_(True)
        expected = environment_seed_gram_reference(ref, axis_dim=4)
        (expected_grad,) = torch.autograd.grad(expected, ref, upstream)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
