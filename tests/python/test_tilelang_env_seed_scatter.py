"""Tests for the standalone TileLang environment-seed scatter."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops.env_seed_scatter import (
    environment_seed_scatter_reference,
    tilelang_environment_seed_scatter,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


def _inputs(device="cpu", requires_grad=False):
    geometry = torch.randn((5, 4), dtype=torch.float32, device=device)
    features = torch.randn((5, 7), dtype=torch.float32, device=device)
    inv = torch.tensor([0.5, 0.7, 0.9, 0.3], dtype=torch.float32, device=device)
    dst = torch.tensor([1, 0, 1, 3, 1], dtype=torch.int64, device=device)
    if requires_grad:
        geometry.requires_grad_()
        features.requires_grad_()
        inv.requires_grad_()
    return geometry, features, dst, inv


class EnvironmentSeedScatterTests(unittest.TestCase):
    def test_reference_matches_manual_scatter_and_gradients(self):
        inputs = _inputs(requires_grad=True)
        geometry, features, dst, inv = inputs
        output = environment_seed_scatter_reference(*inputs)
        expected = torch.zeros((4, 4, 7), dtype=torch.float32)
        for edge, node in enumerate(dst.tolist()):
            expected[node] += (
                geometry[edge, :, None] * features[edge, None, :] * inv[node]
            )
        torch.testing.assert_close(output, expected)
        upstream = torch.randn_like(output)
        grads = torch.autograd.grad(output, (geometry, features, inv), upstream)
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all())

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_environment_seed_scatter(*_inputs())

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        inputs = _inputs(device="cuda", requires_grad=True)
        upstream = torch.randn((4, 4, 7), dtype=torch.float32, device="cuda")
        actual = tilelang_environment_seed_scatter(*inputs)
        actual_grads = torch.autograd.grad(
            actual, (inputs[0], inputs[1], inputs[3]), upstream
        )
        ref_inputs = tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            if value.dtype.is_floating_point
            else value
            for value in inputs
        )
        expected = environment_seed_scatter_reference(*ref_inputs)
        expected_grads = torch.autograd.grad(
            expected, (ref_inputs[0], ref_inputs[1], ref_inputs[3]), upstream
        )
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=3.0e-4, atol=2.0e-5
            )


if __name__ == "__main__":
    unittest.main()
