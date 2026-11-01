"""Tests for the standalone TileLang DPA4 GIE scatter operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops.zonal_scatter import (
    tilelang_zonal_scatter,
    zonal_scatter_reference,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


def _inputs(device="cpu", requires_grad=False):
    zonal = torch.tensor(
        [[0.2, -0.1, 0.3], [0.4, 0.5, -0.2], [-0.3, 0.6, 0.1], [0.7, -0.2, 0.4]],
        dtype=torch.float32,
        device=device,
        requires_grad=requires_grad,
    )
    radial = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 2, 3) / 10
    radial.requires_grad_(requires_grad)
    dst = torch.tensor([1, 0, 1, 1], dtype=torch.int64, device=device)
    radial_slot = torch.tensor([0, 0, 1], dtype=torch.int64, device=device)
    inv_sqrt_deg = torch.tensor([0.7, 0.5], dtype=torch.float32, device=device)
    inv_sqrt_deg.requires_grad_(requires_grad)
    return zonal, radial, dst, radial_slot, inv_sqrt_deg


class ZonalScatterTests(unittest.TestCase):
    def test_reference_matches_explicit_segment_sum_and_gradients(self):
        inputs = _inputs(requires_grad=True)
        zonal, radial, dst, radial_slot, inv_sqrt_deg = inputs
        output = zonal_scatter_reference(*inputs)
        expected = torch.zeros((2, 4, 3), dtype=torch.float32)
        for edge in range(4):
            for row, slot in enumerate(radial_slot.tolist()):
                expected[dst[edge], row + 1] += (
                    zonal[edge, row] * radial[edge, slot] * inv_sqrt_deg[dst[edge]]
                )
        torch.testing.assert_close(output, expected)

        upstream = torch.randn_like(output)
        grads = torch.autograd.grad(output, (zonal, radial, inv_sqrt_deg), upstream)
        for value in grads:
            self.assertTrue(torch.isfinite(value).all())

    def test_empty_edge_reference_returns_zero_l0_and_non_scalar_rows(self):
        zonal = torch.empty((0, 3), dtype=torch.float32)
        radial = torch.empty((0, 2, 4), dtype=torch.float32)
        dst = torch.empty((0,), dtype=torch.int64)
        slots = torch.tensor([0, 0, 1], dtype=torch.int64)
        inv_sqrt_deg = torch.ones((2,), dtype=torch.float32)
        output = zonal_scatter_reference(zonal, radial, dst, slots, inv_sqrt_deg)
        self.assertEqual(tuple(output.shape), (2, 4, 4))
        self.assertEqual(output.count_nonzero().item(), 0)

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        inputs = _inputs()
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_zonal_scatter(*inputs)

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        inputs = _inputs(device="cuda", requires_grad=True)
        upstream = torch.randn((2, 4, 3), dtype=torch.float32, device="cuda")
        actual = tilelang_zonal_scatter(*inputs)
        actual_grads = torch.autograd.grad(
            actual,
            (inputs[0], inputs[1], inputs[4]),
            upstream,
        )

        ref_inputs = tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            if value.dtype.is_floating_point
            else value
            for value in inputs
        )
        expected = zonal_scatter_reference(*ref_inputs)
        expected_grads = torch.autograd.grad(
            expected,
            (ref_inputs[0], ref_inputs[1], ref_inputs[4]),
            upstream,
        )
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=3.0e-4, atol=2.0e-5
            )


if __name__ == "__main__":
    unittest.main()
