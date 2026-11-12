"""Opt-in native GIE zonal-scatter accuracy and gradient tests for a DCU."""
import unittest

import torch

from kernels.gie import gie_available, zonal_scatter
from reference.gie import zonal_scatter_reference


@unittest.skipUnless(
    gie_available() and torch.cuda.is_available(),
    "requires a built GIE HIP extension and supported DCU",
)
class GIEZonalScatterAccuracyTests(unittest.TestCase):
    @staticmethod
    def _inputs(device="cuda", degree_shape=(2,), requires_grad=False):
        zonal = torch.tensor(
            [[0.2, -0.1, 0.3], [0.4, 0.5, -0.2], [-0.3, 0.6, 0.1], [0.7, -0.2, 0.4]],
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )
        radial = (torch.arange(24, dtype=torch.float32, device=device).reshape(4, 2, 3) / 10)
        radial.requires_grad_(requires_grad)
        dst = torch.tensor([1, 0, 1, 1], dtype=torch.int64, device=device)
        radial_slot = torch.tensor([0, 0, 1], dtype=torch.int64, device=device)
        inverse = torch.tensor([0.7, 0.5], dtype=torch.float32, device=device).reshape(degree_shape)
        inverse.requires_grad_(requires_grad)
        return zonal, radial, dst, radial_slot, inverse

    def test_forward_and_backward_with_colliding_destinations(self):
        inputs = self._inputs(requires_grad=True)
        upstream = torch.randn(2, 4, 3, device="cuda")
        actual = zonal_scatter(*inputs)
        actual_grads = torch.autograd.grad(actual, (inputs[0], inputs[1], inputs[4]), upstream)

        reference_inputs = tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            if value.dtype.is_floating_point else value
            for value in inputs
        )
        expected = zonal_scatter_reference(*reference_inputs)
        expected_grads = torch.autograd.grad(
            expected, (reference_inputs[0], reference_inputs[1], reference_inputs[4]), upstream
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(actual[:, 0, :].count_nonzero().item(), 0)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=4e-4, atol=2e-5)

    def test_three_dimensional_inverse_degree_layout(self):
        inputs = self._inputs(degree_shape=(2, 1, 1))
        actual = zonal_scatter(*inputs)
        expected = zonal_scatter_reference(*inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_empty_edge_case(self):
        zonal = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        radial = torch.empty((0, 2, 4), dtype=torch.float32, device="cuda")
        dst = torch.empty((0,), dtype=torch.int64, device="cuda")
        slots = torch.tensor([0, 0, 1], dtype=torch.int64, device="cuda")
        inverse = torch.ones((2,), dtype=torch.float32, device="cuda")
        result = zonal_scatter(zonal, radial, dst, slots, inverse)
        self.assertEqual(tuple(result.shape), (2, 4, 4))
        self.assertEqual(result.count_nonzero().item(), 0)

    def test_energy_force_style_gradient(self):
        coords = torch.randn(4, 3, device="cuda", dtype=torch.float32, requires_grad=True)
        radial = torch.randn(4, 2, 5, device="cuda", dtype=torch.float32)
        dst = torch.tensor([1, 0, 1, 1], device="cuda", dtype=torch.int64)
        radial_slot = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int64)
        inverse = torch.tensor([0.8, 0.6], device="cuda", dtype=torch.float32)
        zonal = torch.sin(coords) + 0.1 * coords
        energy = zonal_scatter(zonal, radial, dst, radial_slot, inverse).square().sum()
        actual_force = -torch.autograd.grad(energy, coords)[0]

        coords_ref = coords.detach().clone().requires_grad_(True)
        expected = zonal_scatter_reference(
            torch.sin(coords_ref) + 0.1 * coords_ref, radial, dst, radial_slot, inverse
        ).square().sum()
        expected_force = -torch.autograd.grad(expected, coords_ref)[0]
        torch.testing.assert_close(actual_force, expected_force, rtol=4e-4, atol=3e-5)

    def test_current_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            inputs = self._inputs()
            actual = zonal_scatter(*inputs)
            expected = zonal_scatter_reference(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
