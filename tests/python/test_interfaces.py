import unittest

import torch

from dcu_dpa4 import ops


class InterfaceTests(unittest.TestCase):
    def test_zero_distance_unit_has_zero_gradient(self):
        p = torch.zeros(2, 3, requires_grad=True)
        _, _, unit = ops.edge_geometry(p, torch.tensor([[1], [0]]),
            torch.zeros(1, 3, dtype=torch.int64), torch.eye(3)[None], torch.zeros(1, dtype=torch.int64))
        unit.sum().backward()
        torch.testing.assert_close(p.grad, torch.zeros_like(p))

    def test_batched_geometry_and_coordinate_gradient(self):
        pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], requires_grad=True)
        edges = torch.tensor([[1, 2], [0, 0]])
        cells = torch.eye(3).repeat(2, 1, 1)
        v, r, u = ops.edge_geometry(pos, edges, torch.tensor([[1, 0, 0], [0, 0, 0]]), cells, torch.tensor([1, 0]))
        torch.testing.assert_close(v, torch.tensor([[2., 0., 0.], [0., 1., 0.]]))
        r.sum().backward()
        torch.testing.assert_close(pos.grad.sum(0), torch.zeros(3))

    def test_segment_sum_and_gradient(self):
        x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
        y = ops.segment_sum(x, torch.tensor([0, 1, 0]), 3)
        torch.testing.assert_close(y, torch.tensor([[6., 8.], [3., 4.], [0., 0.]]))
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_mask_and_frame_virial(self):
        g = torch.ones(2, 3)
        force, virial = ops.force_virial(g, torch.ones_like(g), torch.tensor([[1, 2], [0, 0]]), torch.tensor([0, 1]), torch.tensor([True, False]), 3, 2)
        torch.testing.assert_close(force.sum(0), torch.zeros(3))
        torch.testing.assert_close(virial[0], -torch.ones(3, 3))
        torch.testing.assert_close(virial[1], torch.zeros(3, 3))

    def test_norm_and_linear_are_differentiable(self):
        x = torch.randn(5, 7, requires_grad=True)
        scale = torch.ones(7, requires_grad=True)
        w = torch.randn(7, 4, requires_grad=True)
        ops.linear(ops.rmsnorm(x, scale), w).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(w.grad).all())

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            ops.radial(torch.ones(1, 1), torch.ones(1, 1), 6., backend="auto")


if __name__ == "__main__":
    unittest.main()
