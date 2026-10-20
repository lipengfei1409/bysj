import math
import unittest

import torch

from dcu_dpa4.geometry import edge_vectors
from dcu_dpa4.radial import c3_cutoff_envelope, radial_basis
from dcu_dpa4.reduce import edge_force_virial


class GeometryTests(unittest.TestCase):
    def test_periodic_edge_vectors_follow_src_dst_convention(self):
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        edge_index = torch.tensor([[1], [0]], dtype=torch.long)
        shifts = torch.tensor([[1, 0, 0]], dtype=torch.long)
        cell = torch.diag(torch.tensor([2.0, 2.0, 2.0]))

        actual = edge_vectors(positions, edge_index, shifts, cell)

        torch.testing.assert_close(actual, torch.tensor([[3.0, 0.0, 0.0]]))

    def test_nonperiodic_edge_vectors_reject_nonzero_shifts(self):
        positions = torch.zeros((2, 3))
        with self.assertRaises(ValueError):
            edge_vectors(
                positions,
                torch.tensor([[0], [1]]),
                torch.tensor([[1, 0, 0]]),
                None,
            )


class RadialTests(unittest.TestCase):
    def test_c3_cutoff_has_expected_endpoints(self):
        distances = torch.tensor([[0.0], [3.0], [6.0], [7.0]])
        output = c3_cutoff_envelope(distances, rcut=6.0, exponent=7)
        self.assertEqual(output.shape, (4, 1))
        self.assertEqual(output[0, 0].item(), 1.0)
        self.assertEqual(output[2, 0].item(), 0.0)
        self.assertEqual(output[3, 0].item(), 0.0)
        self.assertGreater(output[1, 0].item(), 0.0)

    def test_bessel_basis_is_finite_at_zero(self):
        frequencies = torch.tensor([[math.pi / 6.0, 2.0 * math.pi / 6.0]])
        output = radial_basis(
            torch.tensor([[0.0], [6.0]]),
            frequencies,
            rcut=6.0,
            exponent=7,
        )
        torch.testing.assert_close(output[0], frequencies[0])
        torch.testing.assert_close(output[1], torch.zeros(2))


class ReductionTests(unittest.TestCase):
    def test_edge_gradient_produces_force_and_virial(self):
        edge_index = torch.tensor([[1], [0]], dtype=torch.long)
        gradients = torch.tensor([[2.0, 0.0, 0.0]])
        vectors = torch.tensor([[3.0, 0.0, 0.0]])

        force, virial = edge_force_virial(
            gradients, vectors, edge_index, n_atoms=2
        )

        torch.testing.assert_close(force, torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]))
        expected_virial = torch.zeros((3, 3))
        expected_virial[0, 0] = -6.0
        torch.testing.assert_close(virial, expected_virial)

    def test_empty_edges_return_zero_force_and_virial(self):
        force, virial = edge_force_virial(
            torch.empty((0, 3)),
            torch.empty((0, 3)),
            torch.empty((2, 0), dtype=torch.long),
            n_atoms=2,
        )
        torch.testing.assert_close(force, torch.zeros((2, 3)))
        torch.testing.assert_close(virial, torch.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
