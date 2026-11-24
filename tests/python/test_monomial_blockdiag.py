"""Host tests for the two C7 operators' Python surface and oracles.

Nothing here needs a DCU: the oracles are checked against hand-computed values,
the validators are checked against malformed input, and the availability probes
must report False on a host without the extension.
"""

from __future__ import annotations

import unittest

import torch

import kernels
from kernels.blockdiag import blockdiag_available, blockdiag_gemm
from kernels.monomial import monomial_runs, monomial_runs_available
from reference.blockdiag import blockdiag_gemm_reference
from reference.monomial import monomial_runs_reference


class MonomialOracleTests(unittest.TestCase):
    def test_a_constant_monomial_is_one(self):
        q = torch.randn(8, 4)
        exps = torch.zeros(1, 4, dtype=torch.int8)
        coeff = torch.ones(1, 1)
        torch.testing.assert_close(
            monomial_runs_reference(q, exps, coeff), torch.ones(8, 1)
        )

    def test_a_linear_monomial_selects_a_component(self):
        q = torch.randn(5, 4)
        exps = torch.tensor([[0, 1, 0, 0]], dtype=torch.int8)
        torch.testing.assert_close(
            monomial_runs_reference(q, exps, torch.ones(1, 1)), q[:, 1:2]
        )

    def test_a_unit_quaternion_keeps_the_basis_bounded(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        exps = torch.tensor([[6, 0, 0, 0]], dtype=torch.int8)
        self.assertAlmostEqual(
            float(monomial_runs_reference(q, exps, torch.ones(1, 1))[0, 0]), 1.0
        )


class BlockDiagonalOracleTests(unittest.TestCase):
    def test_one_block_is_a_plain_matmul(self):
        x = torch.randn(6, 4)
        w = torch.randn(1, 4, 4)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        torch.testing.assert_close(
            blockdiag_gemm_reference(x, w, offsets), x @ w[0].t()
        )

    def test_blocks_do_not_talk(self):
        # A zero block must contribute nothing, whatever sits beside it.
        x = torch.ones(3, 6)
        weights = torch.zeros(2, 3, 3)
        weights[0] = torch.eye(3)
        offsets = torch.tensor([0, 3, 6], dtype=torch.int64)
        out = blockdiag_gemm_reference(x, weights, offsets)
        torch.testing.assert_close(out[:, :3], x[:, :3])
        torch.testing.assert_close(out[:, 3:], torch.zeros(3, 3))


class SurfaceTests(unittest.TestCase):
    def test_the_operators_are_exported(self):
        for name in (
            "monomial_runs",
            "monomial_runs_backward",
            "monomial_runs_available",
            "blockdiag_gemm",
            "blockdiag_gemm_backward",
            "blockdiag_available",
        ):
            self.assertIn(name, kernels.__all__)

    def test_availability_is_false_without_the_extension(self):
        if torch.cuda.is_available() and torch.version.hip is not None:
            self.skipTest("this host can actually load the extension")
        self.assertFalse(monomial_runs_available())
        self.assertFalse(blockdiag_available())

    def test_validators_reject_malformed_input(self):
        q = torch.randn(4, 4)
        exps = torch.zeros(2, 4, dtype=torch.int8)
        coeff = torch.zeros(3, 2)
        with self.assertRaises(TypeError):
            monomial_runs(q.double(), exps, coeff)
        with self.assertRaises(TypeError):
            monomial_runs(q, exps.to(torch.int32), coeff)
        with self.assertRaises(ValueError):
            monomial_runs(q, exps, torch.zeros(3, 3))

        weights = torch.zeros(2, 3, 3)
        offsets = torch.tensor([0, 3, 6], dtype=torch.int64)
        with self.assertRaises(ValueError):
            blockdiag_gemm(torch.randn(4, 5), weights, offsets)
        with self.assertRaises(TypeError):
            blockdiag_gemm(torch.randn(4, 6), weights, offsets.to(torch.int32))
        with self.assertRaises(ValueError):
            blockdiag_gemm(torch.randn(4, 6), weights, torch.tensor([0, 3], dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
