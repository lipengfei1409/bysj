"""Host tests for the megakernel SO(2) conv reference (C8).

Two things are checked here and nothing else can: that the two reference
implementations (the kernel-order loop and the batched matmuls) agree, so a
transcription slip cannot pass silently, and that the reference has the
properties the operator's header documents -- a rotation round-trip with zero
kernels is the identity, the gate multiplies the silu of the content, and the
envelope weight is applied after the inverse rotation.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

from reference.so2_conv_dmk import (
    so2_conv_dmk_kernel_order,
    so2_conv_dmk_reference,
)

ROOT = Path(__file__).resolve().parents[2]
OP_HEADER = ROOT / "csrc" / "so2_conv_dmk" / "so2_conv_op.h"


def _orthonormal(edges: int, dim: int, red: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A rotation whose transpose really is its inverse (red == dim needed)."""
    assert dim == red
    q, _ = torch.linalg.qr(torch.randn(edges, dim, dim, dtype=torch.float64))
    return q.to(torch.float32), q.transpose(1, 2).to(torch.float32)


def _problem(edges=6, dim=4, red=4, layers=2, channels=3, nodes=5, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(nodes, dim, channels)
    src = torch.randint(0, nodes, (edges,))
    rot, rot_t = _orthonormal(edges, dim, red)
    kc = torch.randn(edges, layers, 2, red, red) * 0.2
    alpha = torch.rand(edges) + 0.5
    return x, src, rot, rot_t, kc, alpha


class AgreementTests(unittest.TestCase):
    def test_the_loop_and_the_batched_form_agree(self) -> None:
        problem = _problem()
        loop = so2_conv_dmk_kernel_order(*problem)
        batched = so2_conv_dmk_reference(*problem)
        torch.testing.assert_close(loop, batched, rtol=1e-9, atol=1e-9)

    def test_a_wider_profiled_shape_also_agrees(self) -> None:
        # lmax=3: dim 16, red 10 is not square, so build the transpose directly.
        torch.manual_seed(3)
        edges, dim, red, channels = 4, 16, 10, 8
        x = torch.randn(3, dim, channels)
        src = torch.randint(0, 3, (edges,))
        rot = torch.randn(edges, red, dim) * 0.3
        rot_t = torch.randn(edges, dim, red) * 0.3
        kc = torch.randn(edges, 4, 2, red, red) * 0.1
        alpha = torch.rand(edges) + 0.5
        loop = so2_conv_dmk_kernel_order(x, src, rot, rot_t, kc, alpha)
        batched = so2_conv_dmk_reference(x, src, rot, rot_t, kc, alpha)
        torch.testing.assert_close(loop, batched, rtol=1e-9, atol=1e-9)


class PropertyTests(unittest.TestCase):
    def test_zero_kernels_turn_the_round_trip_into_the_identity(self) -> None:
        # With no mixing the pipeline is rotate -> rotate back -> weight, and an
        # orthonormal R makes that the identity, so msg = alpha * x[src].
        edges, dim, channels = 5, 4, 3
        torch.manual_seed(1)
        x = torch.randn(4, dim, channels)
        src = torch.randint(0, 4, (edges,))
        rot, rot_t = _orthonormal(edges, dim, dim)
        kc = torch.zeros(edges, 2, 2, dim, dim)
        alpha = torch.rand(edges) + 0.5
        out = so2_conv_dmk_reference(x, src, rot, rot_t, kc, alpha)
        expected = x[src].to(torch.float64) * alpha.to(torch.float64).view(-1, 1, 1)
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_a_zero_content_kernel_leaves_the_residual_alone(self) -> None:
        # silu(0) == 0, so a zero content kernel makes the gated term vanish
        # whatever the gate does -- the stack then leaves xl untouched.
        problem = list(_problem())
        kc = problem[4].clone()
        kc[:, :, 0, :, :] = 0.0  # content kernel; the gate stays random
        problem[4] = kc
        out = so2_conv_dmk_reference(*problem)
        x, src, rot, rot_t, _, alpha = problem
        identity = x[src].to(torch.float64) * alpha.to(torch.float64).view(-1, 1, 1)
        torch.testing.assert_close(out, identity, rtol=1e-5, atol=1e-5)

    def test_swapping_content_and_gate_changes_the_result(self) -> None:
        # Pins the order of the two kernels: a port that swaps them must fail.
        x, src, rot, rot_t, kc, alpha = _problem()
        plain = so2_conv_dmk_reference(x, src, rot, rot_t, kc, alpha)
        swapped = so2_conv_dmk_reference(
            x, src, rot, rot_t, kc.flip(2), alpha
        )
        self.assertFalse(torch.allclose(plain, swapped))

    def test_the_envelope_weight_is_applied_last(self) -> None:
        x, src, rot, rot_t, kc, alpha = _problem()
        full = so2_conv_dmk_reference(x, src, rot, rot_t, kc, alpha)
        ones = so2_conv_dmk_reference(x, src, rot, rot_t, kc, torch.ones_like(alpha))
        torch.testing.assert_close(
            full, ones * alpha.to(torch.float64).view(-1, 1, 1), rtol=1e-12, atol=1e-12
        )

    def test_the_output_shape_is_per_edge(self) -> None:
        x, src, rot, rot_t, kc, alpha = _problem(edges=7)
        out = so2_conv_dmk_reference(x, src, rot, rot_t, kc, alpha)
        self.assertEqual(tuple(out.shape), (7, x.shape[1], x.shape[2]))
        self.assertEqual(out.dtype, torch.float64)  # the oracle works in float64


class HeaderContractTests(unittest.TestCase):
    def test_the_page_budget_matches_the_reference_formula(self) -> None:
        # 2*red*dim + 2*layers*red*red + 1 floats per wave, pinned in the header.
        text = OP_HEADER.read_text()
        self.assertIn("2 * red * dim + 2 * layers * red * red + 1", text)
        self.assertIn("so2_conv_table_bytes(16, 10, 4) <= config::kPageSize", text)
        floats = 2 * 10 * 16 + 2 * 4 * 10 * 10 + 1
        self.assertEqual(floats, 1121)

    def test_the_documented_stage_order_matches_this_oracle(self) -> None:
        # The header's formula block and the reference's docstring must describe
        # the same five stages in the same order.
        text = OP_HEADER.read_text()
        stage_comments = [
            "rotate to the local frame",
            "block-diagonal content",
            "block-diagonal gate",
            "gated mixing + residual",
            "rotate back",
        ]
        positions = [text.find(s) for s in stage_comments]
        self.assertNotIn(-1, positions, positions)
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
