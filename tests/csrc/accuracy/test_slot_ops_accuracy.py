"""Opt-in accuracy tests for the slot-axis operators on a DCU."""
import unittest

import torch

from kernels.slot import slot_available, slot_pack_scale, slot_reduce
from reference.slot import slot_pack_scale_reference, slot_reduce_reference


@unittest.skipUnless(
    slot_available() and torch.cuda.is_available(),
    "requires a built slot HIP extension and a supported DCU",
)
class SlotOperatorAccuracyTests(unittest.TestCase):
    device = "cuda"

    @staticmethod
    def _slot(rows_out, rows_in, *, reserved):
        """A pack map with an optional zero row at index 0."""
        slot = (torch.arange(rows_out, dtype=torch.int64) % rows_in)
        if reserved:
            slot[0] = -1
        return slot

    def test_pack_matches_reference_for_every_channel_width(self):
        for channels in (1, 3, 64, 640):
            with self.subTest(channels=channels):
                edges, rows_in, rows_out = 512, 3, 10
                src = torch.randn(edges, rows_in, channels, device=self.device)
                slot = self._slot(rows_out, rows_in, reserved=True).to(self.device)
                actual = slot_pack_scale(src, slot)
                expected = slot_pack_scale_reference(src, slot)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual[:, 0].count_nonzero().item(), 0)

    def test_pack_with_scale_matches_reference(self):
        edges, rows_in, rows_out, channels = 256, 3, 10, 64
        src = torch.randn(edges, rows_in, channels, device=self.device)
        slot = self._slot(rows_out, rows_in, reserved=True).to(self.device)
        scale = torch.rand(edges, rows_out, channels, device=self.device) + 0.1
        actual = slot_pack_scale(src, slot, scale)
        expected = slot_pack_scale_reference(src, slot, scale)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_empty_edge_list_produces_an_empty_tensor(self):
        src = torch.zeros(0, 3, 8, device=self.device)
        slot = self._slot(10, 3, reserved=True).to(self.device)
        out = slot_pack_scale(src, slot)
        self.assertEqual(tuple(out.shape), (0, 10, 8))

    def test_reduce_matches_reference(self):
        for channels in (1, 64, 640):
            with self.subTest(channels=channels):
                edges, rows = 512, 10
                src = torch.randn(edges, rows, channels, device=self.device)
                weight = torch.rand(edges, rows, device=self.device) + 0.1
                torch.testing.assert_close(
                    slot_reduce(src), slot_reduce_reference(src), rtol=2e-5, atol=2e-6
                )
                torch.testing.assert_close(
                    slot_reduce(src, weight),
                    slot_reduce_reference(src, weight),
                    rtol=2e-4,
                    atol=2e-5,
                )

    def test_autograd_flows_through_both_operators(self):
        edges, rows_in, rows_out, channels = 64, 3, 10, 32
        src = torch.randn(edges, rows_in, channels, device=self.device, requires_grad=True)
        slot = self._slot(rows_out, rows_in, reserved=True).to(self.device)
        scale = torch.rand(edges, rows_out, channels, device=self.device, requires_grad=True)
        packed = slot_pack_scale(src, slot, scale)
        reduced = slot_reduce(packed, torch.rand(edges, rows_out, device=self.device))
        grad_src, grad_scale = torch.autograd.grad(
            reduced.sum(), (src, scale), allow_unused=False
        )
        self.assertTrue(bool(torch.isfinite(grad_src).all()))
        self.assertTrue(bool(torch.isfinite(grad_scale).all()))
        self.assertEqual(grad_scale[:, 0].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
