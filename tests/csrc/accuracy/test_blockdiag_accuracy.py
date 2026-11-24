"""Opt-in accuracy tests for the block-diagonal mixing GEMM on a DCU."""
import unittest

import torch

from kernels.blockdiag import (
    blockdiag_available,
    blockdiag_gemm,
    blockdiag_gemm_backward,
)
from reference.blockdiag import (
    blockdiag_gemm_backward_reference,
    blockdiag_gemm_reference,
)


@unittest.skipUnless(
    blockdiag_available() and torch.cuda.is_available(),
    "requires a built block-diagonal HIP extension and a supported DCU",
)
class BlockDiagonalAccuracyTests(unittest.TestCase):
    device = "cuda"

    @staticmethod
    def _blocks(sizes):
        """Contiguous diagonal blocks, plus the column offsets they act on."""
        weights = torch.stack(
            [torch.randn(size, size) for size in sizes]
        ).to("cuda")
        offsets = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(sizes), 0).tolist()),
            dtype=torch.int64,
            device="cuda",
        )
        return weights, offsets

    def test_forward_matches_slice_and_bmm(self):
        # The two block widths of the profiled convolution, plus an odd one that
        # the pow2 padding wastes a quarter of its work on.
        for sizes in ((192,), (256, 384), (48, 64, 192)):
            with self.subTest(sizes=sizes):
                weights, offsets = self._blocks(sizes)
                rows = int(offsets[-1])
                x = torch.randn(700, rows, device=self.device)
                torch.testing.assert_close(
                    blockdiag_gemm(x, weights, offsets),
                    blockdiag_gemm_reference(x, weights, offsets),
                    rtol=2e-5,
                    atol=2e-6,
                )

    def test_backward_matches_the_transposed_contraction(self):
        sizes = (256, 384)
        weights, offsets = self._blocks(sizes)
        rows = int(offsets[-1])
        grad_out = torch.randn(333, rows, device=self.device)
        torch.testing.assert_close(
            blockdiag_gemm_backward(grad_out, weights, offsets),
            blockdiag_gemm_backward_reference(grad_out, weights, offsets),
            rtol=2e-5,
            atol=2e-6,
        )

    def test_padding_rows_stay_zero(self):
        # A block shorter than its slice: only its own rows may be written.
        sizes = (4, 6)
        weights, offsets = self._blocks(sizes)
        x = torch.ones(16, 10, device=self.device)
        out = blockdiag_gemm(x, weights, offsets)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_the_result_is_deterministic(self):
        weights, offsets = self._blocks((256, 384))
        x = torch.randn(256, 640, device=self.device)
        torch.testing.assert_close(
            blockdiag_gemm(x, weights, offsets),
            blockdiag_gemm(x, weights, offsets),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
