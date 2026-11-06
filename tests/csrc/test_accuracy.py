"""Opt-in DUMMA GEMM correctness tests; run with ROCm PyTorch on DCU."""
import unittest

import torch

from kernels import gemm
from kernels.native import native_available
from reference import gemm as reference_gemm


@unittest.skipUnless(native_available() and torch.cuda.is_available(),
                     "requires built DUMMA extension and supported DCU")
class DUMMAGemmAccuracyTests(unittest.TestCase):
    def test_tile_and_tail_dimensions(self):
        for m, k, n in ((16, 4, 16), (19, 7, 23), (1, 1, 3)):
            with self.subTest(shape=(m, k, n)):
                a = torch.randn(m, k, device="cuda", dtype=torch.float32)
                b = torch.randn(k, n, device="cuda", dtype=torch.float32)
                actual = gemm(a, b)
                expected = reference_gemm(a, b)
                torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)

    def test_zero_inner_dimension(self):
        a = torch.empty(3, 0, device="cuda", dtype=torch.float32)
        b = torch.empty(0, 5, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(gemm(a, b), torch.zeros(3, 5, device="cuda"))

    def test_current_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            a = torch.randn(23, 19, device="cuda", dtype=torch.float32)
            b = torch.randn(19, 17, device="cuda", dtype=torch.float32)
            actual = gemm(a, b)
            expected = reference_gemm(a, b)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


if __name__ == "__main__":
    unittest.main()
