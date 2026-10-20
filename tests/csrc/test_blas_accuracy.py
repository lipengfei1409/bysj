"""Opt-in real DCU tests: missing native/library/device is failure when requested."""
import itertools
import os
import unittest

import torch
from dcu_dpa4 import blas


@unittest.skipUnless(os.environ.get('DPA4_RUN_ROCBLAS_TESTS') == '1',
                     'set DPA4_RUN_ROCBLAS_TESTS=1 with built rocBLAS extension on DCU')
class ROCBLASAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        blas._rocblas_extension()
        if not torch.cuda.is_available():
            raise RuntimeError('rocBLAS test requested but DCU is unavailable')
        cls.old_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')

    @classmethod
    def tearDownClass(cls):
        torch.set_float32_matmul_precision(cls.old_precision)

    def test_transposes_batched_values_and_gradients(self):
        for dtype, batch, ta, tb in itertools.product(
                (torch.float32, torch.float64), (None, 3), (False, True), (False, True)):
            with self.subTest(dtype=dtype, batch=batch, ta=ta, tb=tb):
                prefix = () if batch is None else (batch,)
                a0 = torch.randn(*prefix, *((17, 11) if ta else (11, 17)), dtype=dtype, device='cuda')
                b0 = torch.randn(*prefix, *((7, 17) if tb else (17, 7)), dtype=dtype, device='cuda')
                c0 = torch.randn(*prefix, 11, 7, dtype=dtype, device='cuda')
                g = torch.randn_like(c0)
                results = []
                for backend in ('reference', 'rocblas'):
                    a, b, c = [x.clone().requires_grad_() for x in (a0, b0, c0)]
                    out = blas.gemm(a, b, c, trans_a=ta, trans_b=tb, alpha=1.3, beta=-.7,
                                    backend=backend)
                    results.append((out, *torch.autograd.grad(out, (a, b, c), g)))
                    torch.testing.assert_close(c, c0)
                tolerance = 3e-4 if dtype == torch.float32 else 3e-12
                for actual, expected in zip(results[1], results[0]):
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    def test_noncontiguous_linear_and_current_stream(self):
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        outputs = []
        for stream in streams:
            with torch.cuda.stream(stream):
                x = torch.randn(5, 31, device='cuda').T
                w = torch.randn(11, 5, device='cuda').T
                b = torch.randn(11, device='cuda')
                actual = blas.linear(x, w, b, backend='rocblas')
                expected = x @ w + b
                outputs.append((actual, expected))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for actual, expected in outputs:
            torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

    def test_degenerate_nan_and_empty_inputs(self):
        for dtype in (torch.float32, torch.float64):
            c = torch.randn(2, 3, dtype=dtype, device='cuda')
            y = blas.gemm(torch.empty(2, 0, dtype=dtype, device='cuda'),
                          torch.empty(0, 3, dtype=dtype, device='cuda'), c,
                          beta=.5, backend='rocblas')
            torch.testing.assert_close(y, c * .5)
            a = torch.full((2, 5), float('nan'), dtype=dtype, device='cuda')
            b = torch.full((5, 3), float('nan'), dtype=dtype, device='cuda')
            torch.testing.assert_close(blas.gemm(a, b, c, alpha=0, beta=.5, backend='rocblas'), c * .5)
            if dtype == torch.float32:
                torch.testing.assert_close(blas.gemm(a, b, torch.full_like(c, float('nan')),
                                                   alpha=1e-100, beta=1e-100, backend='rocblas'),
                                           torch.zeros_like(c))
            torch.testing.assert_close(blas.gemm(torch.ones_like(a), torch.ones_like(b),
                                       torch.full_like(c, float('nan')), beta=0, backend='rocblas'),
                                       torch.full_like(c, 5.))
            y = blas.gemm(torch.empty(0, 2, 5, dtype=dtype, device='cuda'),
                          torch.empty(0, 5, 3, dtype=dtype, device='cuda'), backend='rocblas')
            self.assertEqual(y.shape, (0, 2, 3))

    def test_multiple_devices_when_available(self):
        if torch.cuda.device_count() < 2:
            self.skipTest('requires two DCUs to verify per-device handle isolation')
        for device in (1, 0, 1):
            a = torch.randn(5, 13, device=f'cuda:{device}')
            b = torch.randn(13, 7, device=f'cuda:{device}')
            torch.testing.assert_close(blas.gemm(a, b, backend='rocblas'), a @ b,
                                       atol=3e-4, rtol=3e-4)


if __name__ == '__main__':
    unittest.main()
