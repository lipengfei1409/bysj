import itertools
import unittest

import torch


class BLASContractTests(unittest.TestCase):
    def api(self):
        from dcu_dpa4 import blas
        return blas

    def test_rectangular_transpose_alpha_beta_and_no_mutation(self):
        api = self.api()
        for ta, tb in itertools.product((False, True), repeat=2):
            a = torch.randn((5, 3) if ta else (3, 5), dtype=torch.float64)
            b = torch.randn((4, 5) if tb else (5, 4), dtype=torch.float64)
            c = torch.randn(3, 4, dtype=torch.float64)
            saved = c.clone()
            out = api.gemm(a, b, c, trans_a=ta, trans_b=tb, alpha=1.3, beta=-.7)
            expected = 1.3 * ((a.T if ta else a) @ (b.T if tb else b)) - .7 * c
            torch.testing.assert_close(out, expected)
            torch.testing.assert_close(c, saved)
            self.assertNotEqual(out.data_ptr(), c.data_ptr())

    def test_batched_noncontiguous_and_gradients(self):
        api = self.api()
        for ta, tb in itertools.product((False, True), repeat=2):
            a = torch.randn(2, 3, 5, dtype=torch.float64).transpose(-1, -2)
            # a physical [2,5,3]; logical shape depends on trans_a.
            m, k = (3, 5) if ta else (5, 3)
            b = torch.randn(2, 4, k, dtype=torch.float64) if tb else torch.randn(2, k, 4, dtype=torch.float64)
            a = a.requires_grad_(); b = b.requires_grad_()
            c = torch.randn(2, m, 4, dtype=torch.float64, requires_grad=True)
            out = api.gemm(a, b, c, trans_a=ta, trans_b=tb, alpha=.9, beta=.2)
            ref = .9 * ((a.transpose(-1, -2) if ta else a) @
                        (b.transpose(-1, -2) if tb else b)) + .2 * c
            g = torch.randn_like(ref)
            actual = torch.autograd.grad(out, (a, b, c), g, retain_graph=True)
            expected = torch.autograd.grad(ref, (a, b, c), g)
            torch.testing.assert_close(out, ref)
            for x, y in zip(actual, expected):
                torch.testing.assert_close(x, y)

    def test_beta_zero_does_not_read_nan_c(self):
        api = self.api()
        a, b = torch.ones(2, 3), torch.ones(3, 4)
        c = torch.full((2, 4), float('nan'), requires_grad=True)
        result = api.gemm(a, b, c, beta=0)
        torch.testing.assert_close(result, torch.full((2, 4), 3.))
        torch.testing.assert_close(torch.autograd.grad(result.sum(), c)[0], torch.zeros_like(c))

    def test_coefficients_follow_input_precision(self):
        api = self.api()
        a = torch.full((2, 3), float('nan'))
        b = torch.full((3, 4), float('nan'))
        # Both coefficients become zero when represented as FP32 BLAS scalars.
        c = torch.full((2, 4), float('nan'))
        torch.testing.assert_close(api.gemm(a, b, c, alpha=1e-100, beta=1e-100),
                                   torch.zeros(2, 4))

    def test_zero_k_zero_batch_and_zero_alpha(self):
        api = self.api()
        c = torch.randn(3, 4, requires_grad=True)
        out = api.gemm(torch.empty(3, 0), torch.empty(0, 4), c, beta=.5)
        torch.testing.assert_close(out, c * .5)
        self.assertEqual(api.gemm(torch.empty(0, 3, 5), torch.empty(0, 5, 4)).shape, (0, 3, 4))
        a = torch.full((3, 5), float('nan'), requires_grad=True)
        b = torch.full((5, 4), float('nan'), requires_grad=True)
        out = api.gemm(a, b, c, alpha=0, beta=.5)
        torch.testing.assert_close(out, c * .5)
        grads = torch.autograd.grad(out.sum(), (a, b, c))
        torch.testing.assert_close(grads[0], torch.zeros_like(a))
        torch.testing.assert_close(grads[1], torch.zeros_like(b))

    def test_linear_and_bias_gradient(self):
        api = self.api()
        x = torch.randn(7, 5, requires_grad=True)
        w = torch.randn(5, 11, requires_grad=True)
        bias = torch.randn(11, requires_grad=True)
        y = api.linear(x, w, bias)
        torch.testing.assert_close(y, x @ w + bias)
        torch.testing.assert_close(torch.autograd.grad(y.sum(), bias)[0], torch.full_like(bias, 7.))

    def test_invalid_contracts(self):
        api = self.api()
        a, b = torch.ones(3, 5), torch.ones(5, 4)
        for kwargs in ({'backend': 'hip'}, {'beta': 1}, {'alpha': float('nan')},
                       {'trans_a': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                api.gemm(a, b, **kwargs)
        for bad_a, bad_b in ((a.half(), b.half()), (a, b.double()),
                             (a, torch.ones(2, 5, 4)),
                             (torch.ones(2, 3, 5), torch.ones(1, 5, 4))):
            with self.assertRaises((ValueError, TypeError)):
                api.gemm(bad_a, bad_b)
        with self.assertRaises(ValueError):
            api.gemm(a, b, torch.ones(4))
        with self.assertRaises(ValueError):
            api.linear(a, b, torch.ones(1, 4))

    def test_rocblas_never_silently_falls_back(self):
        api = self.api()
        if not torch.version.hip:
            with self.assertRaisesRegex(RuntimeError, 'ROCm|DTK'):
                api.gemm(torch.ones(2, 3), torch.ones(3, 4), backend='rocblas')

    def test_autocast_is_explicitly_rejected(self):
        api = self.api()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with self.assertRaisesRegex(RuntimeError, 'autocast'):
                api.gemm(torch.ones(2, 3), torch.ones(3, 4))


if __name__ == '__main__':
    unittest.main()
