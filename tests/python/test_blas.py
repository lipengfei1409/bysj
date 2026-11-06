import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from kernels import gemm
from kernels import native
from reference import gemm as reference_gemm


class GemmContractTests(unittest.TestCase):
    def test_reference_matches_torch_matmul(self):
        a = torch.randn(19, 7, dtype=torch.float32)
        b = torch.randn(7, 23, dtype=torch.float32)
        torch.testing.assert_close(reference_gemm(a, b), torch.matmul(a, b))

    def test_contract_and_signature(self):
        self.assertEqual(tuple(inspect.signature(gemm).parameters), ("a", "b"))
        self.assertFalse(hasattr(__import__("kernels"), "ops"))
        self.assertFalse(hasattr(__import__("kernels"), "blas"))

    def test_invalid_contracts(self):
        a, b = torch.ones(3, 5), torch.ones(5, 4)
        cases = (
            (([1], b), TypeError),
            ((a.double(), b.double()), TypeError),
            ((a, torch.ones(5)), ValueError),
            ((a, torch.ones(4, 5)), ValueError),
            ((a, torch.ones(5, 4, dtype=torch.float64)), TypeError),
            ((a, b.t()), ValueError),
        )
        for operator in (gemm, reference_gemm):
            for args, error in cases:
                with self.subTest(operator=operator.__module__, args=args):
                    with self.assertRaises(error):
                        operator(*args)

    def test_production_requires_dcu_and_never_uses_reference(self):
        a, b = torch.ones(2, 3), torch.ones(3, 4)
        with patch("kernels.native.require_dcu", side_effect=RuntimeError("DCU unavailable")):
            with self.assertRaisesRegex(RuntimeError, "DCU unavailable"):
                gemm(a, b)

    def test_native_calls_only_the_extension_gemm(self):
        a, b = torch.ones(2, 3), torch.ones(3, 4)
        expected = torch.full((2, 4), 3.)
        extension = SimpleNamespace(gemm=lambda lhs, rhs: expected)
        with patch("kernels.native.require_dcu", return_value=torch.device("cuda:0")), \
             patch("kernels.native._extension", return_value=extension):
            self.assertIs(gemm(a, b), expected)

    def test_autograd_is_rejected_for_native_contract(self):
        with self.assertRaisesRegex(RuntimeError, "does not support autograd"):
            gemm(torch.ones(2, 3, requires_grad=True), torch.ones(3, 4))

    def test_native_guard_rejects_non_hip_pytorch(self):
        with patch.object(torch.version, "hip", None):
            with self.assertRaisesRegex(RuntimeError, "HIP-enabled"):
                native.require_dcu()

    def test_native_guard_rejects_unsupported_architecture(self):
        with patch.object(torch.version, "hip", "test-hip"), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.get_device_properties",
                   return_value=SimpleNamespace(gcnArchName="gfx90a")):
            with self.assertRaisesRegex(RuntimeError, "unsupported DCU architecture"):
                native.require_dcu()


if __name__ == "__main__":
    unittest.main()
