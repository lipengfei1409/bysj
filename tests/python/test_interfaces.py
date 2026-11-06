import unittest
from unittest.mock import patch

import torch

from kernels.gemm import gemm


class GemmInterfaceTests(unittest.TestCase):
    def test_invalid_type_and_noncontiguous_inputs_fail_before_runtime_check(self):
        with self.assertRaisesRegex(TypeError, "must be tensors"):
            gemm(None, torch.ones(2, 2))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            gemm(torch.ones(3, 5), torch.ones(4, 5).t())

    def test_cpu_input_reaches_dcu_guard(self):
        with patch("kernels.native._require_hip_runtime",
                   side_effect=RuntimeError("DCU unavailable")):
            with self.assertRaisesRegex(RuntimeError, "DCU unavailable"):
                gemm(torch.ones(2, 3), torch.ones(3, 4))


if __name__ == "__main__":
    unittest.main()
