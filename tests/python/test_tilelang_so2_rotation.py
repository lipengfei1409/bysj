"""Tests for independent TileLang DPA4 SO2 rotation operators."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops.so2_rotate_back_l1 import (
    rotate_back_l1_reference,
    tilelang_rotate_back_l1,
)
from kernels.tilelang_ops.so2_rotate_to_local_l1 import (
    rotate_to_local_l1_reference,
    tilelang_rotate_to_local_l1,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class SO2RotationTests(unittest.TestCase):
    def test_local_rotation_reference_and_gradients(self):
        x = torch.randn((4, 4, 3), dtype=torch.float64, requires_grad=True)
        rotation = torch.randn((5, 4, 4), dtype=torch.float64, requires_grad=True)
        src = torch.tensor([0, 1, 1, 3, 2], dtype=torch.int64)
        actual = rotate_to_local_l1_reference(x, rotation, src)
        row_index = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
        expected = rotation.index_select(1, row_index) @ x.index_select(0, src)
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            actual, (x, rotation), upstream, retain_graph=True
        )
        expected_grads = torch.autograd.grad(expected, (x, rotation), upstream)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_rotate_back_reference_and_gradients(self):
        x_local = torch.randn((5, 2, 4, 3), dtype=torch.float64, requires_grad=True)
        rotation = torch.randn((5, 4, 4), dtype=torch.float64, requires_grad=True)
        actual = rotate_back_l1_reference(x_local, rotation)
        col_index = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
        expected = rotation.index_select(2, col_index) @ x_local.permute(
            0, 2, 1, 3
        ).reshape(5, 4, 6)
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            actual, (x_local, rotation), upstream, retain_graph=True
        )
        expected_grads = torch.autograd.grad(expected, (x_local, rotation), upstream)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_tilelang_ops_do_not_silently_fall_back_on_cpu(self):
        x = torch.zeros((2, 4, 3), dtype=torch.float32)
        rot = torch.eye(4, dtype=torch.float32).expand(2, 4, 4).contiguous()
        src = torch.tensor([0, 1], dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_rotate_to_local_l1(x, rot, src)
        local = torch.zeros((2, 1, 4, 3), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_rotate_back_l1(local, rot)

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_references(self):
        device = "cuda"
        x = torch.randn(
            (7, 4, 8), dtype=torch.float32, device=device, requires_grad=True
        )
        rotation = torch.randn(
            (11, 4, 4), dtype=torch.float32, device=device, requires_grad=True
        )
        src = torch.tensor(
            [0, 1, 1, 3, 2, 6, 0, 2, 4, 1, 5], dtype=torch.int64, device=device
        )
        upstream = torch.randn((11, 4, 8), dtype=torch.float32, device=device)
        actual = tilelang_rotate_to_local_l1(x, rotation, src)
        actual_grads = torch.autograd.grad(actual, (x, rotation), upstream)
        ref_x = x.detach().clone().requires_grad_(True)
        ref_rotation = rotation.detach().clone().requires_grad_(True)
        expected = rotate_to_local_l1_reference(ref_x, ref_rotation, src)
        expected_grads = torch.autograd.grad(expected, (ref_x, ref_rotation), upstream)
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=2.0e-4, atol=2.0e-5
            )

        local = torch.randn(
            (11, 1, 4, 8), dtype=torch.float32, device=device, requires_grad=True
        )
        rot_back = torch.randn(
            (11, 4, 4), dtype=torch.float32, device=device, requires_grad=True
        )
        actual_back = tilelang_rotate_back_l1(local, rot_back)
        back_upstream = torch.randn_like(actual_back)
        actual_back_grads = torch.autograd.grad(
            actual_back, (local, rot_back), back_upstream
        )
        ref_local = local.detach().clone().requires_grad_(True)
        ref_rot_back = rot_back.detach().clone().requires_grad_(True)
        expected_back = rotate_back_l1_reference(ref_local, ref_rot_back)
        expected_back_grads = torch.autograd.grad(
            expected_back, (ref_local, ref_rot_back), back_upstream
        )
        torch.testing.assert_close(actual_back, expected_back, rtol=2.0e-5, atol=2.0e-6)
        for actual_grad, expected_grad in zip(actual_back_grads, expected_back_grads):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=2.0e-4, atol=2.0e-5
            )


if __name__ == "__main__":
    unittest.main()
