"""Contracts and explicit-reference checks for SO(2) envelope reduction."""
from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

import torch

from kernels.so2_reduce import so2_envelope_reduce, so2_envelope_reduce_available
from reference.so2 import so2_envelope_reduce_reference

_so2_module = importlib.import_module("kernels.so2_reduce")


def _inputs(*, requires_grad=False):
    message = torch.randn(5, 4, 3, dtype=torch.float32, requires_grad=requires_grad)
    edge_weight = torch.tensor(
        [0.2, 0.8, 0.0, 0.5, 0.3], dtype=torch.float32, requires_grad=requires_grad
    )
    dst = torch.tensor([2, 0, 2, 1, 0], dtype=torch.int64)
    inverse = torch.tensor([0.5, 0.75, 1.25], dtype=torch.float32).reshape(3, 1, 1)
    inverse.requires_grad_(requires_grad)
    return message, edge_weight, dst, inverse


def _fake_forward(message, edge_weight, dst, inverse):
    pre_norm = message.new_zeros((inverse.numel(), message.shape[1], message.shape[2]))
    if dst.numel():
        pre_norm.index_add_(0, dst, message * edge_weight[:, None, None])
    return pre_norm * inverse.reshape(-1, 1, 1), pre_norm


def _fake_backward(grad_output, message, edge_weight, dst, inverse, pre_norm):
    node = dst
    inverse_node = inverse.index_select(0, node)
    grad_message = (
        grad_output.index_select(0, node)
        * edge_weight[:, None, None]
        * inverse_node[:, None, None]
    )
    grad_weight = (
        grad_output.index_select(0, node) * message
    ).sum(dim=(1, 2)) * inverse_node
    grad_inverse = (grad_output * pre_norm).sum(dim=(1, 2))
    return grad_message, grad_weight, grad_inverse


class _FakeExtension:
    so2_envelope_reduce_forward = staticmethod(_fake_forward)
    so2_envelope_reduce_backward = staticmethod(_fake_backward)


class SO2EnvelopeReduceReferenceTests(unittest.TestCase):
    def test_reference_matches_destination_segment_sum_and_gradients(self):
        actual_inputs = _inputs(requires_grad=True)
        expected_inputs = tuple(
            value.detach().clone().requires_grad_(index != 2)
            if index != 2
            else value.clone()
            for index, value in enumerate(actual_inputs)
        )
        actual = so2_envelope_reduce_reference(*actual_inputs)
        expected = torch.zeros((3, 4, 3), dtype=torch.float32)
        message, edge_weight, dst, inverse = expected_inputs
        for edge, node in enumerate(dst.tolist()):
            expected[node] += message[edge] * edge_weight[edge]
        expected = expected * inverse
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            actual, (actual_inputs[0], actual_inputs[1], actual_inputs[3]), upstream
        )
        expected_grads = torch.autograd.grad(
            expected,
            (expected_inputs[0], expected_inputs[1], expected_inputs[3]),
            upstream,
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_empty_edges_return_zero_node_updates(self):
        message = torch.empty((0, 4, 3), dtype=torch.float32)
        edge_weight = torch.empty((0,), dtype=torch.float32)
        dst = torch.empty((0,), dtype=torch.int64)
        inverse = torch.ones((2, 1, 1), dtype=torch.float32)
        output = so2_envelope_reduce_reference(message, edge_weight, dst, inverse)
        self.assertEqual(tuple(output.shape), (2, 4, 3))
        self.assertEqual(output.count_nonzero().item(), 0)

    def test_reference_rejects_invalid_destination(self):
        message, edge_weight, dst, inverse = _inputs()
        with self.assertRaisesRegex(ValueError, "within the node range"):
            so2_envelope_reduce_reference(
                message, edge_weight, torch.full_like(dst, 4), inverse
            )


class SO2EnvelopeReduceWrapperTests(unittest.TestCase):
    def test_wrapper_dispatch_and_gradients_with_reference_extension(self):
        inputs = _inputs(requires_grad=True)
        actual = so2_envelope_reduce_reference(*inputs)
        upstream = torch.randn_like(actual)
        expected_grads = torch.autograd.grad(
            actual, (inputs[0], inputs[1], inputs[3]), upstream, retain_graph=True
        )
        with patch.object(_so2_module, "require_dcu"), patch.object(
            _so2_module, "_extension", return_value=_FakeExtension
        ):
            wrapped = so2_envelope_reduce(*inputs)
            actual_grads = torch.autograd.grad(
                wrapped, (inputs[0], inputs[1], inputs[3]), upstream
            )
        torch.testing.assert_close(wrapped, actual)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_invalid_contracts_fail_before_dcu_check(self):
        message, edge_weight, dst, inverse = _inputs()
        with self.assertRaisesRegex(ValueError, "edge_weight"):
            so2_envelope_reduce(message, edge_weight[:-1], dst, inverse)
        with self.assertRaisesRegex(TypeError, "dst must be int64"):
            so2_envelope_reduce(message, edge_weight, dst.to(torch.int32), inverse)
        with self.assertRaisesRegex(ValueError, "message must have shape"):
            so2_envelope_reduce(message.reshape(5, 12), edge_weight, dst, inverse)

    def test_valid_cpu_inputs_reach_dcu_guard(self):
        with self.assertRaisesRegex(RuntimeError, "HIP-enabled"):
            so2_envelope_reduce(*_inputs())

    def test_native_availability_is_false_without_hip(self):
        if torch.version.hip is None:
            self.assertFalse(so2_envelope_reduce_available())

    def test_native_dc_u_parity_and_gradients(self):
        if not so2_envelope_reduce_available():
            self.skipTest("requires a supported DCU and built HIP extension")
        inputs = tuple(
            value.to("cuda").detach().requires_grad_(index != 2)
            if index != 2
            else value.to("cuda")
            for index, value in enumerate(_inputs(requires_grad=True))
        )
        reference_inputs = tuple(
            value.detach().cpu().requires_grad_(index != 2)
            if index != 2
            else value.detach().cpu()
            for index, value in enumerate(inputs)
        )
        actual = so2_envelope_reduce(*inputs)
        expected = so2_envelope_reduce_reference(*reference_inputs).to("cuda")
        torch.testing.assert_close(actual, expected, rtol=2.0e-4, atol=2.0e-5)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            actual, (inputs[0], inputs[1], inputs[3]), upstream
        )
        expected_grads = torch.autograd.grad(
            expected,
            (reference_inputs[0], reference_inputs[1], reference_inputs[3]),
            upstream.cpu(),
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad.cpu(), expected_grad, rtol=3.0e-4, atol=3.0e-5
            )


if __name__ == "__main__":
    unittest.main()
