"""Opt-in accuracy and gradient tests for the fused segmented HIP operators."""
import unittest

import torch

from kernels.segment import (
    force_assembly,
    group_by_source,
    segment_available,
    segment_offsets,
    segment_softmax,
    segment_sum,
)
from reference.segment import (
    force_assembly_reference,
    segment_softmax_reference,
    segment_sum_reference,
)


@unittest.skipUnless(
    segment_available() and torch.cuda.is_available(),
    "requires a built fused-segment HIP extension and a supported DCU",
)
class SegmentOperatorAccuracyTests(unittest.TestCase):
    device = "cuda"

    @staticmethod
    def _graph(group_sizes):
        dst = torch.repeat_interleave(
            torch.arange(len(group_sizes), dtype=torch.int64),
            torch.tensor(group_sizes, dtype=torch.int64),
        )
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.tensor(group_sizes).cumsum(0)])
        return dst, offsets

    def test_segment_sum_matches_reference_for_all_channel_widths(self):
        for channels in (1, 3, 64, 640):
            with self.subTest(channels=channels):
                dst, offsets = self._graph([2, 0, 3])
                dst, offsets = dst.to(self.device), offsets.to(self.device)
                src = torch.randn(5, channels, device=self.device)
                edge_scale = torch.rand(5, device=self.device) + 0.5
                node_scale = torch.rand(3, device=self.device) + 0.5
                actual = segment_sum(src, dst, offsets, edge_scale, node_scale)
                expected = segment_sum_reference(src, dst, offsets, edge_scale, node_scale)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                self.assertEqual(actual[1].count_nonzero().item(), 0)

    def test_segment_sum_gradients_match_reference(self):
        dst, offsets = self._graph([2, 1, 3])
        dst, offsets = dst.to(self.device), offsets.to(self.device)
        src = torch.randn(6, 8, device=self.device, requires_grad=True)
        edge_scale = (torch.rand(6, device=self.device) + 0.5).requires_grad_(True)
        node_scale = (torch.rand(3, device=self.device) + 0.5).requires_grad_(True)
        upstream = torch.randn(3, 8, device=self.device)
        actual = segment_sum(src, dst, offsets, edge_scale, node_scale)
        grads = torch.autograd.grad(actual, (src, node_scale), upstream)

        reference_src = src.detach().clone().requires_grad_(True)
        reference_scale = node_scale.detach().clone().requires_grad_(True)
        expected = segment_sum_reference(
            reference_src, dst, offsets, edge_scale.detach(), reference_scale
        )
        expected_grads = torch.autograd.grad(expected, (reference_src, reference_scale), upstream)
        torch.testing.assert_close(grads[0], expected_grads[0], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(grads[1], expected_grads[1], rtol=2e-5, atol=2e-6)

    def test_segment_softmax_forward_and_backward_match_reference(self):
        dst, offsets = self._graph([2, 0, 3, 1])
        dst, offsets = dst.to(self.device), offsets.to(self.device)
        logit = torch.randn(6, 16, device=self.device, requires_grad=True)
        env = torch.tensor([0.9, -0.2, 0.7, 0.0, 1.1, 0.3], device=self.device).requires_grad_(True)
        null_logit = torch.randn(4, 16, device=self.device, requires_grad=True)
        upstream = torch.randn(6, 16, device=self.device)
        upstream_null = torch.randn(4, 16, device=self.device)

        alpha, null_alpha = segment_softmax(logit, env, offsets, null_logit)
        grads = torch.autograd.grad((alpha, null_alpha), (logit, env, null_logit), (upstream, upstream_null))

        reference_logit = logit.detach().clone().requires_grad_(True)
        reference_env = env.detach().clone().requires_grad_(True)
        reference_null = null_logit.detach().clone().requires_grad_(True)
        expected_alpha, expected_null = segment_softmax_reference(
            reference_logit, reference_env, dst, offsets, reference_null
        )
        expected_grads = torch.autograd.grad(
            (expected_alpha, expected_null),
            (reference_logit, reference_env, reference_null),
            (upstream, upstream_null),
        )
        torch.testing.assert_close(alpha, expected_alpha, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(null_alpha, expected_null, rtol=2e-5, atol=2e-6)
        for actual_grad, expected_grad in zip(grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-4, atol=2e-5)
        per_segment = torch.zeros(4, 16, device=self.device).index_add_(0, dst, alpha)
        torch.testing.assert_close(per_segment + null_alpha, torch.ones(4, 16, device=self.device), rtol=2e-5, atol=2e-6)

    def test_segment_softmax_without_null_mass(self):
        dst, offsets = self._graph([2, 1, 3])
        dst, offsets = dst.to(self.device), offsets.to(self.device)
        logit = torch.randn(6, 8, device=self.device)
        env = torch.rand(6, device=self.device) + 0.5
        alpha = segment_softmax(logit, env, offsets)
        expected, _ = segment_softmax_reference(logit, env, dst, offsets)
        torch.testing.assert_close(alpha, expected, rtol=2e-5, atol=2e-6)
        per_segment = torch.zeros(3, 8, device=self.device).index_add_(0, dst, alpha)
        torch.testing.assert_close(per_segment, torch.ones(3, 8, device=self.device), rtol=2e-5, atol=2e-6)

    def test_force_assembly_matches_reference(self):
        devices = self.device
        dst, dst_offsets = self._graph([2, 1, 3, 0])
        dst, dst_offsets = dst.to(devices), dst_offsets.to(devices)
        src = torch.tensor([2, 0, 2, 1, 0, 2], dtype=torch.int64, device=devices)
        grad_edge = torch.randn(6, 3, device=devices)
        edge_vec = torch.randn(6, 3, device=devices)

        force, virial = force_assembly(grad_edge, edge_vec, src, dst, dst_offsets)
        order, src_offsets = group_by_source(src, 4)
        expected_force, expected_virial = force_assembly_reference(
            grad_edge, edge_vec, src, dst, dst_offsets, src_offsets
        )
        torch.testing.assert_close(force, expected_force, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(virial, expected_virial, rtol=2e-5, atol=2e-6)
        self.assertEqual(force[3].count_nonzero().item(), 0)

    def test_large_graph_segment_sum_matches_float64_ground_truth(self):
        # At ~60 edges per atom the ATen reference is not a bit baseline: it
        # sums with row atomics while the kernel sums in ascending order, so an
        # fp32 comparison has to be made against a float64 accumulation with an
        # fp32 error budget (~sqrt(K) * eps * sum|terms|).
        torch.manual_seed(11)
        atoms, edges, channels = 64, 4096, 32
        dst = torch.sort(torch.randint(0, atoms, (edges,), device=self.device)).values
        offsets = segment_offsets(dst, atoms)
        src = torch.randn(edges, channels, device=self.device)
        node_scale = torch.rand(atoms, device=self.device) + 0.5
        actual = segment_sum(src, dst, offsets, None, node_scale)

        expected = torch.zeros((atoms, channels), dtype=torch.float64, device=self.device)
        expected.index_add_(0, dst, src.double())
        expected = (expected * node_scale.double().unsqueeze(1)).float()
        budget = 1e-5 * float(expected.abs().max())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=budget)

    def test_large_grouped_graph_matches_reference(self):
        torch.manual_seed(7)
        atoms, edges, channels = 64, 4096, 32
        dst = torch.sort(torch.randint(0, atoms, (edges,), device=self.device)).values
        offsets = segment_offsets(dst, atoms)
        src = torch.randint(0, atoms, (edges,), device=self.device)
        src_order, src_offsets = group_by_source(src, atoms)
        grad_edge = torch.randn(edges, 3, device=self.device)
        edge_vec = torch.randn(edges, 3, device=self.device)

        force, virial = force_assembly(
            grad_edge, edge_vec, src, dst, offsets, src_offsets, src_order
        )
        expected_force, expected_virial = force_assembly_reference(
            grad_edge, edge_vec, src, dst, offsets, src_offsets
        )
        torch.testing.assert_close(force, expected_force, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(virial, expected_virial, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
