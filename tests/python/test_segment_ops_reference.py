"""Host-side contracts and closed-form gradient checks for the fused segment ops.

The HIP operators are only compiled on a DCU, but their mathematics is fixed
here: every reference composition is checked against an independent loop, and
every closed-form backward implemented by the kernels is checked against the
autograd gradient of the reference chain. A CPU-only machine therefore still
verifies the specification that the device tests compare against.
"""
import unittest

import torch

from kernels.segment import group_by_source, segment_offsets
from reference.segment import (
    force_assembly_reference,
    segment_softmax_reference,
    segment_sum_reference,
)


def _graph(group_sizes):
    """Destination-grouped edge list with the requested per-atom edge counts."""
    dst = torch.repeat_interleave(
        torch.arange(len(group_sizes), dtype=torch.int64),
        torch.tensor(group_sizes, dtype=torch.int64),
    )
    return dst, torch.cat([torch.zeros(1, dtype=torch.int64), torch.tensor(group_sizes).cumsum(0)])


class OffsetsTests(unittest.TestCase):
    def test_offsets_match_group_sizes(self):
        dst, offsets = _graph([2, 0, 3])
        self.assertTrue(torch.equal(segment_offsets(dst, 3), offsets))

    def test_group_by_source_is_stable_and_complete(self):
        src = torch.tensor([2, 0, 2, 1, 0], dtype=torch.int64)
        order, offsets = group_by_source(src, 3)
        self.assertTrue(torch.equal(offsets, torch.tensor([0, 2, 3, 5], dtype=torch.int64)))
        self.assertTrue(torch.equal(src[order], torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)))
        # stable: edges 1 and 4 keep their relative order inside group 0
        self.assertEqual(order.tolist()[:2], [1, 4])

    def test_rejects_out_of_range_destinations(self):
        with self.assertRaises(ValueError):
            segment_offsets(torch.tensor([0, 3], dtype=torch.int64), 3)


class SegmentSumReferenceTests(unittest.TestCase):
    def test_matches_manual_loop(self):
        dst, offsets = _graph([2, 1, 3])
        src = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        edge_scale = torch.linspace(0.5, 1.5, 6)
        node_scale = torch.tensor([1.0, 2.0, 3.0])
        actual = segment_sum_reference(src, dst, offsets, edge_scale, node_scale)

        expected = torch.zeros(3, 2)
        for node in range(3):
            for edge in range(offsets[node], offsets[node + 1]):
                expected[node] += edge_scale[edge] * src[edge]
            expected[node] *= node_scale[node]
        torch.testing.assert_close(actual, expected)

    def test_empty_group_stays_zero(self):
        dst, offsets = _graph([0, 2, 0])
        src = torch.ones(2, 3)
        actual = segment_sum_reference(src, dst, offsets)
        self.assertEqual(actual[0].count_nonzero().item(), 0)
        self.assertEqual(actual[2].count_nonzero().item(), 0)
        torch.testing.assert_close(actual[1], torch.full((3,), 2.0))

    def test_rejects_offsets_that_do_not_match_dst(self):
        dst, _ = _graph([2, 1, 3])
        with self.assertRaises(ValueError):
            segment_sum_reference(torch.ones(6, 2), dst, torch.tensor([0, 3, 5, 6]))

    def test_closed_form_backward_matches_autograd(self):
        dst, offsets = _graph([2, 1, 3])
        src = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 7
        edge_scale = torch.linspace(0.5, 1.5, 6)
        node_scale = torch.tensor([1.0, 2.0, 3.0])
        src.requires_grad_(True)
        edge_scale.requires_grad_(True)
        node_scale.requires_grad_(True)
        upstream = torch.randn(3, 2)
        out = segment_sum_reference(src, dst, offsets, edge_scale, node_scale)
        grad_src, grad_edge_scale, grad_node_scale = torch.autograd.grad(
            out, (src, edge_scale, node_scale), upstream
        )

        prefix = torch.zeros(3, 2)
        for node in range(3):
            for edge in range(offsets[node], offsets[node + 1]):
                prefix[node] += edge_scale.detach()[edge] * src.detach()[edge]
        expected_src = torch.zeros_like(src)
        for edge in range(6):
            expected_src[edge] = upstream[dst[edge]] * node_scale.detach()[dst[edge]] * edge_scale.detach()[edge]
        expected_node_scale = (upstream * prefix).sum(dim=1)
        expected_edge_scale = (
            upstream[dst] * node_scale.detach()[dst].unsqueeze(1) * src.detach()
        ).sum(dim=1)
        torch.testing.assert_close(grad_src, expected_src)
        torch.testing.assert_close(grad_node_scale, expected_node_scale)
        torch.testing.assert_close(grad_edge_scale, expected_edge_scale)


class SegmentSoftmaxReferenceTests(unittest.TestCase):
    @staticmethod
    def _inputs(seed=0, null=True):
        generator = torch.Generator().manual_seed(seed)
        dst, offsets = _graph([2, 1, 3])
        logit = torch.randn(6, 3, generator=generator)
        env = torch.tensor([0.9, -0.2, 0.7, 0.0, 1.1, 0.3])
        null_logit = torch.randn(3, 3, generator=generator) if null else None
        return dst, offsets, logit, env, null_logit

    def test_matches_manual_per_segment_softmax(self):
        dst, offsets, logit, env, null_logit = self._inputs()
        alpha, null_alpha = segment_softmax_reference(logit, env, dst, offsets, null_logit)
        alpha = alpha.detach()
        null_alpha = null_alpha.detach()

        for node in range(3):
            edges = list(range(offsets[node], offsets[node + 1]))
            for channel in range(3):
                logits = []
                for edge in edges:
                    if env[edge] > 0:
                        logits.append(logit[edge, channel].item() + 2 * torch.log(env[edge]).item())
                null_value = null_logit[node, channel].item()
                peak = max(logits + [null_value])
                edges_weight = [torch.exp(torch.tensor(v - peak)).item() for v in logits]
                null_weight = torch.exp(torch.tensor(null_value - peak)).item()
                total = null_weight + sum(edges_weight)
                attributed = 0.0
                for edge, weight in zip([e for e in edges if env[e] > 0], edges_weight):
                    torch.testing.assert_close(
                        alpha[edge, channel], torch.tensor(weight / total), rtol=1e-5, atol=1e-6
                    )
                    attributed += weight / total
                torch.testing.assert_close(
                    null_alpha[node, channel], torch.tensor(null_weight / total), rtol=1e-5, atol=1e-6
                )
                torch.testing.assert_close(
                    attributed + null_alpha[node, channel], torch.tensor(1.0), rtol=1e-5, atol=1e-6
                )

    def test_non_positive_envelope_carries_no_mass(self):
        dst, offsets, logit, env, null_logit = self._inputs()
        alpha, _ = segment_softmax_reference(logit, env, dst, offsets, null_logit)
        self.assertEqual(alpha.detach()[1].count_nonzero().item(), 0)
        self.assertEqual(alpha.detach()[3].count_nonzero().item(), 0)

    def test_without_null_mass_still_normalizes(self):
        dst, offsets, logit, env, _ = self._inputs(null=False)
        alpha, null_alpha = segment_softmax_reference(logit, env, dst, offsets)
        self.assertIsNone(null_alpha)
        per_segment = torch.zeros(3, 3).index_add_(0, dst, alpha.detach())
        torch.testing.assert_close(per_segment, torch.ones(3, 3), rtol=1e-5, atol=1e-6)

    def test_closed_form_backward_matches_autograd(self):
        dst, offsets, logit, env, null_logit = self._inputs(seed=3)
        logit = logit.clone().requires_grad_(True)
        env = env.abs().requires_grad_(True)
        null_logit = null_logit.clone().requires_grad_(True)
        alpha, null_alpha = segment_softmax_reference(logit, env, dst, offsets, null_logit)
        grad_alpha = torch.randn_like(alpha.detach())
        grad_null = torch.randn_like(null_alpha.detach())
        grad_logit, grad_env, grad_null_logit = torch.autograd.grad(
            (alpha, null_alpha),
            (logit, env, null_logit),
            (grad_alpha, grad_null),
            allow_unused=False,
        )

        alpha_value, null_value = alpha.detach(), null_alpha.detach()
        expectation = torch.zeros(3, 3).index_add_(0, dst, alpha_value * grad_alpha)
        expectation += null_value * grad_null
        expected_logit = alpha_value * (grad_alpha - expectation[dst])
        expected_null_logit = null_value * (grad_null - expectation)
        expected_env = (expected_logit.sum(dim=1)) * torch.where(
            env.detach() > 0, 2.0 / env.detach().clamp_min(1e-30), torch.zeros_like(env.detach())
        )
        torch.testing.assert_close(grad_logit, expected_logit, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grad_null_logit, expected_null_logit, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grad_env, expected_env, rtol=1e-4, atol=1e-6)


class ForceAssemblyReferenceTests(unittest.TestCase):
    @staticmethod
    def _inputs(seed=0):
        generator = torch.Generator().manual_seed(seed)
        dst, dst_offsets = _graph([2, 1, 3])
        src = torch.tensor([2, 0, 2, 1, 0, 2], dtype=torch.int64)
        order, src_offsets = group_by_source(src, 3)
        grad_edge = torch.randn(6, 3, generator=generator)
        edge_vec = torch.randn(6, 3, generator=generator)
        return grad_edge, edge_vec, src, dst, dst_offsets, src_offsets, order

    def test_matches_manual_loop(self):
        grad_edge, edge_vec, src, dst, dst_offsets, src_offsets, _ = self._inputs()
        force, virial = force_assembly_reference(
            grad_edge, edge_vec, src, dst, dst_offsets, src_offsets
        )

        expected_force = torch.zeros(3, 3)
        expected_virial = torch.zeros(3, 9)
        for edge in range(6):
            expected_force[dst[edge]] += grad_edge[edge]
            expected_force[src[edge]] -= grad_edge[edge]
            expected_virial[src[edge]] -= torch.outer(grad_edge[edge], edge_vec[edge]).reshape(9)
        torch.testing.assert_close(force, expected_force)
        torch.testing.assert_close(virial, expected_virial)

    def test_net_force_vanishes_for_an_isolated_graph(self):
        # Every edge contributes +g to one endpoint and -g to the other.
        grad_edge, edge_vec, src, dst, dst_offsets, src_offsets, _ = self._inputs(seed=1)
        force, _ = force_assembly_reference(grad_edge, edge_vec, src, dst, dst_offsets, src_offsets)
        torch.testing.assert_close(force.sum(dim=0), torch.zeros(3), rtol=1e-6, atol=1e-6)

    def test_rejects_a_source_grouping_that_does_not_match(self):
        grad_edge, edge_vec, src, dst, dst_offsets, _, order = self._inputs()
        with self.assertRaises(ValueError):
            force_assembly_reference(
                grad_edge, edge_vec, src, dst, dst_offsets, torch.tensor([0, 1, 2, 6])
            )


if __name__ == "__main__":
    unittest.main()
