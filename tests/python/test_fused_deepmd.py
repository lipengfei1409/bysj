"""Host tests for the DeepMD fused-dispatch adapter.

No DCU and no DeepMD are needed: the two seams are exercised against stand-in
modules and a stand-in operator bundle that delegates to the torch reference
implementations, so the patched path can be compared with the stock one
numerically. What the device itself has to prove -- that the HIP kernels run and
are faster -- lives in ``tests/csrc/accuracy/test_fused_deepmd_parity.py``.
"""
from __future__ import annotations

import types
import unittest

import torch

from model import fused_deepmd
from model.dpa4 import _normalize_fused
from reference.segment import segment_softmax_reference, segment_sum_reference


def _grouped_dst(counts):
    dst = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.int64), torch.tensor(counts, dtype=torch.int64)
    )
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.tensor(counts).cumsum(0)])
    return dst, offsets


def _dst_from_csr(offsets, order):
    """Recover the natural-order segment ids from a CSR view."""
    counts = (offsets[1:] - offsets[:-1]).tolist()
    grouped = torch.repeat_interleave(
        torch.arange(offsets.numel() - 1, dtype=torch.int64), torch.tensor(counts, dtype=torch.int64)
    )
    dst = torch.empty_like(grouped)
    dst[order] = grouped
    return dst


class FakeOps:
    """Operator bundle that computes with the torch references and records calls."""

    def __init__(self):
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def segment_softmax(self, logit, env, offsets, null_logit=None, order=None):
        self.calls.append(("softmax", tuple(logit.shape)))
        dst = _dst_from_csr(offsets, order)
        alpha, _ = segment_softmax_reference(logit, env, dst, offsets, null_logit)
        return alpha

    def segment_sum(self, src, dst, offsets, edge_scale=None, node_scale=None, order=None):
        self.calls.append(("sum", tuple(src.shape)))
        natural = _dst_from_csr(offsets, order)
        return segment_sum_reference(src, natural, offsets, edge_scale, node_scale)


class FakeEdgeCache:
    def __init__(self, counts, channels=4):
        self.dst, self.offsets = _grouped_dst(counts)
        n_edges = int(self.dst.numel())
        n_nodes = len(counts)
        self.edge_env = torch.rand(n_edges, 1) + 0.1
        self.inv_sqrt_deg = (torch.rand(n_nodes, 1, 1) + 0.5)
        self.edge_src_gate = None
        self.channels = channels


class FakeBlock:
    def __init__(self, message):
        self.training = False
        self._segment_softmax_fn = None
        self._message = message

    def edge_message(self, x, edge_cache, radial_feat):
        return self._message, None

    def forward_envelope(self, x, edge_cache, radial_feat):
        # The stock aggregation: zero-init, atomic scatter, degree scale.
        message, _ = self.edge_message(x, edge_cache, radial_feat)
        weight = edge_cache.edge_env if edge_cache.edge_src_gate is None else (
            edge_cache.edge_env * edge_cache.edge_src_gate
        )
        out = x.new_zeros((x.shape[0], x.shape[1], x.shape[2]))
        out.index_add_(0, edge_cache.dst, message * weight.unsqueeze(-1))
        out.mul_(edge_cache.inv_sqrt_deg)
        return out


class FakeModel:
    def __init__(self, blocks):
        self._blocks = blocks

    def modules(self):
        return list(self._blocks)


def _namespace(name):
    module = types.ModuleType(name)

    def reference(logits, edge_env, dst, n_nodes, z_bias_raw, eps, src_weight=None):
        n_edge, n_focus, n_head = logits.shape
        channel = n_focus * n_head
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64), torch.bincount(dst, minlength=n_nodes).cumsum(0)]
        )
        order = torch.argsort(dst, stable=True)
        null = torch.log(torch.nn.functional.softplus(z_bias_raw.reshape(-1)) + eps)
        alpha, _ = segment_softmax_reference(
            logits.reshape(n_edge, channel),
            edge_env.reshape(n_edge),
            dst,
            offsets,
            null.reshape(1, channel).expand(n_nodes, channel),
        )
        return alpha.reshape(n_edge, n_focus, n_head)

    module.segment_envelope_gated_softmax = reference
    return module


class FusedDispatchTests(unittest.TestCase):
    def setUp(self):
        fused_deepmd._ACCEPT_ANY_DEVICE = True
        self.addCleanup(fused_deepmd.uninstall)
        self.addCleanup(setattr, fused_deepmd, "_ACCEPT_ANY_DEVICE", False)

    def test_softmax_seam_matches_the_reference_path(self):
        namespace = _namespace("fake.attention")
        ops = FakeOps()
        model = FakeModel([])
        stats = fused_deepmd.install(model, ops=ops, namespaces=[namespace], envelope=False)
        self.assertTrue(stats.bound_softmax)

        dst, _ = _grouped_dst([2, 0, 3])
        logits = torch.randn(5, 2, 3)
        env = torch.rand(5, 1) + 0.1
        bias = torch.randn(2, 3)
        actual = namespace.segment_envelope_gated_softmax(logits, env, dst, 3, bias, 1e-6)

        fused_deepmd.uninstall()
        expected = namespace.segment_envelope_gated_softmax(logits, env, dst, 3, bias, 1e-6)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(stats.softmax_hits, 1)
        self.assertEqual(ops.calls[0][0], "softmax")

    def test_source_gated_softmax_falls_back(self):
        namespace = _namespace("fake.attention")
        ops = FakeOps()
        fused_deepmd.install(FakeModel([]), ops=ops, namespaces=[namespace], envelope=False)
        dst, _ = _grouped_dst([2, 0, 3])
        logits = torch.randn(5, 2, 3)
        env = torch.rand(5, 1) + 0.1
        gate = torch.rand(5, 1)
        stats = fused_deepmd._SESSION.stats
        namespace.segment_envelope_gated_softmax(logits, env, dst, 3, torch.randn(2, 3), 1e-6, gate)
        self.assertEqual(stats.softmax_fallbacks, 1)
        self.assertEqual(stats.softmax_hits, 0)
        self.assertEqual(ops.calls, [])

    def test_envelope_seam_matches_the_stock_aggregation(self):
        cache = FakeEdgeCache([2, 1, 3], channels=4)
        message = torch.randn(int(cache.dst.numel()), 2, 4)
        block = FakeBlock(message)
        ops = FakeOps()
        stats = fused_deepmd.install(FakeModel([block]), ops=ops, softmax=False)
        self.assertTrue(stats.bound_envelope)

        x = torch.randn(3, 2, 4)
        actual = block.forward_envelope(x, cache, None)

        fused_deepmd.uninstall()
        expected = block.forward_envelope(x, cache, None)
        # The fused path sums in CSR order, the stock path with atomics: fp32
        # summation order is the only difference, so compare against the scale.
        budget = 1e-5 * float(expected.abs().max())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=budget)
        self.assertEqual(stats.envelope_hits, 1)
        self.assertEqual(ops.calls[0][0], "sum")

    def test_install_restores_every_rebound_name(self):
        namespace = _namespace("fake.attention")
        original = namespace.segment_envelope_gated_softmax
        cache = FakeEdgeCache([1, 2])
        block = FakeBlock(torch.randn(int(cache.dst.numel()), 2, 4))
        installed = block.forward_envelope
        fused_deepmd.install(FakeModel([block]), ops=FakeOps(), namespaces=[namespace])
        self.assertIsNot(namespace.segment_envelope_gated_softmax, original)
        self.assertIn("forward_envelope", block.__dict__)

        fused_deepmd.uninstall()
        self.assertIs(namespace.segment_envelope_gated_softmax, original)
        self.assertNotIn("forward_envelope", block.__dict__)
        self.assertIs(block.forward_envelope.__func__, FakeBlock.forward_envelope)
        self.assertIs(block.forward_envelope.__self__, block)

    def test_install_is_a_noop_without_the_extension(self):
        model = FakeModel([FakeBlock(torch.zeros(2, 1, 1))])
        stats = fused_deepmd.install(model, ops=None)
        self.assertFalse(stats.bound_softmax or stats.bound_envelope)
        self.assertEqual(fused_deepmd.status(stats)["blocks"], 1)

    def test_installing_twice_keeps_one_patch_and_resets_counters(self):
        namespace = _namespace("fake.attention")
        original = namespace.segment_envelope_gated_softmax
        model = FakeModel([])
        first = fused_deepmd.install(model, ops=FakeOps(), namespaces=[namespace], envelope=False)
        second = fused_deepmd.install(model, ops=FakeOps(), namespaces=[namespace], envelope=False)
        patched = [key for key in fused_deepmd._SAVED if key.endswith(".segment_envelope_gated_softmax")]
        self.assertEqual(len(patched), 1)
        self.assertEqual(first.softmax_hits, 0)
        self.assertEqual(second.softmax_hits, 0)
        fused_deepmd.uninstall()
        self.assertIs(namespace.segment_envelope_gated_softmax, original)
        self.assertEqual(fused_deepmd.status(None), {})


class FusedSwitchTests(unittest.TestCase):
    def test_true_enables_every_seam(self):
        self.assertEqual(_normalize_fused(True), {"softmax": True, "envelope": True})

    def test_names_and_sequences_select_seams(self):
        self.assertEqual(_normalize_fused("softmax"), {"softmax": True})
        self.assertEqual(_normalize_fused(("envelope",)), {"envelope": True})
        self.assertEqual(
            _normalize_fused({"softmax": True, "envelope": False}),
            {"softmax": True, "envelope": False},
        )

    def test_unknown_seam_is_rejected(self):
        with self.assertRaises(ValueError):
            _normalize_fused("flash")
        with self.assertRaises(ValueError):
            _normalize_fused({"flash": True})
        with self.assertRaises(TypeError):
            _normalize_fused(3)

    def test_false_and_none_disable_the_patch(self):
        self.assertIsNone(_normalize_fused(False))
        self.assertIsNone(_normalize_fused(None))
        self.assertIsNone(_normalize_fused(()))


if __name__ == "__main__":
    unittest.main()
