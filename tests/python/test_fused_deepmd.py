"""Host tests for the DeepMD fused-dispatch adapter.

No DCU and no DeepMD are needed: the two seams are exercised against stand-in
modules and a stand-in operator bundle that delegates to the torch reference
implementations, so the patched path can be compared with the stock one
numerically. What the device itself has to prove -- that the HIP kernels run and
are faster -- lives in ``tests/csrc/accuracy/test_fused_deepmd_parity.py``.
"""
from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from model import fused_deepmd
from model.dpa4 import _normalize_fused
from reference.segment import (
    force_assembly_reference,
    segment_softmax_reference,
    segment_sum_reference,
)


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
        """Mirror kernels.segment.segment_softmax: a 2-tuple when a null logit is given."""
        self.calls.append(("softmax", tuple(logit.shape)))
        dst = _dst_from_csr(offsets, order)
        alpha, null_alpha = segment_softmax_reference(logit, env, dst, offsets, null_logit)
        return alpha if null_logit is None else (alpha, null_alpha)

    def segment_sum(self, src, dst, offsets, edge_scale=None, node_scale=None, order=None):
        self.calls.append(("sum", tuple(src.shape)))
        natural = _dst_from_csr(offsets, order)
        return segment_sum_reference(src, natural, offsets, edge_scale, node_scale)

    def force_assembly(self, grad_edge, edge_vec, src=None, dst=None,
                       dst_offsets=None, src_offsets=None, src_order=None, dst_order=None):
        """Mirror kernels.segment.force_assembly, including the optional src/dst."""
        self.calls.append(("force", tuple(grad_edge.shape)))
        dst_natural = _dst_from_csr(dst_offsets, dst_order)
        src_natural = _dst_from_csr(src_offsets, src_order)
        return force_assembly_reference(
            grad_edge, edge_vec, src_natural, dst_natural, dst_offsets, src_offsets
        )


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


class FakeAttentionBlock:
    """Minimal SO(2) block: the attributes and the four methods the seam uses."""

    ebed_dim_full = 2
    attn_n_focus = 2
    attn_focus_dim = 4
    n_atten_head = 2
    head_dim = 2
    hidden_channels = 8
    compute_dtype = torch.float32
    dtype = torch.float32
    attn_v_proj = None
    attn_o_proj = None

    def __init__(self, message, alpha):
        self.training = False
        self._segment_softmax_fn = None
        self._message = message
        self._alpha = alpha

    def edge_message(self, x, edge_cache, radial_feat):
        return self._message, None

    def attention_weights(self, x_l0_node, edge_cache, radial_feat):
        return self._alpha

    def attention_head_gate(self, x_l0_node):
        return torch.rand(
            x_l0_node.shape[0], self.attn_n_focus, self.n_atten_head,
            generator=torch.Generator().manual_seed(7),
        ) + 0.5

    def forward_envelope(self, x, edge_cache, radial_feat):
        return x

    def forward_attention_dense(self, x, edge_cache, radial_feat, x_l0_node):
        n_edge = self._message.shape[0]
        dst = edge_cache.dst
        value_heads = self._message.reshape(
            n_edge, self.ebed_dim_full, self.attn_n_focus, self.n_atten_head, self.head_dim
        )
        weighted = value_heads * self._alpha.reshape(
            n_edge, 1, self.attn_n_focus, self.n_atten_head, 1
        )
        out = torch.zeros(
            x.shape[0], self.ebed_dim_full, self.attn_n_focus,
            self.n_atten_head, self.head_dim, dtype=weighted.dtype,
        )
        out.index_add_(0, dst, weighted)
        gate = self.attention_head_gate(x_l0_node)
        out = out * gate.reshape(x.shape[0], 1, self.attn_n_focus, self.n_atten_head, 1)
        return out.reshape(x.shape[0], self.ebed_dim_full, self.hidden_channels)


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
        stats = fused_deepmd.deepmd_patch(
            model, ops=ops, namespaces=[namespace], envelope=False, verbose=False
        )
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
        fused_deepmd.deepmd_patch(
            FakeModel([]), ops=ops, namespaces=[namespace], envelope=False, verbose=False
        )
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
        stats = fused_deepmd.deepmd_patch(
            FakeModel([block]), ops=ops, softmax=False, verbose=False
        )
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

    def test_envelope_fallback_calls_the_stock_method(self):
        # fp64 inputs must go through the stock aggregation unchanged; the
        # captured callable is a bound method, so the fallback must not pass the
        # block again.
        cache = FakeEdgeCache([2, 1, 3], channels=4)
        message = torch.randn(int(cache.dst.numel()), 2, 4, dtype=torch.float64)
        block = FakeBlock(message)
        stats = fused_deepmd.deepmd_patch(
            FakeModel([block]), ops=FakeOps(), softmax=False, verbose=False
        )
        x = torch.randn(3, 2, 4, dtype=torch.float64)
        actual = block.forward_envelope(x, cache, None)
        fused_deepmd.uninstall()
        expected = block.forward_envelope(x, cache, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(stats.envelope_hits, 0)
        self.assertEqual(stats.envelope_fallbacks, 1)

    def test_install_restores_every_rebound_name(self):
        namespace = _namespace("fake.attention")
        original = namespace.segment_envelope_gated_softmax
        cache = FakeEdgeCache([1, 2])
        block = FakeBlock(torch.randn(int(cache.dst.numel()), 2, 4))
        installed = block.forward_envelope
        fused_deepmd.deepmd_patch(
            FakeModel([block]), ops=FakeOps(), namespaces=[namespace], verbose=False
        )
        self.assertIsNot(namespace.segment_envelope_gated_softmax, original)
        self.assertIn("forward_envelope", block.__dict__)

        fused_deepmd.uninstall()
        self.assertIs(namespace.segment_envelope_gated_softmax, original)
        self.assertNotIn("forward_envelope", block.__dict__)
        self.assertIs(block.forward_envelope.__func__, FakeBlock.forward_envelope)
        self.assertIs(block.forward_envelope.__self__, block)

    def test_install_is_a_noop_without_the_extension(self):
        # `ops=None` means "resolve the extension"; on a DCU host with the
        # extension built that resolution succeeds and the seams do bind, so the
        # branch under test has to be forced rather than inferred from the host.
        model = FakeModel([FakeBlock(torch.zeros(2, 1, 1))])
        with mock.patch.object(fused_deepmd, "_load_ops", lambda: None):
            stats = fused_deepmd.deepmd_patch(model, ops=None, verbose=False)
        self.assertFalse(stats.bound_softmax or stats.bound_envelope)
        self.assertEqual(fused_deepmd.status(stats)["blocks"], 1)

    def test_installing_twice_keeps_one_patch_and_resets_counters(self):
        namespace = _namespace("fake.attention")
        original = namespace.segment_envelope_gated_softmax
        model = FakeModel([])
        first = fused_deepmd.deepmd_patch(
            model, ops=FakeOps(), namespaces=[namespace], envelope=False, verbose=False
        )
        second = fused_deepmd.deepmd_patch(
            model, ops=FakeOps(), namespaces=[namespace], envelope=False, verbose=False
        )
        patched = [key for key in fused_deepmd._SAVED if key.endswith(".segment_envelope_gated_softmax")]
        self.assertEqual(len(patched), 1)
        self.assertEqual(first.softmax_hits, 0)
        self.assertEqual(second.softmax_hits, 0)
        fused_deepmd.uninstall()
        self.assertIs(namespace.segment_envelope_gated_softmax, original)
        self.assertEqual(fused_deepmd.status(None), {})


class FakeSlotOps:
    """Mirror kernels.slot, computed by the torch reference."""

    def __init__(self):
        self.calls: list[tuple] = []

    def slot_pack_scale(self, src, slot, scale=None):
        from reference.slot import slot_pack_scale_reference

        self.calls.append(tuple(src.shape))
        return slot_pack_scale_reference(src, slot, scale)


class FakeGatedActivation:
    """``GatedActivation`` in the ``fndc`` layout, with the stock forward body."""

    layout = "fndc"
    lmax = 2
    channels = 2
    n_focus = 1
    dtype = torch.float32
    training = False

    def __init__(self, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.expand_index = torch.tensor([0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=torch.int64)
        self._linear = torch.nn.Linear(self.channels, self.lmax * self.channels, bias=False)
        with torch.no_grad():
            self._linear.weight.copy_(torch.randn_like(self._linear.weight))
        self.scalar_act = lambda tensor: torch.nn.functional.silu(tensor)

    def gate_linear(self, tensor):
        return self._linear(tensor)

    def forward(self, x, gate=None):
        degree_axis = 2
        source = gate if gate is not None else x
        gate_scalar_source = source.select(dim=degree_axis, index=0)
        if gate is not None:
            x0 = x.narrow(degree_axis, 0, 1) * self.scalar_act(gate.narrow(degree_axis, 0, 1))
        else:
            x0 = self.scalar_act(x.narrow(degree_axis, 0, 1))
        gate_src = gate_scalar_source.to(dtype=self.dtype)
        gate_logits = self.gate_linear(gate_src.transpose(0, 1)).transpose(0, 1)
        gating = torch.sigmoid(gate_logits).to(dtype=gate_scalar_source.dtype)
        gating = gating.reshape(
            x.shape[0], gate_scalar_source.shape[1], self.lmax, self.channels
        )
        gates = gating.index_select(dim=2, index=self.expand_index)
        out = x.new_empty(x.shape)
        out.narrow(degree_axis, 0, 1).copy_(x0)
        out.narrow(degree_axis, 1, x.shape[degree_axis] - 1).copy_(
            x.narrow(degree_axis, 1, x.shape[degree_axis] - 1) * gates
        )
        return out


class GatedActivationSeamTests(unittest.TestCase):
    """The gate expansion plus the value multiply is one slot_pack_scale."""

    def setUp(self):
        fused_deepmd._ACCEPT_ANY_DEVICE = True
        self.addCleanup(fused_deepmd.uninstall)
        self.addCleanup(setattr, fused_deepmd, "_ACCEPT_ANY_DEVICE", False)

    @staticmethod
    def _model(activation):
        return SimpleNamespace(modules=lambda: [activation])

    def test_seam_matches_the_stock_forward(self):
        activation = FakeGatedActivation()
        slot_ops = FakeSlotOps()
        stats = fused_deepmd.deepmd_patch(
            self._model(activation), ops=FakeOps(), softmax=False, envelope=False,
            force=False, aggregate=False, pack=True, slot_ops=slot_ops, verbose=False,
        )
        self.assertTrue(stats.bound_pack)
        x = torch.randn(1, 5, 10, 2)
        actual = activation.forward(x)
        fused_deepmd.uninstall()
        expected = activation.forward(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(stats.pack_hits, 1)
        self.assertEqual(stats.pack_fallbacks, 0)
        self.assertEqual(len(slot_ops.calls), 1)

    def test_modules_that_merely_hold_an_expand_index_are_left_alone(self):
        """SO3Linear and the norm layers register expand_index too.

        Rebinding their one-argument forward crashed inference: the shim fell
        back with original(x, None), which a forward(self, x) cannot take.
        """

        class SO3Like:
            def __init__(self):
                self.expand_index = torch.tensor([0, 1], dtype=torch.int64)
                self.lmax = 1

            def forward(self, x):
                return x * 2.0

        module = SO3Like()
        stats = fused_deepmd.deepmd_patch(
            self._model(module), ops=FakeOps(), softmax=False, envelope=False,
            force=False, aggregate=False, pack=True, slot_ops=FakeSlotOps(), verbose=False,
        )
        self.assertFalse(stats.bound_pack)
        self.assertNotIn("forward", module.__dict__)
        x = torch.randn(3, 4, 5)
        torch.testing.assert_close(module.forward(x), x * 2.0)

    def test_seam_falls_back_when_the_layout_is_not_fndc(self):
        activation = FakeGatedActivation()
        activation.layout = "ndfc"
        stats = fused_deepmd.deepmd_patch(
            self._model(activation), ops=FakeOps(), softmax=False, envelope=False,
            force=False, aggregate=False, pack=True, slot_ops=FakeSlotOps(), verbose=False,
        )
        x = torch.randn(1, 5, 10, 2)
        result = activation.forward(x)
        self.assertEqual(tuple(result.shape), (1, 5, 10, 2))
        self.assertEqual(stats.pack_hits, 0)
        self.assertEqual(stats.pack_fallbacks, 1)

    def test_uninstall_restores_the_stock_forward(self):
        activation = FakeGatedActivation()
        fused_deepmd.deepmd_patch(
            self._model(activation), ops=FakeOps(), softmax=False, envelope=False,
            force=False, aggregate=False, pack=True, slot_ops=FakeSlotOps(), verbose=False,
        )
        self.assertIn("forward", activation.__dict__)
        fused_deepmd.uninstall()
        self.assertNotIn("forward", activation.__dict__)
        self.assertIs(activation.forward.__func__, FakeGatedActivation.forward)


class AggregationSeamTests(unittest.TestCase):
    """The dense attention reduction is one segment_sum over the edge weights."""

    def setUp(self):
        fused_deepmd._ACCEPT_ANY_DEVICE = True
        self.addCleanup(fused_deepmd.uninstall)
        self.addCleanup(setattr, fused_deepmd, "_ACCEPT_ANY_DEVICE", False)

    @staticmethod
    def _inputs(counts=(2, 1, 3), seed=5):
        generator = torch.Generator().manual_seed(seed)
        dst, offsets = _grouped_dst(counts)
        cache = SimpleNamespace(dst=dst, offsets=offsets)
        n_edge, atoms = int(dst.numel()), len(counts)
        block = FakeAttentionBlock(
            message=torch.randn(n_edge, 2, 8, generator=generator),
            alpha=torch.rand(n_edge, 2, 2, generator=generator) + 0.1,
        )
        return block, cache, atoms

    def _patch(self, block, ops=None):
        return fused_deepmd.deepmd_patch(
            FakeModel([block]),
            ops=ops or FakeOps(),
            softmax=False,
            envelope=False,
            force=False,
            verbose=False,
        )

    def test_seam_matches_the_stock_method(self):
        block, cache, atoms = self._inputs()
        stats = self._patch(block)
        self.assertTrue(stats.bound_aggregate)
        x = torch.randn(atoms, 2, 8)
        x_l0 = torch.randn(atoms, 2, 4)
        torch.manual_seed(1)
        actual = block.forward_attention_dense(x, cache, None, x_l0)
        fused_deepmd.uninstall()
        torch.manual_seed(1)
        expected = block.forward_attention_dense(x, cache, None, x_l0)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(stats.aggregate_hits, 1)
        self.assertEqual(stats.aggregate_fallbacks, 0)

    def test_seam_accepts_a_non_contiguous_x(self):
        block, cache, atoms = self._inputs()
        stats = self._patch(block)
        wide = torch.randn(atoms, 2, 16)
        x = wide[..., ::2]  # float32 CUDA-eligible, strides not contiguous
        x_l0 = torch.randn(atoms, 2, 4)
        self.assertFalse(x.is_contiguous())
        torch.manual_seed(1)
        result = block.forward_attention_dense(x, cache, None, x_l0)
        self.assertEqual(tuple(result.shape), (atoms, 2, 8))
        self.assertEqual(stats.aggregate_hits, 1)
        self.assertEqual(stats.aggregate_fallbacks, 0)

    def test_seam_falls_back_for_a_non_fp32_message(self):
        block, cache, atoms = self._inputs()
        block._message = block._message.double()
        stats = self._patch(block)
        x = torch.randn(atoms, 2, 8).double()
        x_l0 = torch.randn(atoms, 2, 4).double()
        result = block.forward_attention_dense(x, cache, None, x_l0)
        self.assertEqual(tuple(result.shape), (atoms, 2, 8))
        self.assertEqual(stats.aggregate_hits, 0)
        self.assertEqual(stats.aggregate_fallbacks, 1)

    def test_uninstall_restores_the_stock_method(self):
        block, cache, atoms = self._inputs()
        self._patch(block)
        self.assertIn("forward_attention_dense", block.__dict__)
        fused_deepmd.uninstall()
        self.assertNotIn("forward_attention_dense", block.__dict__)
        self.assertIs(
            block.forward_attention_dense.__func__, FakeAttentionBlock.forward_attention_dense
        )


class ForceSeamTests(unittest.TestCase):
    """The force/virial seam: DeepMD's two CSR permutations in, force+virial out."""

    def setUp(self):
        fused_deepmd._ACCEPT_ANY_DEVICE = True
        self.addCleanup(fused_deepmd.uninstall)
        self.addCleanup(setattr, fused_deepmd, "_ACCEPT_ANY_DEVICE", False)

    @staticmethod
    def _graph(atoms=4, edges=9, seed=0):
        generator = torch.Generator().manual_seed(seed)
        dst = torch.randint(0, atoms, (edges,), generator=generator)
        src = torch.randint(0, atoms, (edges,), generator=generator)
        dst_order = torch.argsort(dst, stable=True)
        src_order = torch.argsort(src, stable=True)
        return dst, src, dst_order, src_order

    @staticmethod
    def _force_modules():
        transform = types.ModuleType("fake.transform_output")
        transform.triton_infer_level = lambda: 0
        transform.use_cutile_infer = lambda: False
        assembly = types.ModuleType("fake.edge_force_virial")

        def original(*args, **kwargs):
            raise AssertionError("DeepMD's own force assembly must not run")

        assembly.edge_force_assembly = original
        return transform, assembly

    def _install(self, ops=None):
        transform, assembly = self._force_modules()
        stats = fused_deepmd.deepmd_patch(
            FakeModel([]),
            ops=ops or FakeOps(),
            softmax=False,
            envelope=False,
            force_targets=(transform, assembly),
            verbose=False,
        )
        return transform, assembly, stats

    def test_patched_call_matches_the_dense_reference(self):
        transform, assembly, stats = self._install()
        self.assertTrue(stats.bound_force)
        self.assertEqual(transform.triton_infer_level(), 1)

        atoms, edges = 4, 9
        dst, src, dst_order, src_order = self._graph(atoms, edges)
        dst_offsets = torch.cat([torch.zeros(1, dtype=torch.int64),
                                 torch.bincount(dst, minlength=atoms).cumsum(0)])
        src_offsets = torch.cat([torch.zeros(1, dtype=torch.int64),
                                 torch.bincount(src, minlength=atoms).cumsum(0)])
        g = torch.randn(edges, 3)
        edge_vec = torch.randn(edges, 3)
        force, virial = assembly.edge_force_assembly(
            g, edge_vec, dst_order, dst_offsets, src_order, src_offsets
        )
        expected_force, expected_virial = force_assembly_reference(
            g, edge_vec, src, dst, dst_offsets, src_offsets
        )
        torch.testing.assert_close(force, expected_force, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(virial, expected_virial, rtol=1e-5, atol=1e-6)
        self.assertEqual(stats.force_hits, 1)
        self.assertEqual(stats.force_fallbacks, 0)

    def test_unsupported_dtype_falls_back_to_the_dense_math(self):
        _, assembly, stats = self._install()
        atoms, edges = 4, 9
        dst, src, dst_order, src_order = self._graph(atoms, edges, seed=3)
        dst_offsets = torch.cat([torch.zeros(1, dtype=torch.int64),
                                 torch.bincount(dst, minlength=atoms).cumsum(0)])
        src_offsets = torch.cat([torch.zeros(1, dtype=torch.int64),
                                 torch.bincount(src, minlength=atoms).cumsum(0)])
        g = torch.randn(edges, 3).double()          # fp64: the kernel cannot take it
        edge_vec = torch.randn(edges, 3).double()
        force, virial = assembly.edge_force_assembly(
            g, edge_vec, dst_order, dst_offsets, src_order, src_offsets
        )
        expected_force = torch.zeros(atoms, 3, dtype=g.dtype)
        expected_force.index_add_(0, dst, g)
        expected_force.index_add_(0, src, -g)
        expected_virial = torch.zeros(atoms, 9, dtype=g.dtype)
        expected_virial.index_add_(
            0, src, -torch.einsum("ek,ej->ekj", g, edge_vec).reshape(-1, 9)
        )
        torch.testing.assert_close(force, expected_force, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(virial, expected_virial, rtol=1e-10, atol=1e-12)
        self.assertEqual(stats.force_hits, 0)
        self.assertEqual(stats.force_fallbacks, 1)

    def test_uninstall_restores_the_seam(self):
        transform, assembly = self._force_modules()
        original = assembly.edge_force_assembly
        fused_deepmd.deepmd_patch(
            FakeModel([]), ops=FakeOps(), softmax=False, envelope=False,
            force_targets=(transform, assembly), verbose=False,
        )
        self.assertIsNot(assembly.edge_force_assembly, original)
        self.assertEqual(transform.triton_infer_level(), 1)
        fused_deepmd.uninstall()
        self.assertIs(assembly.edge_force_assembly, original)
        self.assertEqual(transform.triton_infer_level(), 0)


class SoftmaxSeamReturnShapeTests(unittest.TestCase):
    """The seam must cope with the wrapper's real return arity."""

    def setUp(self):
        fused_deepmd._ACCEPT_ANY_DEVICE = True
        self.addCleanup(fused_deepmd.uninstall)
        self.addCleanup(setattr, fused_deepmd, "_ACCEPT_ANY_DEVICE", False)

    def test_two_value_return_is_unpacked(self):
        namespace = _namespace("fake.attention")
        ops = FakeOps()  # returns (alpha, null_alpha) when a null logit is passed
        fused_deepmd.deepmd_patch(
            FakeModel([]), ops=ops, namespaces=[namespace], envelope=False, verbose=False
        )
        dst, _ = _grouped_dst([2, 0, 3])
        logits = torch.randn(5, 2, 3)
        env = torch.rand(5, 1) + 0.1
        bias = torch.randn(2, 3)
        actual = namespace.segment_envelope_gated_softmax(logits, env, dst, 3, bias, 1e-6)
        self.assertEqual(tuple(actual.shape), (5, 2, 3))
        fused_deepmd.uninstall()
        expected = namespace.segment_envelope_gated_softmax(logits, env, dst, 3, bias, 1e-6)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_plain_tensor_return_is_also_accepted(self):
        class TensorOps(FakeOps):
            def segment_softmax(self, logit, env, offsets, null_logit=None, order=None):
                alpha, _ = FakeOps.segment_softmax(self, logit, env, offsets, null_logit, order)
                return alpha

        namespace = _namespace("fake.attention")
        fused_deepmd.deepmd_patch(
            FakeModel([]), ops=TensorOps(), namespaces=[namespace], envelope=False, verbose=False
        )
        dst, _ = _grouped_dst([2, 0, 3])
        alpha = namespace.segment_envelope_gated_softmax(
            torch.randn(5, 2, 3), torch.rand(5, 1) + 0.1, dst, 3, torch.randn(2, 3), 1e-6
        )
        self.assertEqual(tuple(alpha.shape), (5, 2, 3))


class WrapperResolutionTests(unittest.TestCase):
    """A DeepPot is not an nn.Module: the tree sits under deep_eval."""

    def setUp(self):
        fused_deepmd._ACCEPT_ANY_DEVICE = True
        self.addCleanup(fused_deepmd.uninstall)
        self.addCleanup(setattr, fused_deepmd, "_ACCEPT_ANY_DEVICE", False)

    def test_legacy_pt_tree_lives_under_deep_eval_dp(self):
        root = FakeModel([FakeBlock(torch.zeros(2, 1, 1))])
        wrapper = SimpleNamespace(deep_eval=SimpleNamespace(dp=root))
        resolved, path = fused_deepmd._module_root(wrapper)
        self.assertIs(resolved, root)
        self.assertEqual(path, "deep_eval.dp")

    def test_pt_expt_tree_lives_under_deep_eval_dpmodel(self):
        root = FakeModel([])
        wrapper = SimpleNamespace(deep_eval=SimpleNamespace(_dpmodel=root))
        self.assertEqual(fused_deepmd._module_root(wrapper), (root, "deep_eval._dpmodel"))

    def test_wrapper_without_a_tree_resolves_to_nothing(self):
        self.assertEqual(
            fused_deepmd._module_root(SimpleNamespace(deep_eval=SimpleNamespace())),
            (None, ""),
        )
        lines = fused_deepmd.describe_module(SimpleNamespace(deep_eval=SimpleNamespace()))
        self.assertIn("no nn.Module tree", "\n".join(lines))

    def test_patch_finds_blocks_through_a_deeppot_wrapper(self):
        cache = FakeEdgeCache([2, 1, 3], channels=4)
        message = torch.randn(int(cache.dst.numel()), 2, 4)
        block = FakeBlock(message)
        wrapper = SimpleNamespace(deep_eval=SimpleNamespace(dp=FakeModel([block])))
        stats = fused_deepmd.deepmd_patch(
            wrapper, ops=FakeOps(), softmax=False, verbose=False
        )
        self.assertTrue(stats.bound_envelope)
        self.assertEqual(stats.blocks, 1)
        x = torch.randn(3, 2, 4)
        actual = block.forward_envelope(x, cache, None)
        fused_deepmd.uninstall()
        expected = block.forward_envelope(x, cache, None)
        budget = 1e-5 * float(expected.abs().max())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=budget)
        self.assertEqual(stats.envelope_hits, 1)


class ModuleDumpTests(unittest.TestCase):
    def test_describe_module_lists_the_tree_and_the_targets(self):
        cache = FakeEdgeCache([2, 1])
        block = FakeBlock(torch.randn(int(cache.dst.numel()), 2, 4))
        lines = fused_deepmd.describe_module(FakeModel([block]))
        text = "\n".join(lines)
        self.assertIn("module: FakeModel", text)
        self.assertIn("patch targets: 1 SO(2) block(s)", text)
        self.assertIn("forward_envelope -> segment_sum", text)

    def test_verbose_patch_prints_the_dump(self):
        namespace = _namespace("fake.attention")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            fused_deepmd.deepmd_patch(
                FakeModel([]), ops=FakeOps(), namespaces=[namespace], envelope=False
            )
        output = buffer.getvalue()
        self.assertIn("[deepmd_patch] module:", output)
        self.assertIn("segment_envelope_gated_softmax", output)
        fused_deepmd.uninstall()


class FusedSwitchTests(unittest.TestCase):
    def test_true_enables_every_seam(self):
        self.assertEqual(
            _normalize_fused(True),
            {
                "softmax": True,
                "envelope": True,
                "force": True,
                "aggregate": True,
                "pack": True,
            },
        )

    def test_names_and_sequences_select_seams(self):
        self.assertEqual(_normalize_fused("softmax"), {"softmax": True})
        self.assertEqual(_normalize_fused(("envelope",)), {"envelope": True})
        self.assertEqual(_normalize_fused(("force",)), {"force": True})
        self.assertEqual(
            _normalize_fused({"softmax": True, "envelope": False}),
            {
                "softmax": True,
                "envelope": False,
                "force": False,
                "aggregate": False,
                "pack": False,
            },
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
