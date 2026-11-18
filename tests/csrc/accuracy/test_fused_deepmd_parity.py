"""Opt-in DCU parity test for the DeepMD fused dispatch adapter.

Run on a DCU host with DeepMD, the built extension and a real checkpoint-free
harness::

    DP_RUN_FUSED_DEEPMD_PARITY=1 PYTHONPATH=python \\
        python -m unittest tests.csrc.accuracy.test_fused_deepmd_parity -v

Each test patches the seam through :mod:`model.fused_deepmd.deepmd_patch`, calls
the patched name, then uninstalls and calls the stock DeepMD implementation on
the same inputs. Both are the functions DeepMD itself calls, so this is an operator-level
parity check -- it does not need a checkpoint.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import torch

_RUN = os.environ.get("DP_RUN_FUSED_DEEPMD_PARITY") == "1"


@unittest.skipUnless(_RUN, "set DP_RUN_FUSED_DEEPMD_PARITY=1 on a built HIP/DCU host")
class FusedDeepMDDispatchParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kernels.segment import segment_available

        if not segment_available():
            raise RuntimeError("requires the built fused-segment extension on a supported DCU")

    def setUp(self):
        from model import fused_deepmd

        fused_deepmd.uninstall()
        self.addCleanup(fused_deepmd.uninstall)

    @staticmethod
    def _grouped_graph(atoms, edges):
        dst = torch.sort(torch.randint(0, atoms, (edges,), device="cuda")).values
        return dst

    def test_softmax_seam_matches_the_deepmd_reference(self):
        from deepmd.pt.model.descriptor.sezm_nn import attention as attention_module
        from model import fused_deepmd

        atoms, edges, focus, heads = 512, 8192, 2, 8
        dst = self._grouped_graph(atoms, edges)
        logits = torch.randn(edges, focus, heads, device="cuda")
        env = (torch.rand(edges, 1, device="cuda") + 0.1)
        bias = torch.randn(focus, heads, device="cuda")
        eps = 1e-6

        stats = fused_deepmd.deepmd_patch(
            SimpleNamespace(modules=lambda: []),
            namespaces=[attention_module],
            envelope=False,
            verbose=False,
        )
        self.assertTrue(stats.bound_softmax)
        actual = attention_module.segment_envelope_gated_softmax(
            logits, env, dst, atoms, bias, eps
        )
        fused_deepmd.uninstall()
        expected = attention_module.segment_envelope_gated_softmax(
            logits, env, dst, atoms, bias, eps
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(stats.softmax_hits, 1)
        self.assertEqual(stats.softmax_fallbacks, 0)
        # Each segment's edge mass plus the null mass is one.
        per_segment = torch.zeros(atoms, focus * heads, device="cuda")
        per_segment.index_add_(0, dst, actual.reshape(edges, -1))
        self.assertTrue(bool((per_segment <= 1.0 + 1e-5).all()))

    def test_envelope_seam_matches_the_deepmd_reference(self):
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Convolution
        from model import fused_deepmd

        atoms, edges, rows, channels = 64, 4096, 4, 32
        dst = self._grouped_graph(atoms, edges)
        message = torch.randn(edges, rows, channels, device="cuda")
        cache = SimpleNamespace(
            dst=dst,
            edge_env=torch.rand(edges, 1, device="cuda") + 0.1,
            inv_sqrt_deg=torch.rand(atoms, 1, 1, device="cuda") + 0.5,
            edge_src_gate=None,
        )
        block = SimpleNamespace(
            training=False,
            _segment_softmax_fn=None,
            precision="float32",
            compute_precision="float32",
            edge_message=lambda x, edge_cache, radial: (message, None),
        )
        x = torch.zeros(atoms, rows, channels, device="cuda")

        stats = fused_deepmd.deepmd_patch(
            SimpleNamespace(modules=lambda: [block]), softmax=False, verbose=False
        )
        self.assertTrue(stats.bound_envelope)
        actual = block.forward_envelope(x, cache, None)
        fused_deepmd.uninstall()
        expected = SO2Convolution.forward_envelope(block, x, cache, None)
        budget = 1e-5 * float(expected.abs().max())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=budget)
        self.assertEqual(stats.envelope_hits, 1)
        self.assertEqual(stats.envelope_fallbacks, 0)


if __name__ == "__main__":
    unittest.main()
