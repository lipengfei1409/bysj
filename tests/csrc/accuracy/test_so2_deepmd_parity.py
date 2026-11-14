"""Opt-in parity of native reduction against the current DPA4 forward_envelope.

DP_RUN_SO2_DEEPMD_PARITY=1 PYTHONPATH=python python3 -m unittest discover \
    -s tests/csrc/accuracy -p test_so2_deepmd_parity.py -v

The harness supplies already-computed edge messages. This verifies the actual
DeepMD aggregation method, not the preceding SO(2) rotation/mixing stack.
"""
import os
import unittest
from types import SimpleNamespace

import torch

from kernels.so2_reduce import so2_envelope_reduce, so2_envelope_reduce_available


@unittest.skipUnless(os.environ.get("DP_RUN_SO2_DEEPMD_PARITY") == "1",
                     "set DP_RUN_SO2_DEEPMD_PARITY=1 on a built HIP/DCU host")
class SO2DeepMDParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not so2_envelope_reduce_available():
            raise RuntimeError("requires the built native SO2 extension on a supported DCU")
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Convolution
        cls.upstream = staticmethod(SO2Convolution.forward_envelope)

    def test_forward_and_all_value_gradients(self):
        torch.manual_seed(17)
        for edges in (0, 31):
            for coeff in (4, 9, 16):
                for gated in (False, True):
                    with self.subTest(edges=edges, coeff=coeff, gated=gated):
                        n, c = 7, 5
                        dst = torch.arange(edges, device="cuda", dtype=torch.int64) % n
                        message = torch.randn(edges, coeff, c, device="cuda", requires_grad=True)
                        envelope = torch.rand(edges, 1, device="cuda", requires_grad=True)
                        inverse = torch.rand(n, 1, 1, device="cuda", requires_grad=True)
                        gate = torch.rand(edges, 1, device="cuda", requires_grad=True) if gated else None
                        cache = SimpleNamespace(dst=dst, edge_env=envelope,
                            inv_sqrt_deg=inverse, edge_src_gate=gate)
                        harness = SimpleNamespace(precision="float32", compute_precision="float32",
                            edge_message=lambda x, cache, radial: (message, None))
                        expected = self.upstream(harness, torch.zeros(n, coeff, c, device="cuda"), cache, None)
                        weight = envelope if gate is None else envelope * gate
                        actual = so2_envelope_reduce(message, weight, dst, inverse)
                        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
                        values = (message, envelope, inverse) + (() if gate is None else (gate,))
                        upstream = torch.randn_like(actual)
                        actual_grads = torch.autograd.grad(actual, values, upstream, retain_graph=True)
                        # Empty scatter may disconnect message/weight in DeepMD.
                        expected_grads = torch.autograd.grad(expected, values, upstream, allow_unused=True)
                        for value, a, b in zip(values, actual_grads, expected_grads):
                            if b is None:
                                b = torch.zeros_like(value)
                            torch.testing.assert_close(a, b, atol=3e-5, rtol=4e-4)

    def test_backward_rejects_wrong_coefficient_shape(self):
        from kernels.native import _extension
        message = torch.ones(2, 4, 3, device="cuda")
        weight = torch.ones(2, device="cuda")
        dst = torch.tensor([0, 1], device="cuda")
        inverse = torch.ones(2, device="cuda")
        bad = torch.ones(2, 2, 6, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "grad_output must have shape"):
            _extension("so2_envelope_reduce_backward").so2_envelope_reduce_backward(
                bad, message, weight, dst, inverse, bad)


if __name__ == "__main__":
    unittest.main()
