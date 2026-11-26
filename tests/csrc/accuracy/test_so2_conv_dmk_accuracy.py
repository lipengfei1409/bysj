"""Opt-in accuracy tests for the megakernel SO(2) conv operator (C8).

Skipped until the operator is wired, and the skip names the two missing steps so
a later reader does not mistake "no tests ran" for "the operator is fine".

Once it is wired, ``kernels.so2_conv_dmk.so2_conv_dmk`` is expected to have the
same signature as the reference:

    so2_conv_dmk(x, src, rot, rot_t, kc, alpha) -> msg   (E, dim, C)

with ``x``/``src``/``kc`` as the operator reads them (see
``docs/so2-conv-dmk.md`` -- the manifest and the page layout).
"""
import unittest

import torch

from reference.so2_conv_dmk import so2_conv_dmk_reference

try:  # the dmk subtree ships its own extension; absent on a host build
    from kernels.dmk import so2_conv_dmk, so2_conv_dmk_available
except Exception:  # noqa: BLE001 - not wired yet, skip loudly below
    so2_conv_dmk = None

    def so2_conv_dmk_available() -> bool:
        return False


_SKIP = (
    "the dmk extension is not built here: on a DCU run "
    "`python csrc/dmk/setup.py build_ext --inplace`, which provides "
    "kernels.dmk.so2_conv_dmk (see docs/so2-conv-dmk.md)"
)


def _problem(edges, dim, red, layers, channels, nodes=8, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(nodes, dim, channels, device="cuda")
    src = torch.randint(0, nodes, (edges,), device="cuda")
    rot = (torch.randn(edges, red, dim, device="cuda") * 0.3).contiguous()
    rot_t = (torch.randn(edges, dim, red, device="cuda") * 0.3).contiguous()
    kc = (torch.randn(edges, layers, 2, red, red, device="cuda") * 0.1).contiguous()
    alpha = (torch.rand(edges, device="cuda") + 0.5).contiguous()
    return x, src, rot, rot_t, kc, alpha


@unittest.skipUnless(
    so2_conv_dmk_available() and torch.cuda.is_available(), _SKIP
)
class So2ConvDmkAccuracyTests(unittest.TestCase):
    def _check(self, **shape):
        problem = _problem(**shape)
        actual = so2_conv_dmk(*problem)
        expected = so2_conv_dmk_reference(*problem)
        torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)

    def test_the_profiled_shape_matches_the_reference(self) -> None:
        # lmax=3: dim 16, red 10, four gated layers, the 64-lane one-channel-per-lane width.
        self._check(edges=32, dim=16, red=10, layers=4, channels=64)

    def test_a_narrow_channel_width_keeps_the_idle_lanes_quiet(self) -> None:
        # C < 64 leaves lanes idle; their writes must not touch the message.
        self._check(edges=16, dim=16, red=10, layers=4, channels=5)

    def test_edge_tails_are_covered(self) -> None:
        # kConsumerWaves is 4, so 1/3/5/7 edges exercise every tail phase of the
        # wave-strided loop inside run().
        for edges in (1, 3, 5, 7):
            with self.subTest(edges=edges):
                self._check(edges=edges, dim=16, red=10, layers=4, channels=64)

    def test_a_single_layer_single_lmax_matches(self) -> None:
        self._check(edges=5, dim=4, red=4, layers=1, channels=64)

    def test_the_result_is_deterministic(self) -> None:
        problem = _problem(edges=8, dim=16, red=10, layers=4, channels=64)
        first = so2_conv_dmk(*problem)
        second = so2_conv_dmk(*problem)
        torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
