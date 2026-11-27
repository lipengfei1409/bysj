"""Opt-in accuracy test for the persistent GEMM (`csrc/dmk/gemm_launch.cpp`).

The megakernel is launched with a GEMM-only op pack, so this is the whole
persistent path under test: the host split (`python/dmk/generators.py`) produces the
instruction rows, `ops::GemmOp` walks them, and the result must equal a dense
`A @ B`.

Skipped unless the built extension exposes `dmk_gemm` and a DCU is present.
"""
import unittest

import torch

from dmk import dispatch, generators, mk

try:  # the entry point comes with the default build
    from kernels.native import _extension

    _extension("dmk_gemm")
    _available = True
except Exception:  # noqa: BLE001 - absent on a host build, skip loudly below
    _available = False

_SKIP = ("requires the built dmk_gemm entry point (csrc/dmk/gemm_launch.cpp) "
         "and a supported DCU")


def _run(m: int, n: int, k: int, cus: int, seed: int = 0):
    globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=seed)
    globals_.attach(generators.gemm_queues(m, n, k, cus))
    dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b, globals_.c)
    torch.cuda.synchronize()
    return globals_.c, globals_.a @ globals_.b


@unittest.skipUnless(_available and torch.cuda.is_available(), _SKIP)
class PersistentGemmAccuracyTests(unittest.TestCase):
    def test_one_instruction_per_tile_matches_dense(self) -> None:
        c, expected = _run(32, 32, 16, cus=1)
        torch.testing.assert_close(c, expected, rtol=1e-4, atol=1e-4)

    def test_several_cus_cover_the_grid(self) -> None:
        # 128x64 out of 8*4 = 32 tiles over 4 CUs: every tile must be written.
        c, expected = _run(128, 64, 32, cus=4)
        torch.testing.assert_close(c, expected, rtol=1e-4, atol=1e-4)

    def test_a_tall_skinny_shape(self) -> None:
        c, expected = _run(256, 16, 64, cus=3)
        torch.testing.assert_close(c, expected, rtol=1e-4, atol=1e-4)

    def test_more_cus_than_tiles_still_covers_everything(self) -> None:
        # 2 tiles over 6 CUs: four CUs idle, the two that work must be right.
        c, expected = _run(32, 16, 16, cus=6)
        torch.testing.assert_close(c, expected, rtol=1e-4, atol=1e-4)

    def test_the_result_is_deterministic(self) -> None:
        c1, _ = _run(64, 64, 32, cus=2, seed=7)
        c2, _ = _run(64, 64, 32, cus=2, seed=7)
        torch.testing.assert_close(c1, c2, rtol=0, atol=0)

    def test_a_shape_that_needs_an_edge_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generators.gemm_instructions(33, 16, 16)


if __name__ == "__main__":
    unittest.main()
