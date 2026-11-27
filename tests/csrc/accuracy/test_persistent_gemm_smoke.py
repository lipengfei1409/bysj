"""First thing to run on a DCU: five shapes that each isolate one mechanism.

`test_persistent_gemm.py` is the real accuracy suite; this file exists because a
failure there says "some shape is wrong" while a failure here says *which*
mechanism is wrong.  Run it before the full suite, and in this order:

    single tile      -> pages, staging, fragment load/mma/store epilogue
    two tiles        -> two consumer waves, one instruction
    full run of four -> four waves, four C pages, the storer over four tiles
    short run (N=48) -> the last run is three tiles: wave 3 must stay idle
    deep K           -> accumulation across many k-tiles, four waves in flight

Every case compares against `torch.matmul`, which is rocBLAS on the device, so
this is the vendor product and not a hand-written reference.
"""
import unittest

import torch

from dmk import dispatch, emitters, mk

try:
    _AVAILABLE = dispatch.available()
except Exception:  # noqa: BLE001 - a host build has no extension
    _AVAILABLE = False

_SKIP = ("requires the built persistent GEMM (kernels._C.dmk_gemm from "
         "csrc/dmk/megakernel.cpp) and a supported DCU")


def _check(m: int, n: int, k: int, cus: int = 1, seed: int = 0) -> None:
    globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=seed)
    generator = emitters.GemmEmitter(m, n, k)
    globals_.attach(generator.queues(cus))
    dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b, globals_.c)
    torch.cuda.synchronize()
    expected = globals_.a @ globals_.b
    torch.testing.assert_close(globals_.c, expected, rtol=1e-4, atol=1e-4)


@unittest.skipUnless(_AVAILABLE and torch.cuda.is_available(), _SKIP)
class SmokeTests(unittest.TestCase):
    def test_single_tile_single_k_step(self) -> None:
        # 16³: one instruction, one k-step, one wave. Nothing else can be wrong.
        _check(16, 16, 16)

    def test_two_tiles(self) -> None:
        # 32 wide, 16 tall: one instruction covering two tiles, waves 0 and 1.
        _check(16, 32, 16)

    def test_a_full_run_of_four_tiles(self) -> None:
        # 16x64: four tiles in one instruction, all four consumer waves working.
        _check(16, 64, 16)

    def test_a_short_run_leaves_the_last_wave_idle(self) -> None:
        # 16x48: three tiles, so wave 3 has no column and must write nothing.
        _check(16, 48, 16)

    def test_deep_k_accumulates_across_waves(self) -> None:
        # 128³ with four waves in flight over eight k-tiles each.
        _check(128, 128, 128)

    def test_a_rectangular_shape_with_several_runs(self) -> None:
        # 64x256: four tile rows, each one full run of four.
        _check(64, 256, 64)


if __name__ == "__main__":
    unittest.main()
