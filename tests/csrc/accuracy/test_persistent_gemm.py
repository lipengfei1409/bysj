"""DCU accuracy tests for the persistent GEMM, in both of its shapes.

Two kernels, one schedule:

* `csrc/dmk/megakernel.cpp` -- the scheduler: controller / loader / consumer / storer
  waves, semaphores, page permutation.  Entry point `dmk_gemm`.
* `csrc/dmk/gemm_flat.cpp` -- the same instruction stream executed with one role:
  cooperative staging, `__syncthreads()` instead of handshakes, and a double-buffered
  k loop.  Entry point `dmk_gemm_flat`.

Both must produce the vendor product, and they must agree with each other bit for bit,
because they accumulate in the same order from the same fragments.  That agreement is
the cheapest way to catch a scheduling bug: numerically it is not a tolerance question.

`torch.matmul` is rocBLAS on the device, so the comparison is against the vendor
implementation rather than a hand-written reference.
"""
import unittest

import torch

from dmk import dispatch, emitters, mk
from dmk.model_types import GEMM_TILE, GEMM_TILES_PER_INSTRUCTION

try:  # both engines come with the default build
    _AVAILABLE = dispatch.available(flat=False)
    _FLAT_AVAILABLE = dispatch.flat_available()
except Exception:  # noqa: BLE001 - a host build simply has no extension
    _AVAILABLE = _FLAT_AVAILABLE = False

_SKIP = ("requires the built persistent GEMM (kernels._C.dmk_gemm from "
         "csrc/dmk/megakernel.cpp) and a supported DCU")
_FLAT_SKIP = ("requires the role-free persistent GEMM (kernels._C.dmk_gemm_flat from "
              "csrc/dmk/gemm_flat.cpp) and a supported DCU")

#: Small shapes pin the tile machinery (single tile, one k-step, tails); the large ones
#: pin that the schedule and the kernel still agree when the grid has thousands of
#: tiles and the operands no longer fit in L2.
SHAPES = (
    (16, 16, 16),
    (32, 32, 32),
    (64, 48, 32),
    (256, 16, 64),
    (32, 32, 128),
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (2048, 512, 2048),
    (512, 2048, 512),
)


def _run(m: int, n: int, k: int, cus: int, seed: int = 0, *,
         flat: bool = False) -> mk.GemmGlobals:
    globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=seed)
    globals_.attach(emitters.GemmEmitter(m, n, k).queues(cus))
    dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b, globals_.c,
                         flat=flat)
    torch.cuda.synchronize()
    return globals_


@unittest.skipUnless(_AVAILABLE and torch.cuda.is_available(), _SKIP)
class ScheduledGemmAccuracyTests(unittest.TestCase):
    def test_every_shape_matches_a_dense_product(self) -> None:
        for m, n, k in SHAPES:
            for cus in (1, 4, 32):
                with self.subTest(shape=(m, n, k), cus=cus):
                    globals_ = _run(m, n, k, cus)
                    torch.testing.assert_close(globals_.c, globals_.a @ globals_.b,
                                               rtol=1e-4, atol=1e-4)

    def test_the_result_is_independent_of_the_cu_count(self) -> None:
        first = _run(64, 64, 32, cus=1).c.clone()
        for cus in (2, 3, 5):
            torch.testing.assert_close(_run(64, 64, 32, cus=cus).c, first,
                                       rtol=1e-5, atol=1e-5)

    def test_relaunching_does_not_accumulate(self) -> None:
        globals_ = _run(32, 32, 32, cus=2)
        once = globals_.c.clone()
        dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b,
                             globals_.c)
        torch.cuda.synchronize()
        torch.testing.assert_close(globals_.c, once, rtol=0, atol=0)

    def test_more_cus_than_tiles_still_writes_every_tile(self) -> None:
        globals_ = _run(32, 16, 16, cus=6)
        torch.testing.assert_close(globals_.c, globals_.a @ globals_.b,
                                   rtol=1e-4, atol=1e-4)

    def test_a_shape_without_an_edge_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mk.GemmGlobals.from_shapes(17, 16, 16)
        with self.assertRaises(ValueError):
            emitters.GemmEmitter(16, 16, 17)

    def test_the_instruction_count_matches_the_run_grid(self) -> None:
        for m, n, k in SHAPES:
            emitter = emitters.GemmEmitter(m, n, k)
            runs = -(-(n // GEMM_TILE) // GEMM_TILES_PER_INSTRUCTION)
            self.assertEqual(len(emitter.instructions()), (m // GEMM_TILE) * runs)


@unittest.skipUnless(_FLAT_AVAILABLE and torch.cuda.is_available(), _FLAT_SKIP)
class RoleFreeGemmAccuracyTests(unittest.TestCase):
    def test_every_shape_matches_a_dense_product(self) -> None:
        for m, n, k in SHAPES:
            for cus in (1, 4, 32):
                with self.subTest(shape=(m, n, k), cus=cus):
                    globals_ = _run(m, n, k, cus, flat=True)
                    torch.testing.assert_close(globals_.c, globals_.a @ globals_.b,
                                               rtol=1e-4, atol=1e-4)

    def test_both_engines_agree_bit_for_bit(self) -> None:
        for m, n, k in ((64, 48, 32), (512, 512, 512)):
            with self.subTest(shape=(m, n, k)):
                scheduled = _run(m, n, k, cus=4, seed=3).c.clone()
                flat = _run(m, n, k, cus=4, seed=3, flat=True).c
                torch.testing.assert_close(flat, scheduled, rtol=0, atol=0)

    def test_a_shape_without_an_edge_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            emitters.GemmEmitter(17, 16, 16)


if __name__ == "__main__":
    unittest.main()
