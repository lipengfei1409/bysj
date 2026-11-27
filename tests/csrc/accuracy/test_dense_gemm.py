"""DCU accuracy tests for the ordinary dense GEMM (`csrc/gemm.cpp`).

This kernel is the yardstick the persistent ones are measured against, so it has to
be right on the same shapes and against the same reference -- `torch.matmul`, which is
rocBLAS on the device.  Unlike the persistent kernels it handles edges by predicate,
so shapes that do not fill a 64x64 block tile are part of the contract here.
"""
import unittest

import torch

from dmk import dispatch

try:
    _AVAILABLE = dispatch.dense_available()
except Exception:  # noqa: BLE001 - a host build has no extension
    _AVAILABLE = False

_SKIP = ("requires the built dense GEMM (kernels._C.gemm_dense from csrc/gemm.cpp) "
         "and a supported DCU")

#: Tile-filling shapes, then shapes that deliberately leave a partial block.
SHAPES = (
    (64, 64, 8),
    (64, 64, 64),
    (128, 128, 128),
    (256, 512, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (2048, 512, 2048),
    (512, 2048, 512),
)

PARTIAL_SHAPES = (
    (1, 1, 1),
    (17, 33, 5),
    (63, 65, 7),
    (100, 100, 100),
    (129, 257, 33),
)


def _check(m: int, n: int, k: int, seed: int = 0, rtol: float = 1e-4) -> None:
    torch.manual_seed(seed)
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    c = dispatch.launch_dense(a, b)
    torch.cuda.synchronize()
    expected = a @ b
    scale = float(expected.abs().max())
    torch.testing.assert_close(c, expected, rtol=rtol, atol=rtol * max(scale, 1.0))


@unittest.skipUnless(_AVAILABLE and torch.cuda.is_available(), _SKIP)
class DenseGemmAccuracyTests(unittest.TestCase):
    def test_tile_filling_shapes_match_rocblas(self) -> None:
        for m, n, k in SHAPES:
            with self.subTest(shape=(m, n, k)):
                _check(m, n, k)

    def test_partial_tiles_match_rocblas(self) -> None:
        # Edges are handled by predicate rather than by rejecting the shape.
        for m, n, k in PARTIAL_SHAPES:
            with self.subTest(shape=(m, n, k)):
                _check(m, n, k, rtol=1e-4)

    def test_the_result_does_not_depend_on_the_launch_order(self) -> None:
        m, n, k = 128, 128, 64
        torch.manual_seed(7)
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        first = dispatch.launch_dense(a, b).clone()
        second = dispatch.launch_dense(a, b)
        torch.testing.assert_close(second, first, rtol=0, atol=0)

    def test_a_mismatched_k_is_rejected(self) -> None:
        a = torch.zeros(8, 16, device="cuda")
        b = torch.zeros(8, 16, device="cuda")
        with self.assertRaises(RuntimeError):
            dispatch.launch_dense(a, b)

    def test_the_output_is_a_fresh_tensor(self) -> None:
        a = torch.randn(16, 16, device="cuda")
        b = torch.randn(16, 16, device="cuda")
        c = dispatch.launch_dense(a, b)
        self.assertEqual(c.shape, (16, 16))
        self.assertIsNot(c, a)
        self.assertIsNot(c, b)


if __name__ == "__main__":
    unittest.main()
