"""Host tests for the persistent GEMM's schedule and its Python VM.

Scope: everything about the persistent GEMM that can be checked without a DCU --
the instruction sequence, the per-CU split, the emulator's replay, and the shapes
the operator refuses.  The device-side half lives in
`tests/csrc/accuracy/test_persistent_gemm.py`; between the two, the schedule and
the kernel are covered by the same shapes and the same expected product.
"""

from __future__ import annotations

import itertools
import unittest

import torch

from dmk import dispatch, emitters, mk, python_vm
from dmk.demos.gemm import GemmDemo
from dmk.instructions import InstructionStream
from dmk.model_types import GEMM_TILE, GEMM_TILES_PER_INSTRUCTION
from kernels import mk_stream

#: Shapes exercised everywhere below: square, tall, wide, single tile, deep-K,
#: and K = 16 (one k-step).
SHAPES = (
    (16, 16, 16),
    (32, 32, 32),
    (64, 48, 32),
    (256, 16, 64),
    (16, 256, 64),
    (32, 32, 128),
)
CU_COUNTS = (1, 2, 3, 5)


def _schedule(m: int, n: int, k: int, cus: int) -> InstructionStream:
    return InstructionStream(emitters.GemmEmitter(m, n, k).queues(cus))


class ScheduleTests(unittest.TestCase):
    def test_the_tile_grid_is_covered_exactly_once(self) -> None:
        for m, n, k in SHAPES:
            for cus in CU_COUNTS:
                with self.subTest(shape=(m, n, k), cus=cus):
                    stream = _schedule(m, n, k, cus)
                    counts = python_vm.count_tiles(stream.tensor(), n)
                    expected = (m // GEMM_TILE) * (n // GEMM_TILE)
                    self.assertEqual(len(counts), expected)
                    self.assertTrue(all(c == 1 for c in counts.values()), counts)

    def test_every_tile_of_the_grid_appears(self) -> None:
        m, n, k = 64, 48, 32
        stream = _schedule(m, n, k, cus=3)
        seen = set(python_vm.count_tiles(stream.tensor(), n))
        expected = {
            (row * GEMM_TILE, column * GEMM_TILE)
            for row in range(m // GEMM_TILE)
            for column in range(n // GEMM_TILE)
        }
        self.assertEqual(seen, expected)

    def test_every_instruction_carries_the_shapes_k_steps(self) -> None:
        for m, n, k in SHAPES:
            stream = _schedule(m, n, k, cus=2)
            for row in stream.tensor().to("cpu").tolist():
                for words in row:
                    if words[0] != mk_stream.OPCODE_GEMM:
                        continue
                    self.assertEqual(words[3], k // GEMM_TILE, (m, n, k))

    def test_the_rows_are_a_rectangle_of_known_opcodes(self) -> None:
        for m, n, k in SHAPES:
            for cus in CU_COUNTS:
                tensor = _schedule(m, n, k, cus).tensor()
                self.assertEqual(tensor.shape[0], cus)
                opcodes = {words[0] for words in tensor.reshape(-1, 32).tolist()}
                self.assertLessEqual(opcodes, mk_stream.KNOWN_OPCODES)

    def test_idle_cus_get_only_noops(self) -> None:
        # 2 tiles over 5 CUs: three rows must be pure padding.
        stream = _schedule(32, 16, 16, cus=5)
        blocks = stream.tensor().to("cpu").tolist()   # [cu][instruction][word]
        self.assertEqual(stream.length, 1)
        for row in blocks[2:]:
            for words in row:
                self.assertEqual(words, [0] * mk_stream.INSTRUCTION_WIDTH,
                                 "idle CU is not padded with NoOp")

    def test_the_split_is_balanced_to_within_one_instruction(self) -> None:
        for m, n, k in SHAPES:
            for cus in CU_COUNTS:
                rows = emitters.GemmEmitter(m, n, k).queues(cus)
                lengths = [len(row) for row in rows]
                self.assertLessEqual(max(lengths) - min(lengths), 1, (m, n, k, cus))


class EmulatedProductTests(unittest.TestCase):
    def test_the_emulated_product_matches_a_dense_matmul(self) -> None:
        for m, n, k in SHAPES:
            for cus in (1, 3):
                with self.subTest(shape=(m, n, k), cus=cus):
                    globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=7,
                                                          device="cpu")
                    globals_.attach(emitters.GemmEmitter(m, n, k).queues(cus))
                    self.assertLess(mk.verify(globals_, emulated=True), 1e-4)

    def test_k_16_is_a_single_k_step_and_still_correct(self) -> None:
        globals_ = mk.GemmGlobals.from_shapes(16, 16, 16, seed=1, device="cpu")
        globals_.attach(emitters.GemmEmitter(16, 16, 16).queues(2))
        words = globals_.instructions.to("cpu").tolist()
        self.assertEqual([w for row in words for w in row if w[0]][0][3], 1)
        self.assertLess(mk.verify(globals_, emulated=True), 1e-4)

    def test_a_duplicated_tile_is_visible(self) -> None:
        globals_ = mk.GemmGlobals.from_shapes(32, 32, 16, seed=3, device="cpu")
        globals_.attach(emitters.GemmEmitter(32, 32, 16).queues(2))
        rows = globals_.instructions.to("cpu").tolist()
        # Overwrite the last GEMM instruction of one CU with the first one: one
        # tile is computed twice, another not at all.  The product must be wrong
        # and the audit must see both facts.
        words = [row[:] for row in rows]
        first = next(w for row in words for w in row if w[0] == mk_stream.OPCODE_GEMM)
        for index in range(len(words[1]) - 1, -1, -1):
            if words[1][index][0] == mk_stream.OPCODE_GEMM:
                words[1][index] = first
                break
        globals_.instructions = torch.tensor(words, dtype=torch.int32)
        self.assertGreater(mk.verify(globals_, emulated=True), 1e-3)
        counts = python_vm.count_tiles(globals_.instructions, globals_.b.shape[1])
        # 32x32 is two rows of one run (two tiles each).  Copying row 1's
        # instruction from row 0 leaves row 1 unwritten and writes row 0 twice.
        self.assertEqual(max(counts.values()), 2)
        self.assertEqual(len(counts), 2)

    def test_the_result_does_not_depend_on_the_cu_count(self) -> None:
        m, n, k = 64, 64, 32
        products = []
        for cus in CU_COUNTS:
            globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=11, device="cpu")
            globals_.attach(emitters.GemmEmitter(m, n, k).queues(cus))
            products.append(mk.emulate_gemm(globals_))
        for other in products[1:]:
            torch.testing.assert_close(other, products[0], rtol=0, atol=0)


class RejectionTests(unittest.TestCase):
    def test_a_shape_without_an_edge_path_is_rejected_everywhere(self) -> None:
        # Three independent guards: the emitter, the globals, and the C++ launcher
        # (the last one is covered by the DCU test).
        for m, n, k in ((17, 16, 16), (16, 17, 16), (16, 16, 17), (8, 16, 16)):
            with self.subTest(shape=(m, n, k)):
                with self.assertRaises(ValueError):
                    emitters.GemmEmitter(m, n, k)
                with self.assertRaises(ValueError):
                    mk.GemmGlobals.from_shapes(m, n, k, device="cpu")

    def test_a_zero_cu_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            emitters.GemmEmitter(16, 16, 16).queues(0)

    def test_launching_without_a_stream_is_rejected(self) -> None:
        globals_ = mk.GemmGlobals.from_shapes(16, 16, 16, device="cpu")
        with self.assertRaises(RuntimeError):
            mk.launch_gemm(globals_)
        with self.assertRaises(RuntimeError):
            mk.emulate_gemm(globals_)


class AvailabilityTests(unittest.TestCase):
    def test_dispatch_reports_the_build_honestly(self) -> None:
        # Either the extension is loadable, or the call must say so -- never a
        # silent no-op.
        if dispatch.available():
            self.skipTest("this host can launch the persistent GEMM")
        with self.assertRaises(RuntimeError) as caught:
            dispatch.launch_gemm(
                torch.zeros(1, 1, 32, dtype=torch.int32),
                torch.zeros(16, 16), torch.zeros(16, 16), torch.zeros(16, 16),
            )
        self.assertIn("not built", str(caught.exception))


class DemoTests(unittest.TestCase):
    def test_the_demo_builds_a_schedule_for_every_cu_count(self) -> None:
        for cus in (1, 3):
            demo = GemmDemo(m=32, n=48, k=32, cus=cus)
            rows = demo.rows()
            self.assertEqual(len(rows), cus)
            # rows x runs, where a run is GEMM_TILES_PER_INSTRUCTION tiles
        runs = -(-(48 // 16) // GEMM_TILES_PER_INSTRUCTION)
        self.assertEqual(sum(len(row) for row in rows), (32 // 16) * runs)

    def test_the_demo_rejects_an_untileable_shape(self) -> None:
        with self.assertRaises(ValueError):
            GemmDemo(m=20, n=16, k=16).rows()

    def test_the_demo_can_be_checked_without_a_device(self) -> None:
        demo = GemmDemo(m=32, n=32, k=16, cus=2)
        globals_ = demo.globals(seed=5, device="cpu")
        self.assertLess(demo.verify(globals_, emulated=True), 1e-4)


class CrossCheckTests(unittest.TestCase):
    def test_every_shape_is_consistent_across_the_three_layers(self) -> None:
        # Emitter -> InstructionStream -> Python VM: the same tile set, the same
        # product, for every shape and CU count in the matrix above.
        for (m, n, k), cus in itertools.product(SHAPES, (1, 3)):
            with self.subTest(shape=(m, n, k), cus=cus):
                emitter = emitters.GemmEmitter(m, n, k)
                # one instruction per run of up to GEMM_TILES_PER_INSTRUCTION
                # tiles along N
                self.assertEqual(
                    len(emitter.instructions()),
                    (m // GEMM_TILE) * -(-(n // GEMM_TILE) // GEMM_TILES_PER_INSTRUCTION),
                )
                globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=2, device="cpu")
                globals_.attach(emitter.queues(cus))
                self.assertLess(mk.verify(globals_, emulated=True), 1e-4)


if __name__ == "__main__":
    unittest.main()
