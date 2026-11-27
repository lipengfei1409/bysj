"""Host tests for the persistent host side (`python/dmk`).

The layout follows `Megakernels/megakernels`: `model_types`/`utils` hold the
value types and the division of work, `generators` emits instructions,
`scheduler` assigns them, `python_vm` executes a schedule in Python.  Because the
last one runs without a device, the schedule can be checked end to end here --
coverage, order, and the product itself.
"""

from __future__ import annotations

import unittest

import torch

from dmk import dispatch, generators, mk, python_vm, scheduler
from dmk import instructions as I
from dmk.model_types import GEMM, Work
from dmk.utils import (get_cu_count, instruction_counts, plan_cus, split_units,
                       tile_for)
from kernels import mk_stream


class SplitTests(unittest.TestCase):
    def test_the_shares_sum_to_the_total(self) -> None:
        for total in (0, 1, 7, 100, 2047, 2048):
            for parts in (1, 2, 3, 8):
                shares = split_units(total, parts)
                self.assertEqual(sum(shares), total, (total, parts))
                self.assertEqual(len(shares), parts)
                self.assertLessEqual(max(shares) - min(shares), 1)

    def test_the_remainder_goes_to_the_earliest_cus(self) -> None:
        self.assertEqual(split_units(10, 3), [4, 3, 3])

    def test_tiles_follow_the_share(self) -> None:
        self.assertEqual(tile_for(2048, 0), 0)
        self.assertEqual(tile_for(2048, 3), 3)
        self.assertEqual(tile_for(2048, 5000), 2048)

    def test_bad_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_units(5, 0)
        with self.assertRaises(ValueError):
            split_units(-1, 2)


class WorkTests(unittest.TestCase):
    def test_elementwise_is_counted_in_pages(self) -> None:
        from dmk.model_types import ELEMENTWISE, ELEMENTWISE_PAGE

        work = Work(elementwise=ELEMENTWISE_PAGE * 3 + 1)
        self.assertEqual(work.slots(ELEMENTWISE), 4)

    def test_only_kinds_with_work_are_kinds(self) -> None:
        self.assertEqual(Work(gemm_tiles=3).kinds(), (GEMM,))

    def test_every_cu_gets_a_tile_that_fits_its_share(self) -> None:
        work = Work(elementwise=2048 * 10, gemm_tiles=9, embedding_rows=7,
                    attention_queries=5, embedding_tile=4, attention_tile=2)
        plans = plan_cus(work, 4)
        counts = instruction_counts(work, plans)
        self.assertEqual(len(plans), 4)
        for kind in work.kinds():
            shares = split_units(work.slots(kind), 4)
            for plan in plans:
                step = plan.tiles.slots_per_instruction(kind)
                if shares[plan.cu] == 0:
                    self.assertEqual(step, 0, (kind, plan.cu))
                else:
                    self.assertGreaterEqual(step, 1, (kind, plan.cu))
                    self.assertEqual(
                        counts[kind][plan.cu], -(-shares[plan.cu] // step), (kind, plan.cu)
                    )

    def test_the_elementwise_tile_is_one_page(self) -> None:
        from dmk.model_types import CuTileConfig, ELEMENTWISE_PAGE

        plan = plan_cus(Work(elementwise=ELEMENTWISE_PAGE * 4), 1)[0]
        self.assertEqual(plan.tiles.elementwise, ELEMENTWISE_PAGE)
        del CuTileConfig  # the type itself is exercised through plan_cus

    def test_the_cu_count_can_be_overridden(self) -> None:
        import os

        os.environ["DMK_CU_COUNT"] = "3"
        try:
            self.assertEqual(get_cu_count(), 3)
        finally:
            del os.environ["DMK_CU_COUNT"]
        self.assertGreaterEqual(get_cu_count(), 1)


class GeneratorTests(unittest.TestCase):
    def test_one_instruction_per_output_tile(self) -> None:
        instructions = generators.gemm_instructions(64, 48, 32)
        self.assertEqual(len(instructions), (64 // 16) * (48 // 16))
        self.assertTrue(all(item.k_tiles == 2 for item in instructions))
        self.assertEqual({(i.m0, i.n0) for i in instructions},
                         {(m, n) for m in (0, 16, 32, 48) for n in (0, 16, 32)})

    def test_a_shape_without_an_edge_path_is_rejected(self) -> None:
        for shape in ((17, 16, 16), (16, 17, 16), (16, 16, 17)):
            with self.assertRaises(ValueError):
                generators.gemm_instructions(*shape)

    def test_the_split_covers_every_tile_once(self) -> None:
        rows = generators.gemm_queues(64, 64, 32, 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(len(row) for row in rows), 16)
        self.assertEqual([len(row) for row in rows], [6, 5, 5])


class EmulatorTests(unittest.TestCase):
    def _globals(self, m, n, k, cus):
        globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=0, device="cpu")
        globals_.attach(generators.gemm_queues(m, n, k, cus))
        return globals_

    def test_the_emulated_schedule_reproduces_the_product(self) -> None:
        globals_ = self._globals(64, 64, 32, 4)
        self.assertLess(mk.verify(globals_, emulated=True), 1e-4)

    def test_every_tile_is_written_exactly_once(self) -> None:
        globals_ = self._globals(64, 48, 32, 3)
        counts = python_vm.count_tiles(globals_.instructions)
        self.assertEqual(len(counts), (64 // 16) * (48 // 16))
        self.assertTrue(all(count == 1 for count in counts.values()), counts)

    def test_a_missing_tile_shows_up_as_an_error(self) -> None:
        # Drop one instruction: the emulated product must disagree.
        globals_ = self._globals(64, 64, 32, 4)
        rows = globals_.instructions.tolist()
        for row in rows:
            for index, words in enumerate(row):
                if words[0] == mk_stream.OPCODE_GEMM:
                    row[index] = [0] * mk_stream.INSTRUCTION_WIDTH
                    break
            else:
                continue
            break
        globals_.instructions = torch.tensor(rows, dtype=torch.int32)
        self.assertGreater(mk.verify(globals_, emulated=True), 1e-3)

    def test_the_emulator_rejects_an_unknown_opcode(self) -> None:
        stream = I.InstructionStream([[I.NoOp()]])
        words = stream.tensor().tolist()
        words[0][0][0] = 99
        with self.assertRaises(ValueError):
            python_vm.decode_row(torch.tensor(words, dtype=torch.int32).tolist()[0])


class DispatchTests(unittest.TestCase):
    def test_the_georgia_path_reports_a_missing_build(self) -> None:
        # On a host without the extension the caller must be told, not silently
        # given an un-launched tensor.
        if dispatch.available():
            self.skipTest("this host can launch the persistent GEMM")
        with self.assertRaises(RuntimeError) as caught:
            dispatch.launch_gemm(torch.zeros(1, 1, 32, dtype=torch.int32),
                                 torch.zeros(16, 16), torch.zeros(16, 16),
                                 torch.zeros(16, 16))
        self.assertIn("not built", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
