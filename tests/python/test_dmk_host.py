"""Host tests for the persistent host side (`python/dmk`).

The layout follows `Megakernels/megakernels`: `model_types`/`utils` hold the
value types and the division of work, `emitters` emits instructions,
`scheduler` assigns them, `python_vm` executes a schedule in Python.  Because the
last one runs without a device, the schedule can be checked end to end here --
coverage, order, and the product itself.
"""

from __future__ import annotations

import unittest

import torch

from dmk import dispatch, emitters, mk, python_vm, scheduler
from dmk import instructions as I
from dmk.model_types import GEMM, GEMM_TILES_PER_INSTRUCTION, Work
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
        self.assertEqual(Work(gemm_runs=3).kinds(), (GEMM,))

    def test_every_cu_gets_a_tile_that_fits_its_share(self) -> None:
        work = Work(elementwise=2048 * 10, gemm_runs=9, embedding_rows=7,
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


class EmitterTests(unittest.TestCase):
    def test_one_instruction_per_run_of_tiles(self) -> None:
        # 64x48 is 4 tile rows x 3 tile columns, so two runs of two per row (the
        # second run is short when the run length does not divide the row).
        instructions = emitters.gemm_instructions(64, 48, 32)
        self.assertEqual(len(instructions), 8)
        self.assertTrue(all(item.k_tiles == 2 for item in instructions))
        self.assertEqual({(i.m0, i.n0) for i in instructions},
                         {(m, n) for m in (0, 16, 32, 48) for n in (0, 32)})

    def test_a_wide_matrix_needs_several_runs_per_row(self) -> None:
        # 16 tile columns of 16, GEMM_TILES_PER_INSTRUCTION per run.
        step = GEMM_TILES_PER_INSTRUCTION * 16
        instructions = emitters.gemm_instructions(16, 256, 16)
        self.assertEqual([(i.m0, i.n0) for i in instructions],
                         [(0, run * step) for run in range(256 // step)])

    def test_a_shape_without_an_edge_path_is_rejected(self) -> None:
        for shape in ((17, 16, 16), (16, 17, 16), (16, 16, 17)):
            with self.assertRaises(ValueError):
                emitters.gemm_instructions(*shape)

    def test_the_split_covers_every_run_once(self) -> None:
        # 64x64 is 4 tile rows x ceil(4 tiles / run) runs.
        runs = (64 // 16) * -(-(64 // 16) // GEMM_TILES_PER_INSTRUCTION)
        rows = emitters.gemm_queues(64, 64, 32, 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(len(row) for row in rows), runs)
        self.assertEqual(max(len(row) for row in rows)
                         - min(len(row) for row in rows), 1)


class EmitterClassTests(unittest.TestCase):
    """Every operator implements the same three things, so the split is shared."""

    def test_each_kind_has_a_generator_and_a_registry_entry(self) -> None:
        from dmk.model_types import ATTENTION, ELEMENTWISE, EMBEDDING, GEMM

        for kind in (GEMM, EMBEDDING, ATTENTION, ELEMENTWISE):
            self.assertIn(kind, emitters.EMITTERS)
        self.assertIs(emitters.emitter_for(GEMM, m=16, n=16, k=16).__class__,
                      emitters.GemmEmitter)

    def test_an_unknown_kind_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            emitters.emitter_for("matmul", m=16)
        self.assertIn("no emitter", str(caught.exception))

    def test_the_base_class_refuses_to_emit_instructions(self) -> None:
        # A subclass that forgets instructions() must fail loudly, not silently
        # produce an empty schedule.
        class Incomplete(emitters.OperatorEmitter):
            kind = "gemm"

        with self.assertRaises(NotImplementedError):
            Incomplete(1).instructions()
        with self.assertRaises(NotImplementedError):
            Incomplete(1).work()

    def test_the_split_is_shared_so_every_operator_behaves_the_same(self) -> None:
        runs = (64 // 16) * -(-(48 // 16) // GEMM_TILES_PER_INSTRUCTION)
        cases = [
            (emitters.GemmEmitter(64, 48, 32), runs),
            (emitters.EmbeddingEmitter(7, elem_per_row=4), 7),
            (emitters.AttentionEmitter(5, num_keys=64, dim=32), 5),
            (emitters.ElementwiseEmitter(2048 * 3 + 10), 4),
        ]
        for generator, expected in cases:
            with self.subTest(kind=generator.kind):
                instructions = generator.instructions()
                self.assertEqual(len(instructions), expected, generator)
                for cus in (1, 2, 3, 8):
                    rows = generator.queues(cus)
                    self.assertEqual(len(rows), cus)
                    self.assertEqual(sum(len(row) for row in rows), expected)
                    self.assertLessEqual(
                        max(len(row) for row in rows) - min(len(row) for row in rows), 1
                    )

    def test_each_generator_reports_its_own_work(self) -> None:
        from dmk.model_types import ATTENTION, ELEMENTWISE, EMBEDDING, GEMM

        self.assertEqual(emitters.GemmEmitter(32, 16, 16).work().kinds(), (GEMM,))
        self.assertEqual(
            emitters.EmbeddingEmitter(3, elem_per_row=2).work().kinds(), (EMBEDDING,)
        )
        self.assertEqual(
            emitters.AttentionEmitter(4, num_keys=8, dim=8).work().kinds(),
            (ATTENTION,),
        )
        self.assertEqual(
            emitters.ElementwiseEmitter(4096).work().kinds(), (ELEMENTWISE,)
        )

    def test_combined_work_adds_the_emitters_together(self) -> None:
        gemm = emitters.GemmEmitter(32, 32, 16)          # 2 rows x 1 run
        embedding = emitters.EmbeddingEmitter(3, elem_per_row=8)
        combined = emitters.combined_work([gemm, embedding])
        self.assertEqual(combined.slots(GEMM), 2)
        self.assertEqual(combined.slots("embedding"), 3)
        self.assertEqual(combined.gemm_k_tiles, 1)

    def test_a_zero_cu_split_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            emitters.GemmEmitter(16, 16, 16).queues(0)


class ModelScheduleTests(unittest.TestCase):
    def _model(self):
        return emitters.ModelSchedule([
            emitters.GemmEmitter(32, 32, 16),
            emitters.EmbeddingEmitter(5, elem_per_row=8),
            emitters.AttentionEmitter(3, num_keys=16, dim=8),
        ])

    def test_a_model_concatenates_its_operators_in_order(self) -> None:
        rows = self._model().rows(2)
        self.assertEqual(len(rows), 2)
        for row in rows:
            kinds = [instruction.kind() for instruction in row]
            self.assertEqual(kinds, sorted(kinds, key=["gemm", "embedding", "attention"].index))

    def test_every_operator_contributes_its_own_slots(self) -> None:
        work = self._model().work()
        self.assertEqual(work.slots("gemm"), 2)
        self.assertEqual(work.slots("embedding"), 5)
        self.assertEqual(work.slots("attention"), 3)

    def test_the_rows_hold_every_instruction_exactly_once(self) -> None:
        model = self._model()
        rows = model.rows(3)
        total = sum(len(row) for row in rows)
        expected = sum(len(g.instructions()) for g in model.emitters)
        self.assertEqual(total, expected)

    def test_a_duplicated_kind_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            emitters.ModelSchedule([
                emitters.GemmEmitter(16, 16, 16),
                emitters.GemmEmitter(16, 16, 16),
            ])

    def test_an_empty_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            emitters.ModelSchedule([])


class EmulatorTests(unittest.TestCase):
    def _globals(self, m, n, k, cus):
        globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=0, device="cpu")
        globals_.attach(emitters.gemm_queues(m, n, k, cus))
        return globals_

    def test_the_emulated_schedule_reproduces_the_product(self) -> None:
        globals_ = self._globals(64, 64, 32, 4)
        self.assertLess(mk.verify(globals_, emulated=True), 1e-4)

    def test_every_tile_is_written_exactly_once(self) -> None:
        globals_ = self._globals(64, 48, 32, 3)
        counts = python_vm.count_tiles(globals_.instructions, globals_.b.shape[1])
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
