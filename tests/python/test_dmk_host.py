"""Host tests for the per-CU megakernel driver (`python/dmk`).

Nothing here needs a DCU: `config` decides tile sizes and shares, `shard` turns
them into instruction rows, and `main` is exercised through its dry-run path.
The three are tested together because the invariant that matters -- every slot of
work is covered exactly once across the CUs -- only holds across all three.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest

from dmk import config, main
from dmk.shard import DataFlow, OpCall, shard
from kernels import mk_stream


def _flow() -> DataFlow:
    return DataFlow(
        [
            OpCall(config.ELEMENTWISE, 40000),
            OpCall(config.GEMM, 100, {"k_tiles": 4}),
            OpCall(config.EMBEDDING, 300),
            OpCall(config.ATTENTION, 40, {"dim": 64}),
        ]
    )


class SplitTests(unittest.TestCase):
    def test_the_shares_sum_to_the_total(self) -> None:
        for total in (0, 1, 7, 100, 2047, 2048):
            for parts in (1, 2, 3, 8):
                shares = config.split(total, parts)
                self.assertEqual(sum(shares), total, (total, parts))
                self.assertEqual(len(shares), parts)
                self.assertLessEqual(max(shares) - min(shares), 1)

    def test_the_remainder_goes_to_the_earliest_cus(self) -> None:
        self.assertEqual(config.split(10, 3), [4, 3, 3])

    def test_bad_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            config.split(5, 0)
        with self.assertRaises(ValueError):
            config.split(-1, 2)


class SlotsTests(unittest.TestCase):
    def test_elementwise_is_counted_in_pages(self) -> None:
        work = config.Work(elementwise=config.ELEMENTWISE_PAGE * 3 + 1)
        self.assertEqual(work.slots(config.ELEMENTWISE), 4)

    def test_the_other_kinds_are_already_addressable(self) -> None:
        work = config.Work(gemm_tiles=7, embedding_rows=9, attention_queries=11)
        self.assertEqual(work.slots(config.GEMM), 7)
        self.assertEqual(work.slots(config.EMBEDDING), 9)
        self.assertEqual(work.slots(config.ATTENTION), 11)

    def test_a_kind_with_no_work_is_not_a_kind(self) -> None:
        self.assertEqual(config.Work(gemm_tiles=3).kinds(), (config.GEMM,))


class PlanTests(unittest.TestCase):
    def test_every_cu_gets_a_tile_that_fits_its_share(self) -> None:
        work = config.Work(elementwise=config.ELEMENTWISE_PAGE * 10, gemm_tiles=9,
                           embedding_rows=7, attention_queries=5,
                           embedding_tile=4, attention_tile=2)
        plans = config.plan_cus(work, 4)
        self.assertEqual(len(plans), 4)
        for kind in work.kinds():
            shares = config.shares_of(work, kind, 4)
            for plan in plans:
                tile = plan.tiles.slots_per_instruction(kind)
                if shares[plan.cu] == 0:
                    self.assertEqual(tile, 0, (kind, plan.cu))
                else:
                    self.assertGreaterEqual(tile, 1, (kind, plan.cu))
                    self.assertLessEqual(tile, shares[plan.cu] + 1, (kind, plan.cu))

    def test_the_elementwise_tile_is_fixed_by_its_page(self) -> None:
        work = config.Work(elementwise=config.ELEMENTWISE_PAGE * 4)
        plan = config.plan_cus(work, 1)[0]
        self.assertEqual(plan.tiles.elementwise, config.ELEMENTWISE_PAGE)
        with self.assertRaises(ValueError):
            config.TilePlan(elementwise=17)

    def test_a_small_share_shrinks_the_tile_to_one_instruction(self) -> None:
        # 3 rows over 4 CUs: the first three own one row each, the fourth none,
        # and the tile follows the share so each owner issues exactly one.
        work = config.Work(embedding_rows=3, embedding_tile=2048)
        plans = config.plan_cus(work, 4)
        counts = config.instruction_counts(work, plans)
        self.assertEqual(counts[config.EMBEDDING], [1, 1, 1, 0])
        self.assertEqual([p.tiles.embedding for p in plans], [1, 1, 1, 0])

    def test_the_idle_cu_gets_zero_tiles(self) -> None:
        work = config.Work(gemm_tiles=2)
        plans = config.plan_cus(work, 4)
        self.assertEqual([p.tiles.gemm_m for p in plans], [16, 16, 0, 0])


class ShardTests(unittest.TestCase):
    def test_one_row_per_cu_and_a_rectangular_array(self) -> None:
        queues = shard(_flow(), 3)
        self.assertEqual(queues.blocks, 3)
        tensor = queues.tensor()
        self.assertEqual(tuple(tensor.shape), (3, queues.length, 32))
        self.assertTrue(str(tensor.dtype).endswith("int32"), tensor.dtype)

    def test_every_instruction_carries_a_known_opcode(self) -> None:
        queues = shard(_flow(), 2)
        opcodes = {row[0] for row in queues.tensor().reshape(-1, 32).tolist()}
        self.assertTrue(opcodes <= mk_stream.KNOWN_OPCODES, opcodes)

    def test_the_shares_cover_the_work_exactly(self) -> None:
        flow = _flow()
        queues = shard(flow, 4)
        work = flow.work()
        for kind in work.kinds():
            shares = config.shares_of(work, kind, 4)
            self.assertEqual(sum(shares), work.slots(kind), kind)
            # Every CU issues ceil(share / step), so no CU leaves slots unserved.
            for plan in queues.plans:
                step = plan.tiles.slots_per_instruction(kind)
                if shares[plan.cu] == 0:
                    self.assertEqual(queues.counts[kind][plan.cu], 0, (kind, plan.cu))
                else:
                    self.assertEqual(
                        queues.counts[kind][plan.cu],
                        -(-shares[plan.cu] // step),
                        (kind, plan.cu),
                    )

    def test_idle_cus_are_padded_with_noops(self) -> None:
        # 2 gemm tiles over 4 CUs: two CUs work, two are padded to the same length.
        queues = shard(DataFlow([OpCall(config.GEMM, 2, {"k_tiles": 1})]), 4)
        blocks = queues.tensor().tolist()  # [cu][instruction][word]
        self.assertEqual(queues.length, 1)
        self.assertEqual(len(blocks), 4)
        for instruction in blocks[2] + blocks[3]:
            self.assertEqual(instruction[0], mk_stream.OPCODE_NOOP)

    def test_the_description_names_every_cu(self) -> None:
        text = shard(_flow(), 3).describe()
        for cu in range(3):
            self.assertIn(f"cu {cu}", text)

    def test_an_empty_flow_is_rejected_rather_than_sharded(self) -> None:
        with self.assertRaises(ValueError) as caught:
            shard(DataFlow([]), 2)
        self.assertIn("no work", str(caught.exception))


class MainTests(unittest.TestCase):
    def test_the_reference_flow_matches_the_work_it_reports(self) -> None:
        work = main.build_flow().work()
        self.assertEqual(work.slots(config.GEMM), 181_838)
        self.assertEqual(work.slots(config.EMBEDDING), 3_000)

    def test_dry_run_succeeds_and_prints_the_queues(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(main.main(["--cus", "3", "--dry-run"]), 0)
        text = buffer.getvalue()
        self.assertIn("3 CUs", text)
        self.assertIn("cu 2", text)

    def test_json_output_is_parseable(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main.main(["--cus", "2", "--dry-run", "--json"])
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["cus"], 2)
        self.assertEqual(len(payload["tiles"]), 2)
        self.assertIn(config.GEMM, payload["counts"])

    def test_a_zero_cu_count_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                main.main(["--cus", "0", "--dry-run"])

    def test_launch_without_the_runtime_is_reported_not_faked(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = main.main(["--cus", "2"])
        self.assertEqual(code, 2)
        self.assertIn("launch unavailable", stderr.getvalue())
        self.assertIn("megakernel runtime is not built", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
