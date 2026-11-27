"""Host tests for the instruction set and the CU scheduler (`python/dmk`).

Ported from the reference Megakernels layout: `instructions.py` owns the words,
`scheduler.py` owns the assignment.  Both are pure host code, so the invariants
worth pinning are about *equivalence and order*, not speed.
"""

from __future__ import annotations

import unittest

from dmk import instructions as I
from dmk import scheduler as S
from kernels import mk_stream


class InstructionTests(unittest.TestCase):
    def test_the_gemm_words_match_the_encoder(self) -> None:
        # Two encoders exist (the dataclass here, mk_stream for the C++ table);
        # they must produce the same 32 words for the same instruction.
        ours = I.Gemm(m0=0, n0=4, k_tiles=2).serialize()
        theirs = list(mk_stream.gemm(m0=0, n0=4, k_tiles=2).words)
        self.assertEqual(ours, theirs)

    def test_every_instruction_fills_exactly_the_width(self) -> None:
        for instruction in (
            I.NoOp(),
            I.Elementwise(tile=3, count=128),
            I.Gemm(m0=16, n0=32, k_tiles=4),
            I.Embedding(src_row=1, dst_row=2, count=7),
            I.Attention(query_tile=0, num_keys=64, dim=32),
        ):
            words = instruction.serialize()
            self.assertEqual(len(words), mk_stream.INSTRUCTION_WIDTH, type(instruction))
            self.assertEqual(words[0], instruction.opcode())
            self.assertIn(words[0], mk_stream.KNOWN_OPCODES)

    def test_a_float_field_is_bit_cast_like_the_runtime(self) -> None:
        words = I.Elementwise(tile=0, count=1, scale=0.5, bias=-1.0).serialize()
        self.assertEqual(words[2], mk_stream.float_bits(0.5))
        self.assertEqual(words[3], mk_stream.float_bits(-1.0))

    def test_cost_ranks_a_gemm_above_a_page_of_elementwise_work(self) -> None:
        self.assertGreater(I.Gemm(0, 0, 4).cost(), I.Elementwise(0, 2048).cost())
        self.assertEqual(I.NoOp().cost(), 0.0)

    def test_the_stream_pads_short_rows_with_noops(self) -> None:
        stream = I.InstructionStream([[I.Gemm(0, 0, 1), I.Gemm(0, 16, 1)], [I.Gemm(0, 32, 1)]])
        self.assertEqual(stream.length, 2)
        self.assertEqual(stream.blocks, 2)
        padded = stream.padded()
        self.assertEqual(len(padded[1]), 2)
        self.assertIsInstance(padded[1][1], I.NoOp)
        words = stream.words()
        self.assertEqual(len(words), 2)
        self.assertEqual(len(words[0][0]), mk_stream.INSTRUCTION_WIDTH)
        self.assertEqual(words[1][1][0], mk_stream.OPCODE_NOOP)

    def test_an_empty_stream_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            I.InstructionStream([])


class SchedulerTests(unittest.TestCase):
    def test_a_chain_keeps_its_order_on_one_cu(self) -> None:
        nodes, end = S.linear_chain([I.Gemm(0, 0, 1), I.Gemm(0, 16, 1), I.Gemm(0, 32, 1)])
        queues = S.assign_dag_to_cus(nodes, 1)
        self.assertEqual(len(queues), 1)
        self.assertEqual([i.n0 for i in queues[0]], [0, 16, 32])

    def test_every_instruction_is_scheduled_exactly_once(self) -> None:
        flow = [I.Gemm(0, 16 * index, 2) for index in range(8)]
        nodes, _ = S.linear_chain(flow)
        queues = S.assign_dag_to_cus(nodes, 3)
        scheduled = [instruction for queue in queues for instruction in queue]
        self.assertEqual(len(scheduled), len(flow))
        self.assertEqual(sorted(i.n0 for i in scheduled), [16 * i for i in range(8)])

    def test_no_cu_runs_an_instruction_before_its_dependency(self) -> None:
        # A chain must stay a chain: element k of the chain may only appear once
        # every earlier element has been placed somewhere.
        flow = [I.Elementwise(tile=index, count=2048) for index in range(6)]
        nodes, _ = S.linear_chain(flow)
        queues = S.assign_dag_to_cus(nodes, 2)
        position = {}
        for cu, queue in enumerate(queues):
            for slot, instruction in enumerate(queue):
                position[instruction.tile] = (cu, slot)
        # Global order: the scheduler places them in dependency order, so the
        # assigned finish times are non-decreasing along the chain.
        ends = [node.end_time for node in nodes]
        self.assertEqual(ends, sorted(ends))

    def test_round_robin_deals_in_order(self) -> None:
        flow = [I.Gemm(0, 16 * index, 1) for index in range(5)]
        queues = S.round_robin_assign(flow, 2)
        self.assertEqual([len(queue) for queue in queues], [3, 2])
        self.assertEqual(queues[0][0].n0, 0)
        self.assertEqual(queues[1][0].n0, 16)

    def test_a_cycle_is_reported_not_ignored(self) -> None:
        first = S.DagNode(instruction=I.NoOp())
        second = S.DagNode(instruction=I.NoOp())
        first.dependencies = [second]
        second.dependencies = [first]
        with self.assertRaises(ValueError) as caught:
            S.assign_dag_to_cus([first, second], 2)
        self.assertIn("cycle", str(caught.exception))

    def test_padding_equalises_the_rows(self) -> None:
        queues = [[I.Gemm(0, 0, 1), I.Gemm(0, 16, 1)], [I.Gemm(0, 32, 1)]]
        padded = S.pad_with_noops(queues)
        self.assertEqual([len(row) for row in padded], [2, 2])
        self.assertIsInstance(padded[1][1], I.NoOp)

    def test_a_zero_cu_count_is_rejected(self) -> None:
        nodes, _ = S.linear_chain([I.NoOp()])
        with self.assertRaises(ValueError):
            S.assign_dag_to_cus(nodes, 0)
        with self.assertRaises(ValueError):
            S.round_robin_assign([I.NoOp()], 0)


if __name__ == "__main__":
    unittest.main()
