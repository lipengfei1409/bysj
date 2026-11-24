"""Host tests for the Python megakernel instruction-stream builder.

The C++ runtime consumes a flat (blocks, num_instructions, 32) int32 tensor;
these tests pin the encodings to the C++ operators' word layouts, the padding
rule and the per-block row semantics. No DCU is needed.
"""
from __future__ import annotations

import unittest

from kernels import mk_stream as mk


class InstructionEncodingTests(unittest.TestCase):
    def test_noop_is_all_zero(self):
        self.assertEqual(mk.noop().words, (0,) * 32)

    def test_elementwise_layout_matches_the_cpp_op(self):
        words = mk.elementwise(tile=3, count=100, scale=0.5, bias=-2.0).words
        self.assertEqual(words[0], mk.OPCODE_ELEMENTWISE)
        self.assertEqual(words[1], 3)
        self.assertEqual(words[2], mk.float_bits(0.5))
        self.assertEqual(words[3], mk.float_bits(-2.0))
        self.assertEqual(words[4], 100)
        self.assertEqual(words[5:], (0,) * 27)

    def test_float_bits_match_the_cpp_reinterpretation(self):
        self.assertEqual(mk.float_bits(0.5), 0x3F000000)
        self.assertEqual(mk.float_bits(-2.0), -0x40000000)
        self.assertEqual(mk.float_bits(1.0), 0x3F800000)

    def test_gemm_layout_matches_the_cpp_op(self):
        words = mk.gemm(m0=16, n0=32, k_tiles=4).words
        self.assertEqual(words[:4], (mk.OPCODE_GEMM, 16, 32, 4))
        self.assertEqual(words[4:], (0,) * 28)

    def test_embedding_layout_matches_the_cpp_op(self):
        words = mk.embedding(src_row=2, dst_row=1, count=15).words
        self.assertEqual(words[:4], (mk.OPCODE_EMBEDDING, 2, 1, 15))

    def test_attention_layout_matches_the_cpp_op(self):
        words = mk.attention(query_tile=1, num_keys=4096, dim=64).words
        self.assertEqual(words[:4], (mk.OPCODE_ATTENTION, 1, 4096, 64))

    def test_rejects_a_bad_instruction(self):
        with self.assertRaises(ValueError):
            mk.Instruction([0] * 31)
        with self.assertRaises(ValueError):
            mk.Instruction([0] * 31 + [2 ** 31])
        with self.assertRaises(TypeError):
            mk.Instruction([0] * 31 + [1.5])


class InstructionStreamTests(unittest.TestCase):
    def test_tensor_is_padded_with_noop_per_block(self):
        stream = mk.InstructionStream(blocks=2)
        stream.emit(0, mk.attention(query_tile=0, num_keys=8, dim=64))
        stream.emit(0, mk.attention(query_tile=2, num_keys=8, dim=64))
        stream.emit(1, mk.attention(query_tile=1, num_keys=8, dim=64))
        rows = stream.numpy()
        self.assertEqual(rows.shape, (2, 2, 32))
        self.assertEqual(int(rows[0, 0, 1]), 0)
        self.assertEqual(int(rows[0, 1, 1]), 2)
        self.assertEqual(int(rows[1, 0, 1]), 1)
        self.assertEqual(int(rows[1, 1].sum()), 0)  # NoOp padding
        self.assertEqual(stream.length, 2)

    def test_attention_stream_round_robins_the_query_tiles(self):
        stream = mk.attention_stream(5, blocks=2, num_keys=16, dim=32)
        rows = stream.numpy()
        self.assertEqual(rows.shape, (2, 3, 32))
        self.assertEqual([int(rows[0, i, 1]) for i in range(3)], [0, 2, 4])
        self.assertEqual([int(rows[1, i, 1]) for i in range(2)], [1, 3])
        self.assertEqual(int(rows[1, 2].sum()), 0)

    def test_matches_the_attention_cpp_test_encoding(self):
        # tests/csrc/test_dmk_attention.cpp builds one instruction per block:
        # [0]=5, [1]=block, [2]=keys, [3]=dim.
        blocks, keys, dim = 4, 64, 16
        rows = mk.attention_stream(blocks, blocks=blocks, num_keys=keys, dim=dim).numpy()
        for block in range(blocks):
            self.assertEqual(int(rows[block, 0, 0]), 5)
            self.assertEqual(int(rows[block, 0, 1]), block)
            self.assertEqual(int(rows[block, 0, 2]), keys)
            self.assertEqual(int(rows[block, 0, 3]), dim)

    def test_fixed_length_is_enforced(self):
        stream = mk.InstructionStream(blocks=1, length=1)
        stream.emit(0, mk.noop())
        with self.assertRaises(ValueError):
            stream.emit(0, mk.noop())

    def test_block_index_is_checked(self):
        stream = mk.InstructionStream(blocks=2)
        with self.assertRaises(IndexError):
            stream.emit(2, mk.noop())
        with self.assertRaises(IndexError):
            stream.row(-1)

    def test_empty_and_unknown_streams_are_rejected(self):
        with self.assertRaises(ValueError):
            mk.InstructionStream(blocks=1).numpy()
        stream = mk.InstructionStream(blocks=1)
        stream.emit(0, mk.Instruction([99] + [0] * 31))
        with self.assertRaises(ValueError):
            stream.numpy()

    def test_emit_all_reaches_every_block(self):
        stream = mk.InstructionStream(blocks=3)
        stream.emit_all(mk.noop())
        rows = stream.numpy()
        self.assertEqual(rows.shape, (3, 1, 32))
        self.assertEqual(int(rows.sum()), 0)

    def test_tensor_wraps_numpy_with_int32(self):
        torch = __import__("torch")
        stream = mk.attention_stream(2, blocks=2, num_keys=8, dim=8)
        tensor = stream.tensor()
        self.assertEqual(tuple(tensor.shape), (2, 1, 32))
        self.assertEqual(tensor.dtype, torch.int32)
        self.assertEqual(int(tensor[1, 0, 1]), 1)


if __name__ == "__main__":
    unittest.main()
