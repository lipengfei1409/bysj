"""Host-side builder for the DCU megakernel instruction stream.

The runtime consumes one flat int32 tensor::

    instructions : (blocks, num_instructions, 32) int32

Every block (CU) walks its own row of ``num_instructions`` 32-word instructions;
``word[0]`` is the opcode and the rest is op-specific. This module builds that
tensor from per-op instruction objects, so a caller describes a model as a small
sequence of op invocations instead of hand-writing int32 arrays::

    stream = InstructionStream(blocks=3)
    for tile in range(3):
        stream.emit(tile, attention(query_tile=tile, num_keys=4096, dim=64))
    instructions = stream.tensor()        # (3, 1, 32) int32

Op encodings mirror the C++ ``dmk::mk`` operators exactly:

    NoOp (0)          all zero
    Elementwise (1)   [1] tile  [2] scale bits  [3] bias bits  [4] count
    Gemm (2)          [1] m0    [2] n0          [3] k_tiles
    Embedding (3)     [1] src row  [2] dst row  [3] count
    Attention (5)     [1] query_tile  [2] num_keys  [3] dim

Floats are bit-cast exactly like the C++ side (``__float_as_int``): a little
endian signed 32-bit reinterpretation, so ``0.5`` becomes ``0x3F000000``.
"""
from __future__ import annotations

import struct

OPCODE_NOOP = 0
OPCODE_ELEMENTWISE = 1
OPCODE_GEMM = 2
OPCODE_EMBEDDING = 3
OPCODE_ATTENTION = 5

INSTRUCTION_WIDTH = 32

KNOWN_OPCODES = frozenset(
    (OPCODE_NOOP, OPCODE_ELEMENTWISE, OPCODE_GEMM, OPCODE_EMBEDDING, OPCODE_ATTENTION)
)

_INT32_MIN = -(2 ** 31)
_INT32_MAX = 2 ** 31 - 1


def _check_int32(value: int, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{what} must be an int, got {type(value).__name__}")
    if not _INT32_MIN <= value <= _INT32_MAX:
        raise ValueError(f"{what} does not fit a signed 32-bit word: {value}")
    return value


def float_bits(value: float) -> int:
    """Bit-cast a float the way ``__float_as_int`` does."""
    packed = struct.pack("<f", float(value))
    return struct.unpack("<i", packed)[0]


class Instruction:
    """One 32-word instruction; ``words[0]`` is the opcode."""

    __slots__ = ("_words",)

    def __init__(self, words):
        words = tuple(words)
        if len(words) != INSTRUCTION_WIDTH:
            raise ValueError(
                f"instruction must have {INSTRUCTION_WIDTH} words, got {len(words)}"
            )
        self._words = tuple(
            _check_int32(word, f"instruction word {index}")
            for index, word in enumerate(words)
        )

    @property
    def words(self) -> tuple[int, ...]:
        return self._words

    @property
    def opcode(self) -> int:
        return self._words[0]

    def __iter__(self):
        return iter(self._words)

    def __eq__(self, other):
        return isinstance(other, Instruction) and self._words == other._words

    def __repr__(self) -> str:
        return f"Instruction(opcode={self.opcode}, words={self._words!r})"


def _blank(opcode: int, values: dict[int, int]) -> Instruction:
    words = [0] * INSTRUCTION_WIDTH
    words[0] = opcode
    for index, value in values.items():
        words[index] = value
    return Instruction(words)


def noop() -> Instruction:
    """Do nothing; the runtime uses it to pad a block to a uniform length."""
    return _blank(OPCODE_NOOP, {})


def elementwise(*, tile: int, count: int, scale: float = 1.0,
                bias: float = 0.0) -> Instruction:
    """``y = scale * x + bias`` over one page-sized tile of ``count`` elements."""
    return _blank(
        OPCODE_ELEMENTWISE,
        {1: tile, 2: float_bits(scale), 3: float_bits(bias), 4: count},
    )


def gemm(*, m0: int, n0: int, k_tiles: int) -> Instruction:
    """One 16x16 output tile over ``k_tiles`` k-steps, starting at (m0, n0)."""
    return _blank(OPCODE_GEMM, {1: m0, 2: n0, 3: k_tiles})


def embedding(*, src_row: int, dst_row: int, count: int) -> Instruction:
    """Copy ``count`` elements from one table row to another."""
    return _blank(OPCODE_EMBEDDING, {1: src_row, 2: dst_row, 3: count})


def attention(*, query_tile: int, num_keys: int, dim: int) -> Instruction:
    """One query tile over all keys: the op walks the key tiles itself."""
    return _blank(OPCODE_ATTENTION, {1: query_tile, 2: num_keys, 3: dim})


class InstructionStream:
    """Per-block instruction rows, padded to one uniform length.

    ``emit`` appends to a block; ``tensor``/``numpy`` pad every shorter block
    with ``NoOp`` and return the rectangular ``(blocks, length, 32)`` array the
    kernel's ``g.instructions`` expects.
    """

    def __init__(self, blocks: int, length: int | None = None):
        if not isinstance(blocks, int) or blocks < 1:
            raise ValueError(f"blocks must be a positive int, got {blocks!r}")
        if length is not None and (not isinstance(length, int) or length < 1):
            raise ValueError(f"length must be a positive int, got {length!r}")
        self.blocks = blocks
        self.fixed_length = length
        self._rows: list[list[Instruction]] = [[] for _ in range(blocks)]

    def __len__(self) -> int:
        return self.length

    @property
    def length(self) -> int:
        """Current length: the fixed one, or the longest block."""
        longest = max((len(row) for row in self._rows), default=0)
        if self.fixed_length is not None:
            return self.fixed_length
        return longest

    def emit(self, block: int, instruction: Instruction) -> "InstructionStream":
        if not isinstance(instruction, Instruction):
            raise TypeError("emit expects an Instruction")
        if not 0 <= block < self.blocks:
            raise IndexError(f"block {block} outside [0, {self.blocks})")
        row = self._rows[block]
        if self.fixed_length is not None and len(row) >= self.fixed_length:
            raise ValueError(
                f"block {block} already has {len(row)} instructions "
                f"(fixed length {self.fixed_length})"
            )
        row.append(instruction)
        return self

    def emit_all(self, instruction: Instruction) -> "InstructionStream":
        """Append the same instruction to every block."""
        for block in range(self.blocks):
            self.emit(block, instruction)
        return self

    def pad(self) -> "InstructionStream":
        """Fill every block with ``NoOp`` up to the current length."""
        target = self.length
        for row in self._rows:
            while len(row) < target:
                row.append(noop())
        return self

    def row(self, block: int, *, padded: bool = True) -> list[Instruction]:
        if not 0 <= block < self.blocks:
            raise IndexError(f"block {block} outside [0, {self.blocks})")
        row = list(self._rows[block])
        if padded:
            while len(row) < self.length:
                row.append(noop())
        return row

    def validate(self) -> None:
        """Reject an empty or unknown-opcode stream before it reaches the device."""
        if self.length == 0:
            raise ValueError("instruction stream is empty")
        for block, row in enumerate(self._rows):
            for index, instruction in enumerate(row):
                if instruction.opcode not in KNOWN_OPCODES:
                    raise ValueError(
                        f"block {block} instruction {index} has unknown opcode "
                        f"{instruction.opcode}"
                    )

    def numpy(self):
        """``(blocks, length, 32)`` int32 array, padded with NoOp."""
        import numpy as np

        self.validate()
        rows = np.zeros((self.blocks, self.length, INSTRUCTION_WIDTH), dtype=np.int32)
        for block in range(self.blocks):
            for index, instruction in enumerate(self.row(block)):
                rows[block, index, :] = instruction.words
        return rows

    def tensor(self, *, device=None, dtype=None):
        """``(blocks, length, 32)`` torch tensor, padded with NoOp."""
        import torch

        tensor = torch.from_numpy(self.numpy())
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor


def attention_stream(query_tiles: int, *, blocks: int, num_keys: int, dim: int,
                     length: int | None = None) -> InstructionStream:
    """One attention instruction per query tile, round-robin over the blocks."""
    if not isinstance(query_tiles, int) or query_tiles < 1:
        raise ValueError(f"query_tiles must be a positive int, got {query_tiles!r}")
    stream = InstructionStream(blocks=blocks, length=length)
    for tile in range(query_tiles):
        stream.emit(tile % blocks, attention(query_tile=tile, num_keys=num_keys, dim=dim))
    return stream


__all__ = [
    "INSTRUCTION_WIDTH",
    "KNOWN_OPCODES",
    "OPCODE_ATTENTION",
    "OPCODE_EMBEDDING",
    "OPCODE_ELEMENTWISE",
    "OPCODE_GEMM",
    "OPCODE_NOOP",
    "Instruction",
    "InstructionStream",
    "attention",
    "attention_stream",
    "elementwise",
    "embedding",
    "float_bits",
    "gemm",
    "noop",
]
