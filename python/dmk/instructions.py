"""Instruction objects, one per operator -- the host's instruction set.

Mirrors `Megakernels/megakernels/instructions.py`: an instruction is a small
dataclass whose :meth:`serialize` produces the 32 int32 words the runtime reads,
with word 0 the opcode.  The numeric opcodes come from `ops::config` (mirrored in
`kernels.mk_stream`), so this module only decides *what* an instruction is, not
how the words are laid out.

Every subclass stays a plain dataclass: `serialize` walks the fields in
declaration order, which keeps a new field impossible to add without it landing
in the words in the same place on both sides of the wire.

Cost
----
:meth:`cost` is the scheduler's estimate of how long one instruction takes, in
"element-ops".  The cost model is deliberately trivial (work units, weighted by
kind) -- it only has to rank instructions consistently, and a real model would
replace it with measured kernel times.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from kernels import mk_stream



@dataclass
class Instruction:
    """Base: subclasses add their fields and the opcode."""

    @classmethod
    def opcode(cls) -> int:
        raise NotImplementedError

    @classmethod
    def kind(cls) -> str:
        raise NotImplementedError

    def serialize(self) -> list[int]:
        """The 32 words the runtime reads, ``words[0]`` = opcode."""
        words = [self.opcode()]
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, int) and not isinstance(value, bool):
                words.append(value)
            elif isinstance(value, float):
                words.append(mk_stream.float_bits(value))
            elif isinstance(value, (tuple, list)):
                words.append(len(value))
                words.extend(int(item) for item in value)
            elif value is None:
                words.append(0)
            else:
                raise ValueError(f"cannot serialize {field.name}={value!r}")
        if len(words) > mk_stream.INSTRUCTION_WIDTH:
            raise ValueError(
                f"{type(self).__name__} needs {len(words)} words, the runtime has "
                f"{mk_stream.INSTRUCTION_WIDTH}"
            )
        words.extend([0] * (mk_stream.INSTRUCTION_WIDTH - len(words)))
        return words

    def cost(self) -> float:
        """Scheduler cost, in FLOPs-equivalent: one cost unit ~ one FLOP."""
        return 1.0

    def units(self) -> int:
        """Work units this instruction covers (for reporting and cost)."""
        return 1


@dataclass
class NoOp(Instruction):
    """Padding: a zeroed instruction word is a no-op."""

    @classmethod
    def opcode(cls) -> int:
        return mk_stream.OPCODE_NOOP

    @classmethod
    def kind(cls) -> str:
        return "noop"

    def serialize(self) -> list[int]:
        return [0] * mk_stream.INSTRUCTION_WIDTH

    def cost(self) -> float:
        return 0.0

    def units(self) -> int:
        return 0


@dataclass
class Elementwise(Instruction):
    """``y = scale * x + bias`` over one page.

    Field order follows the wire: word 1 tile, 2 scale, 3 bias, 4 count -- the
    same layout `kernels.mk_stream.elementwise` writes.  Reordering these fields
    silently changes the encoding, which is why a test compares both encoders
    word for word.
    """

    tile: int        # page index
    scale: float = 1.0
    bias: float = 0.0
    count: int = 0   # elements this instruction covers (<= 2048)

    @classmethod
    def opcode(cls) -> int:
        return mk_stream.OPCODE_ELEMENTWISE

    @classmethod
    def kind(cls) -> str:
        return "elementwise"

    def cost(self) -> float:
        return float(self.count)

    def units(self) -> int:
        return self.count


@dataclass
class Gemm(Instruction):
    """One 16x16 output tile of ``C = A @ B`` over ``k_tiles`` k-steps."""

    m0: int
    n0: int
    k_tiles: int

    @classmethod
    def opcode(cls) -> int:
        return mk_stream.OPCODE_GEMM

    @classmethod
    def kind(cls) -> str:
        return "gemm"

    def cost(self) -> float:
        # 2 * m * n * k per tile, with m = n = 16 and k = 16 per k-step.
        return float(2 * 16 * 16 * 16 * self.k_tiles)

    def units(self) -> int:
        return 1


@dataclass
class Embedding(Instruction):
    """One table row, gathered."""

    src_row: int
    dst_row: int
    count: int

    @classmethod
    def opcode(cls) -> int:
        return mk_stream.OPCODE_EMBEDDING

    @classmethod
    def kind(cls) -> str:
        return "embedding"

    def cost(self) -> float:
        return float(self.count)

    def units(self) -> int:
        return self.count


@dataclass
class Attention(Instruction):
    """One query tile over ``num_keys`` keys."""

    query_tile: int
    num_keys: int
    dim: int

    @classmethod
    def opcode(cls) -> int:
        return mk_stream.OPCODE_ATTENTION

    @classmethod
    def kind(cls) -> str:
        return "attention"

    def cost(self) -> float:
        # Score plus weighted sum: 2 * 2 * keys * dim per query row.
        return float(4 * self.num_keys * self.dim)

    def units(self) -> int:
        return 1


#: Every instruction class, by kind name -- what a flow may contain.
BY_KIND = {
    instruction.kind(): instruction
    for instruction in (NoOp, Elementwise, Gemm, Embedding, Attention)
}


class InstructionStream:
    """Per-CU rows of instructions, padded to one length.

    The counterpart of the reference's ``list[list[Instruction]]`` plus the
    rectangular array the runtime needs; `kernels.mk_stream.InstructionStream`
    does the padding, this keeps the objects around so a caller can inspect them.
    """

    def __init__(self, rows: list[list[Instruction]]):
        if not rows or len(rows) < 1:
            raise ValueError("a stream needs at least one CU row")
        self.rows = [list(row) for row in rows]

    @property
    def blocks(self) -> int:
        return len(self.rows)

    @property
    def length(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    def padded(self) -> list[list[Instruction]]:
        width = self.length
        return [
            row + [NoOp() for _ in range(width - len(row))] for row in self.rows
        ]

    def words(self) -> list[list[list[int]]]:
        """``[cu][instruction][32]`` in the runtime's layout."""
        return [[instruction.serialize() for instruction in row] for row in self.padded()]

    def tensor(self, *, device=None):
        """The ``(blocks, length, 32)`` int32 array the kernel reads."""
        import torch

        return torch.tensor(self.words(), dtype=torch.int32, device=device)

    def describe(self) -> str:
        lines = [f"{self.blocks} CUs, at most {self.length} instructions each"]
        for cu, row in enumerate(self.rows):
            counts: dict[str, int] = {}
            for instruction in row:
                counts[instruction.kind()] = counts.get(instruction.kind(), 0) + 1
            issued = " ".join(f"{k}={v}" for k, v in counts.items() if k != "noop")
            lines.append(f"  cu {cu}: {issued or 'idle'}")
        return "\n".join(lines)


__all__ = [
    "Attention",
    "BY_KIND",
    "Embedding",
    "Elementwise",
    "Gemm",
    "Instruction",
    "InstructionStream",
    "NoOp",
]
