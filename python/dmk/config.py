"""Per-CU configuration for the persistent megakernel.

One CU (a DCU workgroup, one block) walks its own instruction row, so the tile an
instruction covers is a *per-CU* decision: a CU that owns a smaller share should
be given a tile that matches it, not the same tile count as everybody else plus a
long idle tail.

Everything is counted in **slots**, the unit each operator is addressed by:

============ =====================================================
elementwise  one 8 KiB page (2048 float32) -- the instruction
             carries a page index, so a page is the smallest unit
gemm         one 16x16 output tile
embedding    one table row
attention    one query row
============ =====================================================

`plan_cus` splits the slots of every kind across the CUs and hands each CU the
tile sizes it should use; `shard.py` turns the plans into instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

ELEMENTWISE = "elementwise"
GEMM = "gemm"
EMBEDDING = "embedding"
ATTENTION = "attention"

OP_KINDS = (ELEMENTWISE, GEMM, EMBEDDING, ATTENTION)

#: Elements one elementwise instruction covers: one 8 KiB page of float32.
#: Fixed by the operator's encoding, so this tile cannot vary per CU.
ELEMENTWISE_PAGE = 2048


@dataclass(frozen=True)
class TilePlan:
    """The tile sizes one CU uses, per operator kind.

    A zero means "this CU does no work of that kind": `shard.py` then emits no
    instructions for it, which is how a CU with a smaller share stays short
    instead of being padded up to the longest row.
    """

    elementwise: int = 0      # elements per elementwise instruction (a page)
    gemm_m: int = 0           # output tile rows
    gemm_n: int = 0           # output tile columns
    gemm_k_tiles: int = 0     # k-steps one instruction walks
    embedding: int = 0        # table rows per instruction
    attention_query: int = 0  # query rows per instruction

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.elementwise and self.elementwise != ELEMENTWISE_PAGE:
            raise ValueError(
                "the elementwise tile is fixed by its page "
                f"({ELEMENTWISE_PAGE}), got {self.elementwise}"
            )

    def slots_per_instruction(self, kind: str) -> int:
        """How many addressable slots one instruction of ``kind`` consumes."""
        if kind == ELEMENTWISE:
            return 1  # one page
        if kind == GEMM:
            return 1  # one output tile
        if kind == EMBEDDING:
            return self.embedding
        if kind == ATTENTION:
            return self.attention_query
        raise ValueError(f"unknown operator kind {kind!r}")


@dataclass(frozen=True)
class CuPlan:
    """One CU's assignment: its tile sizes (its slot shares live in `Work`)."""

    cu: int
    tiles: TilePlan


@dataclass
class Work:
    """What the model needs computed, in each operator's native amount."""

    elementwise: int = 0        # elements
    gemm_tiles: int = 0         # output tiles
    embedding_rows: int = 0     # table rows
    attention_queries: int = 0  # query rows
    gemm_tile: tuple[int, int] = (16, 16)
    gemm_k_tiles: int = 4
    embedding_tile: int = 2048
    attention_tile: int = 16

    def slots(self, kind: str) -> int:
        """Addressable units of ``kind`` (see the module docstring)."""
        if kind == ELEMENTWISE:
            return (self.elementwise + ELEMENTWISE_PAGE - 1) // ELEMENTWISE_PAGE
        if kind == GEMM:
            return self.gemm_tiles
        if kind == EMBEDDING:
            return self.embedding_rows
        if kind == ATTENTION:
            return self.attention_queries
        raise ValueError(f"unknown operator kind {kind!r}")

    def kinds(self) -> Iterable[str]:
        return tuple(kind for kind in OP_KINDS if self.slots(kind) > 0)


def split(total: int, parts: int) -> list[int]:
    """Divide ``total`` into ``parts`` shares summing exactly to ``total``.

    The remainder goes to the earliest CUs, one unit each, so every CU that can
    get work does and the sum stays exact.
    """
    if parts < 1:
        raise ValueError(f"parts must be positive, got {parts}")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    base, extra = divmod(total, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def _tile_for(default: int, share: int) -> int:
    """Tile size for a CU that owns ``share`` slots.

    Capped at the share (so a small share is one instruction) and floored at one
    (so a non-empty share is never rounded away).
    """
    if share <= 0:
        return 0
    return max(1, min(default, share))


def plan_cus(work: Work, cus: int) -> list[CuPlan]:
    """One plan per CU: the tile sizes that fit the share it will be given."""
    if not isinstance(cus, int) or cus < 1:
        raise ValueError(f"cus must be a positive int, got {cus!r}")
    gemm_m, gemm_n = work.gemm_tile
    shares = {kind: split(work.slots(kind), cus) for kind in work.kinds()}

    plans = []
    for cu in range(cus):
        own = {kind: shares.get(kind, [0] * cus)[cu] for kind in work.kinds()}
        plans.append(
            CuPlan(
                cu=cu,
                tiles=TilePlan(
                    elementwise=ELEMENTWISE_PAGE if own.get(ELEMENTWISE) else 0,
                    gemm_m=gemm_m if own.get(GEMM) else 0,
                    gemm_n=gemm_n if own.get(GEMM) else 0,
                    gemm_k_tiles=work.gemm_k_tiles if own.get(GEMM) else 0,
                    embedding=_tile_for(work.embedding_tile, own.get(EMBEDDING, 0)),
                    attention_query=_tile_for(work.attention_tile,
                                              own.get(ATTENTION, 0)),
                ),
            )
        )
    return plans


def shares_of(work: Work, kind: str, cus: int) -> list[int]:
    """Slot shares of one kind, one per CU."""
    return split(work.slots(kind), cus)


def instruction_counts(work: Work, plans: list[CuPlan]) -> dict[str, list[int]]:
    """Instructions each CU issues per kind: ``ceil(share / slots_per_inst)``."""
    counts: dict[str, list[int]] = {}
    for kind in work.kinds():
        shares = shares_of(work, kind, len(plans))
        counts[kind] = [
            -(-shares[plan.cu] // plan.tiles.slots_per_instruction(kind))
            if shares[plan.cu] and plan.tiles.slots_per_instruction(kind)
            else 0
            for plan in plans
        ]
    return counts


__all__ = [
    "ATTENTION",
    "CuPlan",
    "ELEMENTWISE",
    "ELEMENTWISE_PAGE",
    "EMBEDDING",
    "GEMM",
    "OP_KINDS",
    "TilePlan",
    "Work",
    "instruction_counts",
    "plan_cus",
    "shares_of",
    "split",
]
