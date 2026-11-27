"""Turn a model description into instructions (the reference's generators role).

`Megakernels/megakernels/generators.py` emits a model's instruction sequence from
its layers.  Ours has one model so far -- a dense GEMM -- and the generator does
the two things a schedule needs:

* :func:`gemm_instructions` -- one instruction per 16x16 output tile, in row-major
  order, with the k-steps of that shape;
* :func:`gemm_schedule` -- those instructions assigned to ``cus`` queues, each CU
  using the tile sizes `utils.plan_cus` gave it.

The split is by *tile*, not by instruction count: 100 tiles over 3 CUs is 34/33/33
tiles, and each CU issues exactly one instruction per tile it owns.  A CU that
owns none is left with an empty row, which the stream pads with NoOps.
"""

from __future__ import annotations

from .model_types import GEMM, GEMM_TILE, Work
from .instructions import Instruction, Gemm, NoOp
from .scheduler import DagNode, linear_chain, pad_with_noops
from .utils import plan_cus, split_units


def gemm_instructions(m: int, n: int, k: int) -> list[Gemm]:
    """One ``Gemm`` per output tile of ``C = A @ B``, in row-major tile order."""
    if m % GEMM_TILE or n % GEMM_TILE or k % GEMM_TILE:
        raise ValueError(
            "the persistent GEMM tiles by 16 and has no edge path, so M, N and K "
            f"must be multiples of 16 (got {m}x{n}x{k})"
        )
    k_tiles = k // GEMM_TILE
    tiles_m = m // GEMM_TILE
    tiles_n = n // GEMM_TILE
    return [
        Gemm(m0=row * GEMM_TILE, n0=column * GEMM_TILE, k_tiles=k_tiles)
        for row in range(tiles_m)
        for column in range(tiles_n)
    ]


def gemm_work(m: int, n: int, k: int) -> Work:
    """The per-kind work totals of a GEMM of this shape."""
    return Work(gemm_tiles=(m // GEMM_TILE) * (n // GEMM_TILE),
                gemm_k_tiles=k // GEMM_TILE)


def gemm_queues(m: int, n: int, k: int, cus: int) -> list[list[Instruction]]:
    """Split a GEMM's tiles across ``cus`` rows, row-major within each CU."""
    instructions = gemm_instructions(m, n, k)
    rows: list[list[Instruction]] = [[] for _ in range(cus)]
    shares = split_units(len(instructions), cus)
    first = 0
    for cu, share in enumerate(shares):
        rows[cu].extend(instructions[first:first + share])
        first += share
    return rows


def gemm_schedule(m: int, n: int, k: int, cus: int) -> tuple[list[list[Instruction]], list[DagNode]]:
    """Queues plus the DAG they came from, for callers that want to inspect it."""
    instructions = gemm_instructions(m, n, k)
    nodes, _ = linear_chain(instructions)
    rows = gemm_queues(m, n, k, cus)
    return rows, nodes


def pad(rows: list[list[Instruction]]) -> list[list[Instruction]]:
    """Equalise row lengths with NoOps (the runtime needs a rectangle)."""
    return pad_with_noops(rows)


__all__ = [
    "gemm_instructions",
    "gemm_queues",
    "gemm_schedule",
    "gemm_work",
    "pad",
]
