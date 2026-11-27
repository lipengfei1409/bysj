"""Per-operator instruction generators -- one class per operator kind.

`Megakernels/megakernels/generators.py` emits a whole model's instruction
sequence; this tree needs the same thing one operator at a time, so each operator
gets a :class:`OperatorGenerator` subclass that knows three things:

* how much work it covers, in the unit its operator is addressed by
  (:meth:`OperatorGenerator.work`, which feeds `utils.plan_cus`),
* how to expand that work into instructions (:meth:`instructions`),
* nothing about splitting them across CUs -- that lives in the base class, so
  every operator is split the same way and only its own arithmetic differs.

Adding an operator therefore means writing one ``instructions()`` body, not
re-implementing the split:

    class EmbeddingGenerator(OperatorGenerator):
        kind = EMBEDDING

        def instructions(self):
            return [Embedding(src_row=r, dst_row=r, count=self.elem_per_row)
                    for r in range(self.rows)]

The module keeps thin functions (``gemm_instructions``, ``gemm_queues``) for the
call sites that only need one operator; they are one-liners over the classes.
"""

from __future__ import annotations

from .instructions import (Attention, Embedding, Elementwise, Gemm, Instruction)
from .model_types import (ATTENTION, ELEMENTWISE, EMBEDDING, GEMM, GEMM_TILE,
                          ELEMENTWISE_PAGE, CuTileConfig, Work)
from .utils import split_units


class OperatorGenerator:
    """Base: one operator's work, instructions, and per-CU split."""

    #: Which operator kind this generator emits (one of `model_types.OP_KINDS`).
    kind: str = ""

    def __init__(self, units: int):
        if not isinstance(units, int) or isinstance(units, bool):
            raise TypeError(f"units must be an int, got {type(units).__name__}")
        if units < 0:
            raise ValueError(f"units must be non-negative, got {units}")
        self._units = units

    # --- what the model needs ------------------------------------------------

    def units(self) -> int:
        """Work units this operator contributes, in its native unit."""
        return self._units

    def slots(self) -> int:
        """Addressable slots -- what the split and the tiling count.

        Defaults to the native unit; an operator whose instruction addresses more
        than one element per instruction overrides this (the elementwise op
        addresses pages, so its slots are pages).
        """
        return self._units

    def work(self) -> Work:
        """A `Work` carrying only this kind's total (see `combined_work`)."""
        raise NotImplementedError

    # --- what the runtime runs ----------------------------------------------

    def emit(self, tiles: CuTileConfig, start: int, cover: int) -> Instruction:
        """One instruction covering ``cover`` slots from absolute ``start``.

        This is the only method a new operator has to write: the split, the
        tiling and the per-CU bookkeeping are the base class's job.
        """
        raise NotImplementedError

    def slots_per_instruction(self, tiles: CuTileConfig) -> int:
        """How many slots one instruction covers under ``tiles`` (default 1)."""
        return 1

    def instructions(self) -> list[Instruction]:
        """The whole range as instructions, one CU's worth (used by tests/demos)."""
        tiles = self.default_tiles()
        step = self.slots_per_instruction(tiles)
        if step < 1:
            raise ValueError(
                f"{type(self).__name__} would emit {step} slots per instruction"
            )
        return [
            self.emit(tiles, start, min(step, self.slots() - start))
            for start in range(0, self.slots(), step)
        ]

    def default_tiles(self) -> CuTileConfig:
        """The tile configuration this operator uses when it owns everything."""
        from .utils import plan_cus

        return plan_cus(self.work(), 1)[0].tiles

    # --- the split, shared by every operator --------------------------------

    def queues(self, cus: int) -> list[list[Instruction]]:
        """Hand the operator's work out to ``cus`` rows.

        One code path for every operator: the work totals go through
        `utils.plan_cus` (which decides each CU's tile from its share), the
        instruction count comes from `utils.instruction_counts`, and each CU then
        asks its own :meth:`emit` for the instructions covering its share.
        Contiguous, so a CU never waits on another for its own range, and even,
        because a CU that owns one unit more than its neighbour costs less than a
        CU that owns nothing.
        """
        if not isinstance(cus, int) or isinstance(cus, bool) or cus < 1:
            raise ValueError(f"cus must be a positive int, got {cus!r}")
        from .utils import instruction_counts, plan_cus, split_units

        work = self.work()
        plans = plan_cus(work, cus)
        counts = instruction_counts(work, plans)
        shares = split_units(self.slots(), cus)

        rows: list[list[Instruction]] = [[] for _ in range(cus)]
        first = 0
        for plan in plans:
            step = plan.tiles.slots_per_instruction(self.kind)
            for index in range(counts[self.kind][plan.cu]):
                start = first + index * step
                cover = min(step, shares[plan.cu] - index * step)
                if cover > 0:
                    rows[plan.cu].append(self.emit(plan.tiles, start, cover))
            first += shares[plan.cu]
        return rows

    def __repr__(self) -> str:
        return f"{type(self).__name__}(units={self._units})"


class GemmGenerator(OperatorGenerator):
    """A dense ``C = A @ B``: one instruction per 16x16 output tile."""

    kind = GEMM

    def __init__(self, m: int, n: int, k: int):
        for name, value in (("m", m), ("n", n), ("k", k)):
            if value % GEMM_TILE:
                raise ValueError(
                    "the persistent GEMM tiles by 16 and has no edge path, so M, "
                    f"N and K must be multiples of 16 ({name}={value})"
                )
        super().__init__((m // GEMM_TILE) * (n // GEMM_TILE))
        self.m, self.n, self.k = m, n, k

    def work(self) -> Work:
        return Work(gemm_tiles=self.units(), gemm_k_tiles=self.k // GEMM_TILE)

    def emit(self, tiles: CuTileConfig, start: int, cover: int) -> Gemm:
        # ``start`` is a tile index in row-major order over the 16x16 grid.
        columns = self.n // GEMM_TILE
        del cover  # one GEMM instruction is one whole output tile
        return Gemm(m0=(start // columns) * GEMM_TILE,
                    n0=(start % columns) * GEMM_TILE,
                    k_tiles=self.k // GEMM_TILE)


class EmbeddingGenerator(OperatorGenerator):
    """A table gather: one instruction per destination row."""

    kind = EMBEDDING

    def __init__(self, rows: int, *, elem_per_row: int):
        if elem_per_row < 0:
            raise ValueError(f"elem_per_row must be non-negative, got {elem_per_row}")
        super().__init__(rows)
        self.elem_per_row = elem_per_row

    def work(self) -> Work:
        return Work(embedding_rows=self.units(), embedding_tile=1)

    def emit(self, tiles: CuTileConfig, start: int, cover: int) -> Embedding:
        # One instruction gathers one row: `start` is the row index.
        del tiles, cover
        return Embedding(src_row=start, dst_row=start, count=self.elem_per_row)


class AttentionGenerator(OperatorGenerator):
    """One instruction per query row: the key walk happens inside the op."""

    kind = ATTENTION

    def __init__(self, queries: int, *, num_keys: int, dim: int):
        if num_keys < 0 or dim < 1:
            raise ValueError(
                f"num_keys must be non-negative and dim positive, got "
                f"{num_keys}, {dim}"
            )
        super().__init__(queries)
        self.num_keys, self.dim = num_keys, dim

    def work(self) -> Work:
        return Work(attention_queries=self.units(), attention_tile=1)

    def emit(self, tiles: CuTileConfig, start: int, cover: int) -> Attention:
        # One instruction is one query row: the key walk happens inside the op.
        del tiles, cover
        return Attention(query_tile=start, num_keys=self.num_keys, dim=self.dim)


class ElementwiseGenerator(OperatorGenerator):
    """``y = scale * x + bias``: one instruction per page."""

    kind = ELEMENTWISE

    def __init__(self, elements: int, *, scale: float = 1.0, bias: float = 0.0):
        super().__init__(elements)
        self.scale, self.bias = scale, bias

    def work(self) -> Work:
        return Work(elementwise=self.units())

    def slots(self) -> int:
        # The elementwise op is addressed by page, so its slots are pages.
        return (self.units() + ELEMENTWISE_PAGE - 1) // ELEMENTWISE_PAGE

    def emit(self, tiles: CuTileConfig, start: int, cover: int) -> Elementwise:
        # One instruction is one page: `start` is the page index and the count is
        # what that page actually carries (the last one may be short).
        del tiles, cover
        remaining = self.units() - start * ELEMENTWISE_PAGE
        return Elementwise(tile=start, scale=self.scale, bias=self.bias,
                           count=max(0, min(ELEMENTWISE_PAGE, remaining)))


#: Kind name -> generator class, so a model description can name its operators.
GENERATORS = {
    GEMM: GemmGenerator,
    EMBEDDING: EmbeddingGenerator,
    ATTENTION: AttentionGenerator,
    ELEMENTWISE: ElementwiseGenerator,
}


def generator_for(kind: str, **shapes) -> OperatorGenerator:
    """Build the generator for ``kind`` (``GENERATORS`` is the registry)."""
    try:
        factory = GENERATORS[kind]
    except KeyError:
        raise ValueError(
            f"no generator for {kind!r}; known kinds: {sorted(GENERATORS)}"
        ) from None
    return factory(**shapes)


def combined_work(generators: list[OperatorGenerator]) -> Work:
    """Merge several generators' work into the totals `utils.plan_cus` divides."""
    totals = {
        GEMM: 0,
        EMBEDDING: 0,
        ATTENTION: 0,
        ELEMENTWISE: 0,
    }
    k_tiles = 4
    for generator in generators:
        work = generator.work()
        for kind in (GEMM, EMBEDDING, ATTENTION, ELEMENTWISE):
            totals[kind] += work.slots(kind)
        if generator.kind == GEMM:
            k_tiles = generator.work().gemm_k_tiles
    return Work(
        elementwise=totals[ELEMENTWISE] * ELEMENTWISE_PAGE,
        gemm_tiles=totals[GEMM],
        embedding_rows=totals[EMBEDDING],
        attention_queries=totals[ATTENTION],
        gemm_k_tiles=k_tiles,
    )


class ModelGenerator:
    """A model as a list of operators: one instruction stream for all of them.

    Each operator's `queues` is already per-CU, so assembling a model is a
    concatenation, in model order: CU ``i`` runs operator 1's share of its row,
    then operator 2's, and so on.  That is the order the runtime needs, because a
    CU walks its own row front to back.
    """

    def __init__(self, generators: list[OperatorGenerator]):
        if not generators:
            raise ValueError("a model needs at least one operator generator")
        kinds = [generator.kind for generator in generators]
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"each operator kind may appear once, got {kinds}")
        self.generators = list(generators)

    def work(self) -> Work:
        return combined_work(self.generators)

    def rows(self, cus: int) -> list[list[Instruction]]:
        """Per-CU rows: every operator's share of that CU, in model order."""
        rows: list[list[Instruction]] = [[] for _ in range(cus)]
        for generator in self.generators:
            for cu, row in enumerate(generator.queues(cus)):
                rows[cu].extend(row)
        return rows

    def describe(self) -> str:
        parts = ", ".join(
            f"{generator.kind}={generator.slots()}" for generator in self.generators
        )
        return f"{len(self.generators)} operators ({parts})"

    def __repr__(self) -> str:
        return f"ModelGenerator({self.describe()})"


# --- one-line conveniences for the single-operator call sites ---------------

def gemm_instructions(m: int, n: int, k: int) -> list[Gemm]:
    return GemmGenerator(m, n, k).instructions()


def gemm_queues(m: int, n: int, k: int, cus: int) -> list[list[Instruction]]:
    return GemmGenerator(m, n, k).queues(cus)


def gemm_work(m: int, n: int, k: int) -> Work:
    return GemmGenerator(m, n, k).work()


def embedding_queues(rows: int, *, elem_per_row: int, cus: int):
    return EmbeddingGenerator(rows, elem_per_row=elem_per_row).queues(cus)


def attention_queues(queries: int, *, num_keys: int, dim: int, cus: int):
    return AttentionGenerator(queries, num_keys=num_keys, dim=dim).queues(cus)


__all__ = [
    "GENERATORS",
    "AttentionGenerator",
    "ElementwiseGenerator",
    "EmbeddingGenerator",
    "GemmGenerator",
    "ModelGenerator",
    "OperatorGenerator",
    "attention_queues",
    "combined_work",
    "embedding_queues",
    "gemm_instructions",
    "gemm_queues",
    "gemm_work",
    "generator_for",
]
