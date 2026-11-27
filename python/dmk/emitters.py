"""Per-operator instruction emitters -- one class per operator kind.

`Megakernels/megakernels/emitters.py` emits a whole model's instruction
sequence; this tree needs the same thing one operator at a time, so each operator
gets a :class:`OperatorEmitter` subclass that knows three things:

* how much work it covers, in the unit its operator is addressed by
  (:meth:`OperatorEmitter.work`, which feeds `utils.plan_cus`),
* how to expand that work into instructions (:meth:`instructions`),
* nothing about splitting them across CUs -- that lives in the base class, so
  every operator is split the same way and only its own arithmetic differs.

Why a class rather than one function per operator: the split is the part that is
identical for every operator and easy to get subtly wrong, while the operator
itself only knows how to turn one range into one instruction.  With the split in
the base class the elementwise emitter cannot disagree with the GEMM emitter
about how a share is tiled, and a test can loop over every kind asserting the
same invariants -- which is only expressible if they share an interface.  (The
reference implementation keeps this in one model-level function; a class per
operator is our addition, because our schedule has more than one operator.)

Adding an operator therefore means writing one ``emit()`` body, not re-implementing
the split:

    class EmbeddingEmitter(OperatorEmitter):
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
                          ELEMENTWISE_PAGE, GEMM_TILES_PER_INSTRUCTION, CuTileConfig,
                          Work)
from .utils import split_units


class OperatorEmitter:
    """Base: one operator's work, instructions, and per-CU split."""

    #: Which operator kind this emitter covers (one of `model_types.OP_KINDS`).
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


class GemmEmitter(OperatorEmitter):
    """A dense ``C = A @ B``: one instruction per run of output tiles.

    A run is ``GEMM_TILES_PER_INSTRUCTION`` tiles along N -- one per consumer
    wave, because a DUMMA wave computes a whole 16x16 tile and cannot split one.
    The last run of a row is short when N is not a multiple of the run length;
    the kernel skips the waves past the edge.
    """

    kind = GEMM

    def __init__(self, m: int, n: int, k: int):
        for name, value in (("m", m), ("n", n), ("k", k)):
            if value % GEMM_TILE:
                raise ValueError(
                    "the persistent GEMM tiles by 16 and has no edge path, so M, "
                    f"N and K must be multiples of 16 ({name}={value})"
                )
        self.tiles_n = n // GEMM_TILE
        self.runs_n = -(-self.tiles_n // GEMM_TILES_PER_INSTRUCTION)
        super().__init__((m // GEMM_TILE) * self.runs_n)
        self.m, self.n, self.k = m, n, k

    def work(self) -> Work:
        return Work(gemm_runs=self.units(), gemm_k_tiles=self.k // GEMM_TILE)

    def emit(self, tiles: CuTileConfig, start: int, cover: int) -> Gemm:
        # ``start`` indexes a run of tiles in row-major order over the run grid;
        # ``n0`` is the run's first tile column.
        del cover  # one instruction is one whole run
        run = start % self.runs_n
        return Gemm(m0=(start // self.runs_n) * GEMM_TILE,
                    n0=run * GEMM_TILES_PER_INSTRUCTION * GEMM_TILE,
                    k_tiles=self.k // GEMM_TILE)


class EmbeddingEmitter(OperatorEmitter):
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


class AttentionEmitter(OperatorEmitter):
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


class ElementwiseEmitter(OperatorEmitter):
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


#: Kind name -> emitter class, so a model description can name its operators.
EMITTERS = {
    GEMM: GemmEmitter,
    EMBEDDING: EmbeddingEmitter,
    ATTENTION: AttentionEmitter,
    ELEMENTWISE: ElementwiseEmitter,
}


def emitter_for(kind: str, **shapes) -> OperatorEmitter:
    """Build the emitter for ``kind`` (``EMITTERS`` is the registry)."""
    try:
        factory = EMITTERS[kind]
    except KeyError:
        raise ValueError(
            f"no emitter for {kind!r}; known kinds: {sorted(EMITTERS)}"
        ) from None
    return factory(**shapes)


def combined_work(emitters: list[OperatorEmitter]) -> Work:
    """Merge several emitters' work into the totals `utils.plan_cus` divides."""
    totals = {
        GEMM: 0,
        EMBEDDING: 0,
        ATTENTION: 0,
        ELEMENTWISE: 0,
    }
    k_tiles = 4
    for emitter in emitters:
        work = emitter.work()
        for kind in (GEMM, EMBEDDING, ATTENTION, ELEMENTWISE):
            totals[kind] += work.slots(kind)
        if emitter.kind == GEMM:
            k_tiles = emitter.work().gemm_k_tiles
    return Work(
        elementwise=totals[ELEMENTWISE] * ELEMENTWISE_PAGE,
        gemm_runs=totals[GEMM],
        embedding_rows=totals[EMBEDDING],
        attention_queries=totals[ATTENTION],
        gemm_k_tiles=k_tiles,
    )


class ModelSchedule:
    """A model as a list of operators: one instruction stream for all of them.

    Each operator's `queues` is already per-CU, so assembling a model is a
    concatenation, in model order: CU ``i`` runs operator 1's share of its row,
    then operator 2's, and so on.  That is the order the runtime needs, because a
    CU walks its own row front to back.
    """

    def __init__(self, emitters: list[OperatorEmitter]):
        if not emitters:
            raise ValueError("a model needs at least one operator generator")
        kinds = [emitter.kind for emitter in emitters]
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"each operator kind may appear once, got {kinds}")
        self.emitters = list(emitters)

    def work(self) -> Work:
        return combined_work(self.emitters)

    def rows(self, cus: int) -> list[list[Instruction]]:
        """Per-CU rows: every operator's share of that CU, in model order."""
        rows: list[list[Instruction]] = [[] for _ in range(cus)]
        for emitter in self.emitters:
            for cu, row in enumerate(emitter.queues(cus)):
                rows[cu].extend(row)
        return rows

    def describe(self) -> str:
        parts = ", ".join(
            f"{emitter.kind}={emitter.slots()}" for emitter in self.emitters
        )
        return f"{len(self.emitters)} operators ({parts})"

    def __repr__(self) -> str:
        return f"ModelSchedule({self.describe()})"


# --- one-line conveniences for the single-operator call sites ---------------

def gemm_instructions(m: int, n: int, k: int) -> list[Gemm]:
    return GemmEmitter(m, n, k).instructions()


def gemm_queues(m: int, n: int, k: int, cus: int) -> list[list[Instruction]]:
    return GemmEmitter(m, n, k).queues(cus)


def gemm_work(m: int, n: int, k: int) -> Work:
    return GemmEmitter(m, n, k).work()


def embedding_queues(rows: int, *, elem_per_row: int, cus: int):
    return EmbeddingEmitter(rows, elem_per_row=elem_per_row).queues(cus)


def attention_queues(queries: int, *, num_keys: int, dim: int, cus: int):
    return AttentionEmitter(queries, num_keys=num_keys, dim=dim).queues(cus)


__all__ = [
    "EMITTERS",
    "AttentionEmitter",
    "ElementwiseEmitter",
    "EmbeddingEmitter",
    "GemmEmitter",
    "ModelSchedule",
    "OperatorEmitter",
    "attention_queues",
    "combined_work",
    "embedding_queues",
    "gemm_instructions",
    "gemm_queues",
    "gemm_work",
    "emitter_for",
]
