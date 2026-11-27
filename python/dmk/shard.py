"""Split a compute graph into one instruction queue per CU.

    DataFlow (operator calls) + per-CU plans -> Queues (one row per CU)

The split is by the unit each operator is *addressed* by (see `config`): pages
for the elementwise op, output tiles for the GEMM, rows for the embedding, query
rows for attention.  For every kind:

* the kind's slots are divided into one share per CU (`config.split`),
* a CU issues ``ceil(share / slots_per_instruction)`` instructions covering its
  share in order,
* each instruction carries the absolute slot index it starts at, so the CU needs
  to know nothing about what the other CUs took.

Nothing here touches a device: :meth:`Queues.tensor` is the int32
``(blocks, length, 32)`` array the runtime consumes, and :meth:`Queues.describe`
is what ``main --dry-run`` prints.  Keeping the split host-side means the same
code path is inspected, unit tested, and handed to a DCU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config, instructions


@dataclass
class OpCall:
    """One operator in the data flow: what kind, and how much work.

    ``units`` is in the operator's native amount (elements for elementwise,
    output tiles for the GEMM, rows for the embedding, query rows for attention).
    ``extra`` carries arguments the encoder needs that are not a property of the
    split, such as the GEMM's k-steps or the attention head dimension.
    """

    kind: str
    units: int
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in config.OP_KINDS:
            raise ValueError(f"unknown operator kind {self.kind!r}")
        if not isinstance(self.units, int) or isinstance(self.units, bool):
            raise TypeError(f"units must be an int, got {type(self.units).__name__}")
        if self.units < 0:
            raise ValueError(f"units must be non-negative, got {self.units}")


@dataclass
class DataFlow:
    """What the model wants computed, as a flat list of operator calls."""

    calls: list[OpCall]

    def work(self, *, gemm_tile: tuple[int, int] = (16, 16), gemm_k_tiles: int = 4,
             embedding_tile: int = 2048, attention_tile: int = 16) -> config.Work:
        """Fold the calls into the totals `config.plan_cus` divides."""
        totals = {kind: 0 for kind in config.OP_KINDS}
        for call in self.calls:
            totals[call.kind] += call.units
        k_tiles = next(
            (call.extra["k_tiles"] for call in self.calls if call.kind == config.GEMM),
            gemm_k_tiles,
        )
        dim = next(
            (call.extra.get("dim") for call in self.calls
             if call.kind == config.ATTENTION),
            None,
        )
        if dim is not None:
            attention_tile = min(attention_tile, dim)
        return config.Work(
            elementwise=totals[config.ELEMENTWISE],
            gemm_tiles=totals[config.GEMM],
            embedding_rows=totals[config.EMBEDDING],
            attention_queries=totals[config.ATTENTION],
            gemm_tile=gemm_tile,
            gemm_k_tiles=k_tiles,
            embedding_tile=embedding_tile,
            attention_tile=attention_tile,
        )


@dataclass
class Queues:
    """One instruction row per CU, plus what the split decided."""

    stream: instructions.InstructionStream
    plans: list[config.CuPlan]
    counts: dict[str, list[int]]
    work: config.Work

    @property
    def blocks(self) -> int:
        return self.stream.blocks

    @property
    def length(self) -> int:
        return self.stream.length

    def tensor(self, *, device=None):
        """The ``(blocks, length, 32)`` int32 array the runtime consumes."""
        return self.stream.tensor(device=device)

    def describe(self) -> str:
        return self.stream.describe()

    def _describe_legacy(self) -> str:
        lines = [f"{self.blocks} CUs, at most {self.length} instructions each"]
        for plan in self.plans:
            issued = " ".join(
                f"{kind}={self.counts[kind][plan.cu]}"
                for kind in self.counts
                if self.counts[kind][plan.cu]
            )
            tiles = ", ".join(
                f"{name}={value}"
                for name, value in (
                    ("elementwise", plan.tiles.elementwise),
                    ("gemm", f"{plan.tiles.gemm_m}x{plan.tiles.gemm_n}"),
                    ("embedding", plan.tiles.embedding),
                    ("attention", plan.tiles.attention_query),
                )
                if isinstance(value, int) and value > 0
                or (isinstance(value, str) and value != "0x0")
            )
            lines.append(
                f"  cu {plan.cu}: {issued or 'idle'}"
                + (f"   tiles {tiles}" if tiles else "")
            )
        return "\n".join(lines)


def shard(flow: DataFlow, cus: int, **work_options) -> Queues:
    """Split ``flow`` into ``cus`` instruction queues."""
    if not isinstance(flow, DataFlow):
        raise TypeError("shard expects a DataFlow")
    work = flow.work(**work_options)
    if not work.kinds():
        raise ValueError("the data flow covers no work: nothing to shard")
    plans = config.plan_cus(work, cus)
    counts = config.instruction_counts(work, plans)

    rows: list[list[instructions.Instruction]] = [[] for _ in range(cus)]
    for kind in work.kinds():
        shares = config.shares_of(work, kind, cus)
        first = 0
        for plan in plans:
            share = shares[plan.cu]
            step = plan.tiles.slots_per_instruction(kind)
            if share and step:
                for index in range(counts[kind][plan.cu]):
                    start = first + index * step
                    cover = min(step, share - index * step)
                    rows[plan.cu].append(
                        _instruction(kind, plan, work, start, cover)
                    )
            first += share
    stream = instructions.InstructionStream(rows)
    return Queues(stream=stream, plans=plans, counts=counts, work=work)


def _instruction(kind: str, plan: config.CuPlan, work: config.Work, start: int,
                 cover: int) -> instructions.Instruction:
    """One instruction covering ``cover`` slots from absolute ``start``."""
    if kind == config.ELEMENTWISE:
        return instructions.Elementwise(tile=start, count=cover)
    if kind == config.GEMM:
        return instructions.Gemm(m0=0, n0=start, k_tiles=work.gemm_k_tiles)
    if kind == config.EMBEDDING:
        return instructions.Embedding(src_row=start, dst_row=start, count=cover)
    if kind == config.ATTENTION:
        return instructions.Attention(query_tile=start, num_keys=cover,
                                      dim=plan.tiles.attention_query)
    raise ValueError(f"unknown operator kind {kind!r}")


__all__ = ["DataFlow", "OpCall", "Queues", "shard"]
