"""Small helpers: how many CUs, and how work is divided between them."""

from __future__ import annotations

import os

from .model_types import (ATTENTION, ELEMENTWISE, CuPlan, CuTileConfig, EMBEDDING,
                          GEMM, Work)


def get_cu_count(device=None) -> int:
    """CUs a launch fans out to.

    Overridable with ``DMK_CU_COUNT`` because the number that matters is the one
    the *kernel* is launched with, not what the device reports -- a test may want
    three CUs on a 120-CU part.
    """
    override = os.environ.get("DMK_CU_COUNT")
    if override:
        count = int(override)
        if count < 1:
            raise ValueError(f"DMK_CU_COUNT must be positive, got {count}")
        return count
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:  # noqa: BLE001 - a host build has no device to ask
        pass
    return 1


def split_units(total: int, parts: int) -> list[int]:
    """Divide ``total`` into ``parts`` shares summing exactly to ``total``.

    The remainder goes to the earliest CUs, one unit each -- so every share is
    within one unit of every other, and the sum stays exact.
    """
    if parts < 1:
        raise ValueError(f"parts must be positive, got {parts}")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    base, extra = divmod(total, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def tile_for(default: int, share: int) -> int:
    """Tile size for a CU owning ``share`` slots: capped at it, floored at one."""
    if share <= 0:
        return 0
    return max(1, min(default, share))


def plan_cus(work: Work, cus: int) -> list[CuPlan]:
    """One plan per CU: the tile sizes that fit the share it will be given."""
    if cus < 1:
        raise ValueError(f"cus must be positive, got {cus}")
    shares = {kind: split_units(work.slots(kind), cus) for kind in work.kinds()}

    plans = []
    for cu in range(cus):
        own = {kind: shares.get(kind, [0] * cus)[cu] for kind in work.kinds()}
        plans.append(
            CuPlan(
                cu=cu,
                tiles=CuTileConfig(
                    elementwise=work_pages(own.get(ELEMENTWISE, 0)),
                    gemm_m=16 if own.get(GEMM) else 0,
                    gemm_n=16 if own.get(GEMM) else 0,
                    gemm_k_tiles=work.gemm_k_tiles if own.get(GEMM) else 0,
                    embedding=tile_for(work.embedding_tile, own.get(EMBEDDING, 0)),
                    attention_query=tile_for(work.attention_tile,
                                             own.get(ATTENTION, 0)),
                ),
            )
        )
    return plans


def work_pages(share: int) -> int:
    """The elementwise tile: one page when the CU owns any of it, else zero."""
    from .model_types import ELEMENTWISE_PAGE

    return ELEMENTWISE_PAGE if share else 0


def instruction_counts(work: Work, plans: list[CuPlan]) -> dict[str, list[int]]:
    """Instructions each CU issues per kind: ``ceil(share / slots_per_inst)``."""
    counts: dict[str, list[int]] = {}
    for kind in work.kinds():
        shares = split_units(work.slots(kind), len(plans))
        counts[kind] = [
            -(-shares[plan.cu] // plan.tiles.slots_per_instruction(kind))
            if shares[plan.cu] and plan.tiles.slots_per_instruction(kind)
            else 0
            for plan in plans
        ]
    return counts


__all__ = [
    "get_cu_count",
    "instruction_counts",
    "plan_cus",
    "split_units",
    "tile_for",
    "work_pages",
]
