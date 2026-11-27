"""Types the persistent host side is written in (the reference's role, on CUs).

`Megakernels/megakernels/model_types.py` holds the device/tensor types a model
schedule is expressed with.  Ours is much smaller -- one operator so far -- and
names the DCU's execution unit a **CU** (a workgroup, one block) where the
reference says SM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

#: Where a launch's tensors live.  (Not `torch.device | str`: the annotation is
#: evaluated at runtime here, and `|` needs Python 3.10 while this tree targets
#: 3.9 as well.)
DeviceType = Union[torch.device, str]

#: Operator kinds this host side knows.  The numeric opcodes are the C++ table's
#: (`ops::config`), mirrored in `kernels.mk_stream`.
ELEMENTWISE = "elementwise"
GEMM = "gemm"
EMBEDDING = "embedding"
ATTENTION = "attention"

OP_KINDS = (ELEMENTWISE, GEMM, EMBEDDING, ATTENTION)

#: Elements one elementwise instruction covers: one 8 KiB page of float32,
#: fixed by the operator's encoding (the instruction carries a page index).
ELEMENTWISE_PAGE = 2048

#: Output tile the persistent GEMM walks, in rows and columns.
GEMM_TILE = 16

#: Consumer waves per block, i.e. output tiles one GEMM instruction covers along
#: N.  Mirrors `config::kConsumerWaves` in `csrc/ops/config.h`; a host test reads
#: that header and fails if the two drift.
GEMM_TILES_PER_INSTRUCTION = 2


@dataclass(frozen=True)
class Work:
    """What a model needs computed, in each operator's native amount."""

    elementwise: int = 0        # elements
    gemm_runs: int = 0         # runs of GEMM_TILES_PER_INSTRUCTION 16x16 tiles
    embedding_rows: int = 0     # table rows
    attention_queries: int = 0  # query rows
    gemm_k_tiles: int = 4       # k-steps per GEMM instruction
    embedding_tile: int = 2048
    attention_tile: int = 16

    def slots(self, kind: str) -> int:
        """Addressable units of ``kind``: pages, tiles, rows, query rows."""
        if kind == ELEMENTWISE:
            return (self.elementwise + ELEMENTWISE_PAGE - 1) // ELEMENTWISE_PAGE
        if kind == GEMM:
            return self.gemm_runs
        if kind == EMBEDDING:
            return self.embedding_rows
        if kind == ATTENTION:
            return self.attention_queries
        raise ValueError(f"unknown operator kind {kind!r}")

    def kinds(self) -> tuple[str, ...]:
        return tuple(kind for kind in OP_KINDS if self.slots(kind) > 0)


@dataclass(frozen=True)
class CuTileConfig:
    """The tile sizes one CU uses -- ``config.py``'s job, kept as a value type."""

    elementwise: int = 0
    gemm_m: int = 0
    gemm_n: int = 0
    gemm_k_tiles: int = 0
    embedding: int = 0
    attention_query: int = 0

    def slots_per_instruction(self, kind: str) -> int:
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
    """One CU's assignment: its tile configuration."""

    cu: int
    tiles: CuTileConfig


__all__ = [
    "ATTENTION",
    "CuPlan",
    "CuTileConfig",
    "DeviceType",
    "ELEMENTWISE",
    "ELEMENTWISE_PAGE",
    "EMBEDDING",
    "GEMM",
    "GEMM_TILE",
    "OP_KINDS",
    "Work",
]
