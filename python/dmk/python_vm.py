"""A Python emulator of the instruction stream (the reference's ``python_vm.py``).

`Megakernels/megakernels/python_vm.py` executes a schedule in Python so a model's
instruction stream can be checked without a GPU.  Ours does the same for the one
operator the runtime has: it decodes a ``(cus, length, 32)`` int32 schedule and
applies each instruction to a dense accumulator.

That is worth having because it tests the *schedule*, not the kernel: a tile that
no CU was given, a tile given twice, or a warp-level encoding mistake all show up
here, on a host, with no DCU in the loop.  What it cannot test is anything the
kernel does with pages, semaphores or the DUMMA unit -- those only exist on the
device, which is why `tests/csrc/accuracy/test_persistent_gemm.py` still runs.
"""

from __future__ import annotations

import torch

from kernels import mk_stream

from .model_types import GEMM_TILE, GEMM_TILES_PER_INSTRUCTION


def decode_row(row) -> list[dict]:
    """Decode one CU's row into plain dicts, unknown opcodes rejected."""
    decoded = []
    for words in row:
        opcode = int(words[0])
        if opcode not in mk_stream.KNOWN_OPCODES:
            raise ValueError(f"unknown opcode {opcode}")
        if opcode == mk_stream.OPCODE_NOOP:
            continue
        if opcode == mk_stream.OPCODE_GEMM:
            decoded.append(
                {"kind": "gemm", "m0": int(words[1]), "n0": int(words[2]),
                 "k_tiles": int(words[3])}
            )
            continue
        raise ValueError(f"the emulator has no implementation for opcode {opcode}")
    return decoded


def tiles_of(instruction: dict, n: int) -> list[tuple[int, int]]:
    """The output tiles one GEMM instruction covers, given the matrix width.

    An instruction covers a run of ``GEMM_TILES_PER_INSTRUCTION`` tiles along N --
    one per consumer wave.  The last run of a row is short, and the kernel simply
    skips the waves whose column is past the N edge, which is what this mirrors.
    """
    tiles = []
    for wave in range(GEMM_TILES_PER_INSTRUCTION):
        column = instruction["n0"] + wave * GEMM_TILE
        if column < n:
            tiles.append((instruction["m0"], column))
    return tiles


def emulate_gemm(instructions, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run a decoded GEMM schedule against ``a`` and ``b``; returns ``C``.

    Every instruction writes one 16x16 tile of ``C``; a tile that no CU covers
    stays zero, so a caller comparing against ``a @ b`` catches a schedule that
    leaves work out.  A tile covered twice is *added* twice, which the caller also
    sees -- the emulator does not try to hide a bad split.
    """
    if instructions.dim() != 3 or instructions.shape[2] != mk_stream.INSTRUCTION_WIDTH:
        raise ValueError("expected a (cus, length, 32) instruction tensor")
    c = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32, device="cpu")
    rows = instructions.to("cpu").tolist()
    for row in rows:
        for instruction in decode_row(row):
            for m0, n0 in tiles_of(instruction, b.shape[1]):
                tile = torch.zeros(GEMM_TILE, GEMM_TILE, dtype=torch.float32)
                for k in range(instruction["k_tiles"]):
                    kk = k * GEMM_TILE
                    tile += (a[m0:m0 + GEMM_TILE, kk:kk + GEMM_TILE]
                             @ b[kk:kk + GEMM_TILE, n0:n0 + GEMM_TILE]).to("cpu")
                c[m0:m0 + GEMM_TILE, n0:n0 + GEMM_TILE] += tile
    return c


def count_tiles(instructions, n: int) -> dict[tuple[int, int], int]:
    """How many times each output tile is written -- the split's audit.

    ``n`` is the matrix width, because an instruction covers a run of tiles and
    the last run of a row is short.
    """
    counts: dict[tuple[int, int], int] = {}
    for row in instructions.to("cpu").tolist():
        for instruction in decode_row(row):
            for key in tiles_of(instruction, n):
                counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = ["count_tiles", "decode_row", "emulate_gemm", "tiles_of"]
