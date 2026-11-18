"""Step 0 of the DCU Triton tuning plan: is Triton usable on this device?

The tuning plan's first decision is whether the SeZM Triton kernels can run on
the target DCU at all, and -- if they can -- whether ``tl.dot`` reaches the
matrix core or falls back to scalar FMAs.  Both answers come from the compiled
artifact, not from timing, so this script is deliberately tiny:

1. Print the tuning-relevant device facts (wave size, block thread cap, LDS
   budget, architecture name).
2. Compile and run one pointwise kernel and one ``tl.dot`` kernel, checking
   each against a PyTorch reference.
3. Report, per kernel, the LDS bytes and register count Triton allocated plus
   the instruction / IR signals that identify the precision path.

Usage::

    source /opt/dtk/env.sh
    TRITON_ALWAYS_COMPILE=1 python tests/deepmd/profile/triton_smoke.py
    python tests/deepmd/profile/triton_smoke.py --json /tmp/triton_smoke.json

A non-zero exit code means Triton cannot produce a correct kernel here, which
is the gate that switches the week to plan B.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

try:
    import triton
    import triton.language as tl

    TRITON_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001 - report the import failure verbatim
    triton = None
    tl = None
    TRITON_IMPORT_ERROR = error

# Instruction / IR markers that identify how a kernel was lowered.  The AMD
# set is what a DCU produces; the NVIDIA set is kept so the same script can
# validate a reference machine running the identical Triton version.
_AMD_MARKERS = (
    "mfma",
    "v_mfma",
    "buffer_load",
    "global_load",
    "ds_write",
    "ds_read",
    "v_fma_f32",
    "v_mac_f32",
    "v_pk_fma_f32",
)
_NVIDIA_MARKERS = ("mma.sync", "wgmma", "ldmatrix", "cp.async", "fma.rn.f32")


def _device_facts(device: torch.device) -> dict:
    """Return the device properties the tile/config choices depend on."""
    properties = torch.cuda.get_device_properties(device)
    facts = {
        "name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": None if torch.version.hip else torch.version.cuda,
        "triton": getattr(triton, "__version__", None),
    }
    for name in (
        "gcnArchName",
        "warp_size",
        "max_threads_per_block",
        "max_threads_per_multi_processor",
        "regs_per_multiprocessor",
        "shared_memory_per_block",
        "shared_memory_per_block_optin",
        "shared_memory_per_multiprocessor",
        "total_memory",
        "multi_processor_count",
    ):
        value = getattr(properties, name, None)
        if value is not None:
            facts[name] = value
    return facts


def _kernel_facts(compiled) -> dict:
    """Return the LDS/register footprint and lowering signals of a kernel."""
    facts = {}
    metadata = getattr(compiled, "metadata", None)
    for name in ("shared", "num_warps", "num_stages"):
        value = getattr(metadata, name, None)
        if value is not None:
            facts[name] = value
    for name in ("n_regs", "n_spills", "n_max_threads"):
        value = getattr(compiled, name, None)
        if value is not None:
            facts[name] = value

    asm = getattr(compiled, "asm", None) or {}
    if hasattr(asm, "keys") and hasattr(asm, "get"):
        facts["asm_keys"] = sorted(str(key) for key in asm)
        blob = "\n".join(
            str(asm.get(key, "")) for key in ("ttgir", "llir", "amdgcn", "ptx")
        )
    else:
        blob = str(asm)
    lowered = [marker for marker in (*_AMD_MARKERS, *_NVIDIA_MARKERS) if marker in blob]
    facts["lowering_markers"] = lowered
    matrix = [marker for marker in lowered if "mfma" in marker or "mma" in marker]
    facts["matrix_markers"] = matrix
    facts["uses_matrix_instructions"] = bool(matrix)
    return facts


if triton is not None:

    @triton.jit
    def _pointwise_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, values * 2.0 + 1.0, mask=mask)

    @triton.jit
    def _dot_kernel(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr, IEEE: tl.constexpr):
        rows = tl.arange(0, BLOCK)
        cols = tl.arange(0, BLOCK)
        inner = tl.arange(0, BLOCK)
        block_a = tl.load(a_ptr + rows[:, None] * BLOCK + inner[None, :])
        block_b = tl.load(b_ptr + inner[:, None] * BLOCK + cols[None, :])
        acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        if IEEE:
            acc = tl.dot(block_a, block_b, acc, input_precision="ieee")
        else:
            acc = tl.dot(block_a, block_b, acc)
        tl.store(c_ptr + rows[:, None] * BLOCK + cols[None, :], acc)


def _run_pointwise(device: torch.device) -> dict:
    n = 4096
    x = torch.randn(n, device=device)
    y = torch.empty_like(x)
    block = 1024
    compiled = _pointwise_kernel[(triton.cdiv(n, block),)](x, y, n, BLOCK=block)
    torch.cuda.synchronize()
    max_error = (y - (x * 2.0 + 1.0)).abs().max().item()
    facts = _kernel_facts(compiled)
    facts["max_abs_error"] = max_error
    facts["ok"] = bool(max_error == 0.0)
    return facts


def _run_dot(device: torch.device) -> dict:
    block = 64
    a = torch.randn(block, block, device=device)
    b = torch.randn(block, block, device=device)
    c = torch.zeros(block, block, device=device)
    expected = a @ b
    ieee = True
    try:
        compiled = _dot_kernel[(1,)](a, b, c, BLOCK=block, IEEE=True, num_warps=4)
    except TypeError:
        # Older Triton rejects ``input_precision``; retry with the default path.
        ieee = False
        c.zero_()
        compiled = _dot_kernel[(1,)](a, b, c, BLOCK=block, IEEE=False, num_warps=4)
    torch.cuda.synchronize()
    difference = (c - expected).abs().max().item()
    scale = expected.abs().max().item()
    facts = _kernel_facts(compiled)
    facts["input_precision"] = "ieee" if ieee else "default"
    facts["max_abs_error"] = difference
    facts["max_rel_error"] = difference / max(scale, 1e-12)
    # tf32 keeps ~10 mantissa bits, so anything below 1e-2 counts as correct.
    facts["ok"] = bool(difference / max(scale, 1e-12) < 1e-2)
    return facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", type=str, default=None, help="write the report here")
    args = parser.parse_args(argv)

    if triton is None:
        print(f"triton import failed: {TRITON_IMPORT_ERROR!r}")
        return 2
    if not torch.cuda.is_available():
        print("no DCU/CUDA device visible; run this on the profiling host")
        return 2

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    report = {"device": _device_facts(device), "kernels": {}}
    print(json.dumps(report["device"], indent=2, sort_keys=True, default=str))

    exit_code = 0
    for name, runner in (("pointwise", _run_pointwise), ("dot", _run_dot)):
        try:
            facts = runner(device)
        except Exception as error:  # noqa: BLE001 - a failure is the result here
            facts = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        report["kernels"][name] = facts
        print(f"\n[{name}] " + json.dumps(facts, indent=2, sort_keys=True, default=str))
        if not facts.get("ok"):
            exit_code = 1

    matrix = report["kernels"].get("dot", {}).get("matrix_markers")
    print(
        "\ntl.dot matrix-core path: "
        + (f"{matrix}" if matrix else "NOT FOUND (dot lowered to scalar FMAs)")
    )
    if args.json:
        from pathlib import Path

        Path(args.json).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(f"report written to {args.json}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
