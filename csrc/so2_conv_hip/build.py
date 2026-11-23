#!/usr/bin/env python3
"""Build the translated HIP fused SO(2) convolution as a loadable extension.

The translated sources register ``deepmd::dpa4_so2_conv`` (and its backward,
``dpa4_wigner_runs``, ``dpa4_fp32_ridge``) into the *same* ``deepmd`` operator
namespace DeepMD uses, so upstream's Python wrapper -- ``op_available()``,
``_setup_context``, ``_backward``, the fake impls, the profitability guard --
finds them with no source edit on either side.

v1 ships both directions of one instantiation, ``(lmax=3, focus_dim=64)``, which
is what the profiled checkpoint needs: the forward is what the descriptor calls
and the backward is what the force path differentiates through.  The other
eleven ``(CF, L)`` pairs are the same template -- add their shards to
``tools/hipify_so2_conv.py`` when a checkpoint needs one; compiling all
twenty-four is what upstream's per-shape translation units are for, and it is
the slow part of the build.

Usage (on a DCU host, from the repo root)::

    python csrc/so2_conv_hip/build.py                 # build and register
    python csrc/so2_conv_hip/build.py --check         # build, then report schema
    python csrc/so2_conv_hip/build.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "dpa4_so2_conv_hip"
SOURCES = ("so2_conv.hip", "so2_conv_fwd_c64_l3.hip", "so2_conv_bwd_c64_l3.hip")


def _require_hip() -> None:
    import torch

    if torch.version.hip is None:
        raise RuntimeError(
            "the fused SO(2) convolution is a HIP operator: build it on a DCU host"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("no visible DCU")


def build(*, verbose: bool = False):
    """Compile and load the extension; returns the loaded module.

    Loading registers the operators, so after this returns
    ``torch.ops.deepmd.dpa4_so2_conv`` exists and DeepMD's own
    ``op_available()`` reports ``True``.
    """
    _require_hip()
    from torch.utils.cpp_extension import load

    sources = [str(HERE / name) for name in SOURCES]
    missing = [name for name in sources if not Path(name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing translated sources: {missing}")

    return load(
        name=NAME,
        sources=sources,
        extra_include_paths=[str(HERE)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=verbose,
    )


def registered() -> bool:
    """Whether ``deepmd::dpa4_so2_conv`` is registered in this process."""
    import torch

    op = getattr(torch.ops.deepmd, "dpa4_so2_conv", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--check", action="store_true", help="report the schema after loading")
    args = parser.parse_args(argv)

    build(verbose=args.verbose)
    print(f"registered: {registered()}")
    if args.check:
        import torch

        for name in (
            "dpa4_so2_conv",
            "dpa4_so2_conv_backward",
            "dpa4_wigner_runs",
            "dpa4_fp32_ridge",
        ):
            packet = getattr(torch.ops.deepmd, name, None)
            print(f"  deepmd::{name}: {'ok' if packet is not None else 'MISSING'}")
        if registered():
            print("\ndefault schema:\n ", torch.ops.deepmd.dpa4_so2_conv.default._schema)
        # DeepMD's guard compares this against 73.2 (the calibration part's
        # ridge); printing it is the honest way to see why the fused path is
        # declined or accepted on this device.
        try:
            print("dpa4_fp32_ridge():", torch.ops.deepmd.dpa4_fp32_ridge())
        except Exception as error:  # noqa: BLE001 - reporting only
            print("dpa4_fp32_ridge() failed:", error)
    return 0


if __name__ == "__main__":
    sys.exit(main())
