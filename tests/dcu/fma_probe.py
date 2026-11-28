"""Lazy test-only FFMA-throughput probe and the projection that uses it.

The decision this supports: the fused ``dpa4_so2_conv`` replaces roughly the
whole per-edge span of one ``SO2Convolution`` (the ``G2`` segment of the trace:
163.8 ms forward / 416.2 ms backward at 3000 atoms), but it pays for that in
float32 SIMT multiply-adds -- ``M0^2 + M1^2 + Cf*GATE`` per edge, per focus
stream, per layer.  Upstream's own calibration declines the substitution above
65.5 kFMA per layer, and this checkpoint sits at 225.3 kFMA, so the guard says
no without asking about the device.  The guard compares against the *Triton*
composition, though, and on DCU the baseline is the dense path, which is what
the trace measures -- so the question becomes quantitative:

    projected_ms = 2 * per_layer_fma * edges * layers * blocks / ffma_flops_per_s

If that lands well below 163.8 ms, the port is worth its 24 translation units;
if it lands above, no amount of kernel engineering will rescue it.

``projection`` is pure arithmetic and runs anywhere; ``_load_extension`` and
``measure`` need a HIP-enabled PyTorch with a visible DCU.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

#: Shape of the profiled checkpoint, read off the loaded model
#: (``lmax=3, Cf=64, n_focus=1, mixing_layers=4, mmax=1, n_head=1``, 3 blocks).
AIR_EDGES = 181838
AIR_BLOCKS = 3
AIR_LAYERS = 4
AIR_LMAX = 3
AIR_CF = 64
AIR_N_FOCUS = 1
#: One gated mixing layer: ``M0^2 + M1^2 + Cf*GATE`` with ``M0 = (lmax+1)Cf``,
#: ``M1 = 2 lmax Cf``, ``GATE = lmax Cf``.  This is the constant upstream's
#: ``_profitable`` compares against, and the one its calibration quotes.
AIR_PER_LAYER_FMA = (4 * 64) ** 2 + (6 * 64) ** 2 + 64 * 3 * 64  # 225 280


def air_per_edge_fma(
    *,
    lmax: int = AIR_LMAX,
    cf: int = AIR_CF,
    mixing_layers: int = AIR_LAYERS,
    n_focus: int = AIR_N_FOCUS,
) -> dict:
    """Return the per-edge FMA of the whole value path, by component.

    The mixing stack is ``mixing_layers - 1`` gated layers plus an identity
    final layer (no gate), each contracting the two block-diagonal blocks; the
    two rotations and the radial degree mixer ride on top.  Summing only the
    layers and forgetting the rotations is what made the first projection 15%
    optimistic.
    """
    m0 = (lmax + 1) * cf
    m1 = 2 * lmax * cf
    gate = lmax * cf
    dim = (lmax + 1) ** 2
    red = 3 * lmax + 1
    gated = max(0, mixing_layers - 1)
    components = {
        "gated_layers": gated * (m0 * m0 + m1 * m1 + cf * gate),
        "identity_layer": m0 * m0 + m1 * m1,
        "rotations": 2 * red * dim * cf,
        "radial_mix": red * red * cf,
    }
    components["total"] = sum(components.values()) * n_focus
    components["layer"] = m0 * m0 + m1 * m1 + cf * gate
    return components
#: Dense-path reference for the same span, from the trace.
DENSE_FORWARD_MS = 163.8
DENSE_BACKWARD_MS = 416.2
#: Device bandwidth used to turn the FLOP rate into a ridge point.
BANDWIDTH_GBPS = 630.0
#: Upstream's profitability constants (`cuda/dpa4/so2_conv.py`).
RIDGE_REF = 73.2
MAX_PROFITABLE_LAYER_FMA = 65536


class ProbeError(RuntimeError):
    """Probe extension loading or execution failed."""


def projection(
    tflops: float,
    *,
    edges: int = AIR_EDGES,
    blocks: int = AIR_BLOCKS,
    per_edge_fma: Optional[int] = None,
    dense_ms: float = DENSE_FORWARD_MS,
) -> dict:
    """Project the fused convolution's arithmetic time from a measured FLOP rate.

    The bytes are ignored on purpose: the fused form keeps the per-edge
    intermediates in registers, so its traffic is a few hundred MB per span
    (a couple of milliseconds at this device's bandwidth) and the FLOP term
    dominates.  The dense reference is the trace's device time for the same span.
    """
    if tflops <= 0:
        raise ValueError("tflops must be positive")
    components = air_per_edge_fma()
    per_edge = int(per_edge_fma if per_edge_fma is not None else components["total"])
    flop = 2.0 * per_edge * edges * blocks
    projected_ms = flop / (tflops * 1e12) * 1e3
    return {
        "tflops": tflops,
        "flop": flop,
        "projected_ms": projected_ms,
        "dense_ms": dense_ms,
        "speedup_over_dense": dense_ms / projected_ms if projected_ms else float("inf"),
        "ridge_flop_per_byte": tflops * 1e12 / (BANDWIDTH_GBPS * 1e9),
        "profitable_budget_fma": MAX_PROFITABLE_LAYER_FMA
        * (tflops * 1e12 / (BANDWIDTH_GBPS * 1e9))
        / RIDGE_REF,
        "per_edge_fma": per_edge,
        "per_layer_fma": components["layer"],
        "flop_per_edge": 2.0 * per_edge,
    }


@lru_cache(maxsize=1)
def _load_extension():
    import torch

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise ProbeError("the FFMA probe requires HIP-enabled PyTorch and a visible DCU")
    try:
        from torch.utils.cpp_extension import load
    except ImportError as error:  # pragma: no cover - torch always ships it
        raise ProbeError("torch.utils.cpp_extension is required") from error

    source = Path(__file__).with_name("fma_probe.cpp")
    if not source.is_file():
        raise ProbeError(f"probe source is missing: {source}")

    compiler = os.environ.get("HIPCXX") or "hipcc"
    digest = hashlib.sha1(source.read_bytes()).hexdigest()[:10]
    previous_cxx = os.environ.get("CXX")
    os.environ["CXX"] = compiler
    try:
        return load(
            name=f"dpa4_fma_probe_{digest}",
            sources=[str(source)],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    finally:
        if previous_cxx is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = previous_cxx


def measure(
    *,
    blocks: int = 960,
    threads: int = 256,
    iters: int = 20000,
    nacc: int = 16,
    repeats: int = 5,
) -> float:
    """Return the sustained FFMA rate in TFLOP/s for one launch geometry."""
    import statistics

    import torch

    extension = _load_extension()
    out = torch.zeros(1, device="cuda", dtype=torch.float32)

    def one() -> None:
        extension.run(out, blocks, threads, iters, nacc)

    for _ in range(2):
        one()
    torch.cuda.synchronize()

    samples = []
    for _ in range(max(3, repeats)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        one()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    best_ms = min(samples)
    flop = 2.0 * blocks * threads * iters * nacc
    return flop / (best_ms * 1e-3) / 1e12


def available() -> bool:
    """Whether the probe can run here (no extension load attempted)."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    return bool(torch.cuda.is_available()) and torch.version.hip is not None


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=960)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--nacc", type=int, default=16, choices=(1, 4, 8, 16))
    args = parser.parse_args(argv)

    rate = measure(blocks=args.blocks, threads=args.threads, iters=args.iters, nacc=args.nacc)
    report = projection(rate)
    print(f"grid {args.blocks}x{args.threads}, {args.iters} iters, {args.nacc} accumulators")
    print(f"sustained {rate:.1f} TFLOP/s (ffma only, no memory traffic)")
    print(f"ridge {report['ridge_flop_per_byte']:.1f} FLOP/byte "
          f"(at {BANDWIDTH_GBPS:.0f} GB/s)")
    print(f"per edge: {report['per_edge_fma']} FMA "
          f"({report['flop'] / 1e12:.2f} TFLOP over {AIR_EDGES} edges x {AIR_BLOCKS} blocks)")
    print(f"projected fused conv fwd: {report['projected_ms']:.1f} ms "
          f"vs dense {report['dense_ms']:.1f} ms "
          f"-> {report['speedup_over_dense']:.2f}x")
    print(f"upstream _profitable budget: {report['profitable_budget_fma']:.0f} FMA "
          f"vs {report['per_layer_fma']} required "
          f"({'would decline' if report['per_layer_fma'] > report['profitable_budget_fma'] else 'would accept'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
