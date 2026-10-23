# DPA4 kernels on DTK

This document is the single native-kernel and DPA4 numerical reference for `myproject`. It borrows ThunderKittens' explicit tiles, data ownership, and loader/consumer/storer organization; it does **not** copy its CUDA implementation or claim a full-model MegaKernel.

## Scope and status

- The supported route today is the existing DeePMD eager adapter plus a growing set of project-owned HIP primitives. `DPA4Model` is still experimental: its OMol25 Neo checkpoint, complete E/F/V path, and full VJP have not passed numerical validation.
- Native primitives include geometry, Bessel/C3 radial features, fused edge features, ordinary RMSNorm, linear, segment sum, and force/virial assembly. They do not make every DPA4 stage native.
- The persistent worker is opt-in and accepts only `edge_features` or the two-stage `edge_geometry → radial_basis` subgraph. The planner's larger static DAG is not a native executor.
- No DCU build, accuracy run, or performance result is claimed for the TK-inspired DUMMA linear path added here. See the verification section before interpreting source coverage as hardware support.

## Data and mathematical contracts

For an edge `e`, `edge_index[0]` is `src`, `edge_index[1]` is `dst`; cells are row vectors:

```text
d[e] = pos[src[e]] + shift[e] @ cell[edge_frame[e]] - pos[dst[e]]
r[e] = sqrt(sum(d[e] * d[e]) + eps * eps)
```

Messages gather from `src` and reduce to `dst`. Masked capacity entries must still contain valid indices; `-1` is not padding. Masking must be carried through radial features, attention normalization, reductions, force/virial assembly, and their VJPs.

Radial features use the stable Bessel form `w * sinc(w*r/pi)` with the checkpoint-configured C3 cutoff envelope. Energy forces require the full derivative `g = dE/dd`; assembly follows `F[dst] += g`, `F[src] -= g`, and `virial[frame] -= outer(g,d)`. Native force/virial assembly alone does not provide the missing model VJP.

The complete reference data flow is:

```text
atoms + explicit edge graph
  → type/charge-spin and edge geometry/radial features
  → environment/FiLM and geometric seed
  → focus/gather/rotation/radial mixing/SO(2)/attention/message reduction
  → SO(3)-grid message and FFN paths
  → readout/fitting/frame energy
  → edge-vector VJP → force and virial
```

Checkpoint-selected options such as charge/spin, grids, attention, and radial mixing are part of the model contract; water dimensions must not be substituted for the OMol25 Neo configuration. The machine-readable operator catalogue and shapes are available via `python -m model operators` and `python/dcu_dpa4/catalog.py` rather than a second hand-maintained operator table.

## ThunderKittens ideas translated to DTK

| Kernel level | Project implementation | Contract and limit |
|---|---|---|
| Tile metadata | `tile_layout<R,C>` and `gemm_tile_shape<M,N,K>` in `include/common/tile.cuh` | Compile-time row-major shape/index metadata. It does not expose or emulate a hardware fragment's lane distribution. |
| Thread/register | `register_tile<T,R,C>` | One thread owns the complete small tile; large instances may spill. It is not ThunderKittens' distributed register tile. |
| Block/LDS | `group::load`, `group::store`, `block_sync` | Cooperative synchronous copy with masked zero-fill. The block barrier is a rendezvous; fences are not barriers. |
| Consumer/MMA | `group::mma_16x16<float,4>` / `dcu::dumma` | DTK owns the opaque DUMMA fragment. The complete wave64 must participate uniformly; no lane-wise fragment indexing or masked MMA. |
| Persistent scheduling | `csrc/megakernel/runtime.cu` | A bounded HIP ticket/depends protocol for the two listed edge-feature plans only. It is not CLC and does not provide a grid-wide barrier. |

The opt-in `linear_hip` DUMMA path uses one 64-thread CTA per 16×16 output tile and FP32 K=4 fragments. Each reduction slice is cooperatively loaded to LDS, consumed by DUMMA, and synchronized before reuse; the output tile is staged to LDS so M/N tails can be masked and bias applied. This is a **synchronous** pipeline: there is no TMA, asynchronous copy, WGMMA, cluster barrier, or overlap claim. The ordinary thread-per-output kernel remains the default unless the extension is built with DUMMA enabled.

ThunderKittens CLC, PTX, TMA, WGMMA, warp32 fragment assumptions, and NVIDIA mbarriers are not portable APIs. On DTK, use the installed compiler's supported DUMMA shapes and HIP synchronization, and preserve explicit tile ownership, LDS budget, tail handling, stream, and gradient contracts.

## Build and use

Pure Python install does not build native operators. On a DTK PyTorch environment:

```bash
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
DPA4_BUILD_HIP=1 MAX_JOBS=4 python setup.py build_ext --inplace
```

The DUMMA-tiled linear path is a separate opt-in build. It requires `DPA4_BUILD_HIP=1`, `du_mma.h`, and a supported gfx target; it does not silently fall back if DUMMA is requested but unavailable:

```bash
DPA4_BUILD_HIP=1 DPA4_ENABLE_DUMMA=1 MAX_JOBS=4 \
  python setup.py build_ext --inplace
```

The FP32 K=4 DUMMA wrapper documents gfx926/928/936/938 support. Confirm the actual DTK/compiler target before building. The existing `DPA4_BUILD_ROCBLAS=1` path is a separate **host-launched library baseline**; it cannot be called from a device-side worker and does not imply that the linear kernel uses rocBLAS.

## Tests and evidence

After building, the relevant checks are:

```bash
PYTHONPATH=python python -m unittest tests.csrc.test_instructions.InstructionHeaderTests -v
DPA4_RUN_INSTRUCTION_TESTS=1 PYTHONPATH=python \
  python -m unittest tests.csrc.test_instructions -v
python -m unittest tests.csrc.test_accuracy -v
python -m tests.csrc.benchmark --atoms 192 --edges 12288 --channels 16 --radial 6
```

With the opt-in DUMMA build, the HIP accuracy test exercises the same linear API on non-multiple M/K/N dimensions and checks bias and gradients. A passing host/header test is not evidence of DTK compilation or device correctness. A skipped HIP test is not a pass. Latency must be measured on the target DCU and compared against the same workload on the existing baseline; this repository does not yet contain an end-to-end Qwen serving MegaKernel or measured acceleration claim.
