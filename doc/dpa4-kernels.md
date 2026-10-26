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

Checkpoint-selected options such as charge/spin, grids, attention, and radial mixing are part of the model contract; water dimensions must not be substituted for the OMol25 Neo configuration. The machine-readable operator catalogue and shapes are available via `python -m runtime operators` and `python/kernels/catalog.py` rather than a second hand-maintained operator table.

## DeepMD-kit execution path and operator map

The source references below point into the adjacent `../deepmd-kit` checkout; links are relative to this document. PyTorch registers `SeZM`/`sezm` and `DPA4`/`dpa4` on the same `SeZMModel` ([model registration](../../deepmd-kit/deepmd/pt/model/model/sezm_model.py#L685)) and `DescrptSeZM` ([descriptor registration](../../deepmd-kit/deepmd/pt/model/descriptor/sezm.py#L125)) paths. `pt_expt` also registers `DescrptDPA4`, a wrapper around the dpmodel DPA4 descriptor ([wrapper registration](../../deepmd-kit/deepmd/pt_expt/descriptor/dpa4.py#L193)). DPA4C is a separate model family and is not covered here. The selected checkpoint/configuration determines which optional branches are active.

**Single-rank PyTorch energy path**

```text
SeZMModel.forward
  → forward_common: cast inputs, build local neighbor/edge schema
  → forward_common_lower → core_compute
  → DescrptSeZM.forward_with_edges
  → edge cache + initial embeddings + interaction blocks + scalar readout
  → fitting/output statistics + optional pair potential
  → frame energy reduction → edge_energy_deriv → force / virial outputs
```

The public forward and edge-builder are in [`sezm_model.py:795`](../../deepmd-kit/deepmd/pt/model/model/sezm_model.py#L795), [`sezm_model.py:963–1069`](../../deepmd-kit/deepmd/pt/model/model/sezm_model.py#L963), and [`sezm_model.py:2768–2783`](../../deepmd-kit/deepmd/pt/model/model/sezm_model.py#L2768). The lower/core energy path is [`sezm_model.py:1071–1279`](../../deepmd-kit/deepmd/pt/model/model/sezm_model.py#L1071) and [`sezm_model.py:1407–1691`](../../deepmd-kit/deepmd/pt/model/model/sezm_model.py#L1407); descriptor order is [`sezm.py:1470–1782`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm.py#L1470).

| Stage / operator family | What executes (configuration-dependent branches are marked optional) | DeepMD-kit source |
|---|---|---|
| Neighbor graph and edge schema | Build local neighbors; form `src/dst`, displacement vectors, validity masks, and separate message/force scatter indices. | [`sezm_model.py:963–1069, 2768–2783`](../../deepmd-kit/deepmd/pt/model/model/sezm_model.py#L963); graph lower assembly: [`DeepPotPTExpt.cc:1181–1195`](../../deepmd-kit/source/api_cc/src/DeepPotPTExpt.cc#L1181) |
| Atom/condition embedding | `SeZMTypeEmbedding`; optional frame-level charge/spin embedding; optional native per-atom spin embedding. | [`embedding.py:63–150, 793–1007`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/embedding.py#L63); dispatch: [`sezm.py:1596–1619`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm.py#L1596) |
| Edge feature cache | Mask/type exclusion → safe edge norm → optional inner clamp → C3 cutoff envelope and radial basis → edge quaternion/Wigner-D → source/destination type feature → destination `sum(edge_env²)` and inverse-sqrt degree. Evaluation may replace radial/Wigner compositions with fused kernels. | [`edge_cache.py:436–594, 657–738`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/edge_cache.py#L436); radial: [`radial.py:201–547`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/radial.py#L201); quaternion/Wigner: [`wignerd.py:344–527`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/wignerd.py#L344) |
| Optional SFPG / bridging | `BridgingSwitch` gives per-edge weights; source-wise products are represented by `[log_eta, zero_count]` scatter-add partials, then converted to a source gate. Masked edges contribute the multiplicative identity. | [`edge_cache.py:164–284`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/edge_cache.py#L164); switch: [`radial.py:349–423`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/radial.py#L349) |
| Initial node state | Optional environment-seed matrix/MLP produces FiLM scale/shift; GIE rotates zonal radial features with cached Wigner data and destination-reduces them; optional spin contributes `l=1`. | [`embedding.py:152–416, 417–792`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/embedding.py#L152); assembly: [`sezm.py:1673–1724`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm.py#L1673) |
| Interaction blocks | Optional pre/post equivariant normalization; SO(2) message unit; residuals or depth/block attention-residual; one or more equivariant FFN subblocks. | [`block.py:119–170, 670–818, 858–1040`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/block.py#L119) |
| SO(2) message operator | Pre/post focus channel mixing; global→edge-local Wigner rotation; radial modulation and reduced-degree `DynamicRadialDegreeMixer`/`SO2Linear`; optional focus competition. Aggregate either by envelope × optional source gate + destination scatter-add + degree normalization, or by destination-segment envelope-gated attention; rotate messages back to global coordinates. Optional Cartesian tensor-product/grid-product branches are config-selected. | [`so2.py:127–401, 634–850, 898–955, 1778–1929, 2571–2760`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/so2.py#L127); attention softmax: [`attention.py:18–135`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/attention.py#L18); optional Cartesian: [`cartesian.py:204–489`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/cartesian.py#L204) |
| Equivariant FFN / grid paths | SO(3) channel-linear → equivariant gated activation, or optional S2/SO3-grid GLU/MLP/branch operation → SO(3) projection. | [`ffn.py:54–325`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/ffn.py#L54); grid primitives: [`grid_net.py:181–433, 628–813`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/grid_net.py#L181) |
| Readout and fitting | Keep owned-node rows, apply scalar readout FFN, then atomic fitting; apply output statistics and optional analytic ZBL/pair energy before reducing atomic energies. | [`sezm.py:1750–1782, 1900–1940`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm.py#L1750); fitting/stat/ZBL: [`sezm_model.py:1578–1641`](../../deepmd-kit/deepmd/pt/model/model/sezm_model.py#L1578) |
| Force / virial | Make `edge_vec` the coordinate-gradient leaf; differentiate reduced energy w.r.t. it once; scatter edge gradients into endpoint forces and edge virials. `create_graph` preserves the higher derivative needed for force-loss training. | [`sezm_model.py:1497–1508, 1656–1691`](../../deepmd-kit/deepmd/pt/model/model/sezm_model.py#L1497); derivative/scatter: [`transform_output.py:214–270`](../../deepmd-kit/deepmd/pt/model/model/transform_output.py#L214) |

## Single-rank and multi-rank communication

- **Single rank / ordinary Python forward:** `comm_dict=None`; the neighbor list and descriptor operate on the local node set. There is no `border_op`, MPI feature exchange, DDP, or tensor-parallel collective in this path.
- **Multi-rank LAMMPS domain decomposition:** each rank owns `nlocal` nodes followed by `nghost` ghost nodes; the SeZM parallel descriptor requires one frame per rank. The message-passing graph addresses the extended node set. Before each block's SO(2) convolution—the block's cross-node operation—`exchange_ghost_features` overwrites ghost feature rows with their owner-rank values. The FFN and residual history are node-local; they do not themselves communicate. [`sezm.py:1488–1495, 1542–1563`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm.py#L1488), [`block.py:66–116, 749–818`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/block.py#L66), [`sezm.py:1784–1825`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm.py#L1784)

`comm_dict` contains eight host control tensors: `send_list`, `send_proc`, `recv_proc`, `send_num`, `recv_num`, `communicator`, `nlocal`, and `nghost`. `border_op` packs node rows contiguously, gathers owner rows using each swap's send list, exchanges them with MPI send/receive, and writes received rows into the ghost tail. Its custom backward reverses the swaps, accumulates ghost-output gradients into owner rows, and zeros overwritten ghost-input gradients; Python autograd registration exposes the forward/backward pair as transpose operators. This is **owner↔ghost halo exchange**, not a model-wide all-reduce. [`comm.cc:32–211, 213–422`](../../deepmd-kit/source/op/pt/comm.cc#L32), [`comm.py`](../../deepmd-kit/deepmd/pt_expt/utils/comm.py), [`block.py:66–116`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/block.py#L66)

When SFPG bridging is enabled, the per-source `[log_eta, zero_count]` partials are a separate communication: `border_op_backward` reverse-accumulates ghost partials to owner rows, then `border_op` broadcasts the completed values back to ghosts. The standard feature refresh runs at every SO(2) block input; this extra exchange is only used by the optional bridge gate. [`edge_cache.py:164–284`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/edge_cache.py#L164), [`sezm.py:2518–2559`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm.py#L2518)

For forces, `edge_energy_deriv` emits forces/atomic virials over the edge scatter domain. The generic PyTorch extended-output helper can scatter-reduce extended rows through `mapping`; the LAMMPS multi-rank graph route instead keeps the extended region and lets LAMMPS reverse-communicate ghost forces to owners. These are output ownership steps, distinct from the per-block feature exchange. A small `allreduce_min_int` in the graph runtime is only a collective preflight that ensures every rank has nodes before entering per-layer exchanges; it does not reduce DPA4 features or energy. [`transform_output.py:214–270, 416–481`](../../deepmd-kit/deepmd/pt/model/model/transform_output.py#L214), [`DeepPotPTExpt.cc:799–813, 1126–1163`](../../deepmd-kit/source/api_cc/src/DeepPotPTExpt.cc#L799), [`comm.cc:618–640`](../../deepmd-kit/source/op/pt/comm.cc#L618)

## Exported `.pt2` communication boundary

The DPA4 **graph lower** has a with-comm export path carrying the eight control tensors and opaque `deepmd_export::border_op` calls. `DeepPotPTExpt` selects that artifact for multi-rank LAMMPS; the DPA4 dense/nlist lower does not implement the corresponding `comm_dict` path. The graph API also carries a separate device-side `n_local` owned-node count for model masks; it is not one of the eight host communication tensors. This is LAMMPS MPI/domain decomposition, not PyTorch DDP. [`serialization.py:187–257`](../../deepmd-kit/deepmd/pt_expt/utils/serialization.py#L187), [`ener_model.py:742–828`](../../deepmd-kit/deepmd/pt_expt/model/ener_model.py#L742), [`DeepPotPTExpt.cc:799–832, 1119–1195`](../../deepmd-kit/source/api_cc/src/DeepPotPTExpt.cc#L799), [`test_dpa4_export.py:751–765`](../../deepmd-kit/source/tests/pt_expt/model/test_dpa4_export.py#L751)

**Index-space caveat:** `DescrptSeZM.forward_with_edges` explicitly specifies local indices when `comm_dict=None`, and extended-node indices when communication is active. The current LAMMPS graph assembly retains the extended nodes for that route. Some generic `core_compute`/`forward_lower` docstrings still describe local edge indices, while `pt_expt/utils/edge_schema.py` is explicitly marked legacy and maps edge indices separately from force-scatter indices. Treat the index space as lower-ABI-specific; do not feed the legacy `edge_vec` schema to the `.pt2` graph lower without its matching conversion. [`sezm.py:1488–1495`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm.py#L1488), [`sezm_model.py:1435–1446, 1821–1833`](../../deepmd-kit/deepmd/pt/model/model/sezm_model.py#L1435), [`DeepPotPTExpt.cc:1119–1125, 1181–1195`](../../deepmd-kit/source/api_cc/src/DeepPotPTExpt.cc#L1119), [`edge_schema.py:1–6, 143–199`](../../deepmd-kit/deepmd/pt_expt/utils/edge_schema.py#L1)

## Tile-level primitive development plan

The upstream operator graph above is the numerical contract; it is not a list of one-kernel-per-Python-function. For DTK, organize reusable work around logical **edge tiles** (`T_E` edges), **node/coefficient tiles** (`T_N × D × C`), and **destination-segment tiles**. `T_E`, `T_N`, degree/channel blocking, and waves-per-CTA remain target-profiled parameters, not assumed ThunderKittens sizes. A valid tile must preserve masks, source/destination index spaces, accumulation precision, and the coordinate VJP.

| Priority | Tile primitive to develop | Tile contract / DPA4 composition | Current project coverage and next step |
|---|---|---|---|
| P0 — foundation | Predicated tile memory and layout | Node/edge/coefficient views; masked load with zero-fill; safe indexed gather; masked store; contiguous LDS staging and explicit producer/consumer barriers. Invalid padded indices must never be dereferenced. | [`tile.cuh`](../include/common/tile.cuh) has row-major metadata, `matrix_view`, and one-thread `register_tile`; [`group/memory.cuh`](../include/ops/group/memory.cuh) and [`group/sync.cuh`](../include/ops/group/sync.cuh) provide synchronous group load/store and barriers. Add explicit ownership/gather/reduction contracts; current tile metadata is not a distributed lane layout. |
| P0 — foundation | Tile reductions and segmented accumulation | Warp/CTA sum and max, then CSR destination-segment sum/max with scratch/second-stage reduction when a segment crosses CTAs. Reuse for degree `sum(env²)`, GIE, message aggregation, and energy/virial reductions. | [`reduce.cu`](../csrc/parallel/reduce.cu) has atomic `segment_sum`; develop deterministic/low-contention segmented tile paths and do not assume a grid-wide barrier. |
| P1 — base E/F/V | Edge-feature tile | For each valid edge: endpoint/cell gather → displacement, safe distance/unit vector → Bessel/sinc basis and both C3 envelopes → mask-safe stores; provide the matching edge-vector/frequency VJP. | Fused operator and first-order backward exist in [`edge_features.cu`](../csrc/parallel/edge_features.cu) and [`edge_features.cuh`](../csrc/include/megakernel/edge_features.cuh); factor reusable tile bodies and test edge tails/masks before expanding fusion. Separate geometry/radial kernels are in [`geometry.cu`](../csrc/parallel/geometry.cu) and [`radial.cu`](../csrc/parallel/radial.cu). |
| P1 — base E/F/V | GEMM/linear tile plus epilogue | `A[T_M,T_K] × B[T_K,T_N] → C[T_M,T_N]`, with masked M/N/K tails; optional bias and compatible activation/norm epilogues. Covers radial MLPs, node channel projections, and readout/fitting linears. | [`mma.cuh`](../include/ops/group/mma.cuh) wraps opaque DTK DUMMA fragments; [`linear.cu`](../csrc/gemm/linear.cu) has an opt-in 16×16 FP32/K=4 DUMMA forward. Generalize/reuse this tile without exposing fragment lane layouts; add parity and gradient coverage for every fused epilogue. rocBLAS remains a host-launched baseline, not a device-worker primitive. |
| P1 — base E/F/V | Node pointwise tile | Type/condition table gathers, pair-feature gather/add, channel RMS/equivariant normalization, gated activation, residual/layer-scale, and degree truncation over node/channel tiles. | `rmsnorm` has a HIP forward/backward in [`norm.cu`](../csrc/parallel/norm.cu); type/pair embedding and most equivariant pointwise stages are still upstream PyTorch. Ordinary channel RMSNorm is not a replacement for equivariant per-degree norms. |
| P1 — base E/F/V | Edge-gradient and force/virial tile | Consume `g_e = dE/d(edge_vec_e)`; mask invalid edges; accumulate `+g/-g` to endpoints and `-g ⊗ d` to the frame/atom virial. Handle endpoint contention with atomics or a separate segmented reduction. | [`reduce.cu`](../csrc/parallel/reduce.cu) has the basic force/virial assembly, but a complete accelerated energy path still needs the VJP of every selected descriptor/fitting stage. |
| P2 — descriptor core | Quaternion/Wigner rotation tile | Build edge-aligned rotations and apply structured per-degree/global↔local transforms to coefficient tiles. Keep block-diagonal/per-degree structure; materialize dense `E×D×D` only if profiling proves it worthwhile. | Upstream contract: [`edge_cache.py`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/edge_cache.py#L407), [`wignerd.py`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/wignerd.py#L344). No project HIP tile exists yet. |
| P2 — descriptor core | SO(2) edge-message tile | Gather source-node coefficients → rotate/project to reduced m-major layout → radial degree/channel modulation and `SO2Linear` → equivariant gate/residual/focus → rotate back. Fuse only where intermediate reuse, numerical order, and VJP remain correct. | Upstream sequence: [`so2.py:898–955`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/so2.py#L898), [`block.py:798–818`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/block.py#L798). No project native implementation yet. |
| P2 — descriptor core | Destination attention/reduction tile | For each destination segment, compute stable max, exp/sum, positive null mass, envelope-squared edge mass, optional SFPG source gate, then weighted message accumulation. Preserve empty/all-masked segment behavior. | Upstream contract: [`attention.py:18–135`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/attention.py#L18); project `segment_sum` does not implement this softmax. CSR reordering/partial accumulation may require multiple launches; no inter-CTA spin barrier. |
| P2 — descriptor core | Equivariant FFN tile | Per-degree SO(3) channel linears plus scalar-gated non-scalar activations, equivariant norm and residual. Treat `D`/degree layout explicitly; do not flatten irreps into ordinary RMSNorm semantics. | Upstream: [`ffn.py:54–325`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/ffn.py#L54), [`norm.py`](../../deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/norm.py). Still upstream-only in this project. |
| P3 — configured variants | GIE/environment seed, grid, Cartesian, depth-attention and optional conditioning tiles | Reuse edge/node/segment/GEMM primitives for GIE and FiLM; add S2/SO3 project–grid-op–integrate, Cartesian products, depth attention, SFPG, spin/charge and pair-potential branches only when enabled by the target checkpoint. | Source operators are identified in the operator map above; none are a complete project HIP path. These are optional branches, not prerequisites for claiming the base descriptor works. |

**Backward scope:** this project targets E/F/V inference, so every native replacement on the energy path must preserve `dE/d(edge_vec)` (or intentionally fall back to upstream autograd). Training with force loss additionally needs parameter gradients through forces and higher-order derivatives; that is a separate scope, not implied by a forward tile or by the existing primitive VJPs. For multi-rank LAMMPS, MPI `border_op` remains an explicit host/runtime boundary between blocks, with its own reverse-communication backward; it is not a tile collective and must not be folded into a CTA or persistent worker.

**Bring-up order:** first establish masked/tail-safe layout, loads, reductions, and tile MMA; then validate fused edge features and base node/edge projections; next implement rotations, SO(2) messages, and destination aggregation; finally connect readout plus the full edge-vector VJP. At each stage compare upstream forward and force/VJP, then benchmark the primitive and full model separately. Existing operator inventory in [`catalog.py`](../python/kernels/catalog.py) describes operator-level contracts; it is not an executable tile schedule or evidence that those kernels exist.

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
