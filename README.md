# DCU DPA4 Native Operators

The optional HIP extension in this checkout builds standalone RMSNorm, GIE (`zonal_scatter`) and the fused segmented operators (`segment_sum`, `segment_softmax`, `force_assembly`) for DCU. For DPA4/SeZM inference, `python/model/dpa4.py` adapts DeepMD-kit rather than reimplementing its model graph; DeepMD owns checkpoint loading, neighbor-list handling, energy, forces, and virial. The adapter patches GIE and RMSNorm with the existing HIP/DCU operators by default for eager `pt_expt` `.pt` models, and can bind the fused segmented operators with `fused=`. TileLang has been removed from this project.

## Scope

- `python/kernels/`: native RMSNorm/GIE/fused-segment/slot-axis APIs with DCU extension checks.
- `python/reference/`: explicit PyTorch oracles; these are not imported as production fallbacks.
- `csrc/rmsnorm.cpp`, `csrc/gie.cpp`, `csrc/segment_sum.cpp`, `csrc/segment_softmax.cpp`, `csrc/force_assembly.cpp`, `csrc/slot_ops.cpp` and `csrc/edge_combine.cpp`: native HIP operator sources included by the optional extension build.
- `tests/`: Python contracts, host-side helper contracts, and opt-in DTK/DCU accuracy tests.
- `docs/operators.md`: 逐算子的作用、调用点、时间占比与有效性（含验证阶梯）。
- `docs/deepmd-fusions.md`: deepmd-kit 上游已有的算子融合清单（融合了什么、省多少访存、收益来源、调用点与开关）。
- `tests/deepmd/profile/fx_graph.py`: 打印模型/子模块的 FX 图（`--target model|so2_conv`，CPU 或 DCU 追踪）。
- `python/dmk/`（`config.py` / `shard.py` / `main.py`）: 持久化调度的主机侧——每 CU 的 tile 配置、按 CU 切分计算图、构建数据流并 launch；见 [docs/dmk-host.md](docs/dmk-host.md)。
- `docs/so2-conv-hip.md`: 把上游 fused SO(2) 卷积移植到 HIP 的方案、目录、翻译映射与验证步骤；`DPA4Model(so2_conv="hip")` 或 profile CLI 的 `--so2-conv hip` 启用。
- `docs/optimization-status.md`: 优化项状态表（每个优化项解决什么、方案、完成度、预期收益）。

## Python API

Each operator exposes a thin DCU wrapper next to a pure PyTorch oracle, so the
same shape can be checked on a CPU host:

```python
import torch
from kernels import segment_sum, segment_offsets
from reference import segment_sum_reference

src = torch.randn(6, 4, device="cuda", dtype=torch.float32)
dst = torch.tensor([0, 0, 1, 2, 2, 2], device="cuda")
offsets = segment_offsets(dst, 3)
y = segment_sum(src, dst, offsets, None, None)          # requires the HIP extension
expected = segment_sum_reference(src, dst, offsets)     # explicit oracle
```

The same pairing exists for GIE (`kernels.gie` / `reference.gie`), RMSNorm
(`kernels.rmsnorm` / `reference.rmsnorm`) and the softmax/force seals. The
wrappers validate dtype, shape, device and contiguity before touching the DCU
and never fall back to the oracle implicitly.

## Adapter scope

The DPA4/SeZM `DPA4Model` adapter is a pass-through facade unless `fused=` is given, in which case it binds the fused segmented operators (`segment_sum`, `segment_softmax`) into DeepMD's own dispatch seams -- including the destination aggregation of attention-free SO2Convolution, which the removed `so2_envelope_reduce` selector used to cover; see [docs/operators.md](docs/operators.md). Edge geometry, rotations, radial mixing, the SO(2) mixing stack and the force assembly inside autograd remain in DeepMD. Training, higher-order force-loss gradients, stress/virial-specific kernels, and multi-rank ghost/halo exchange remain outside this native slice.

Run the Python contract and reference tests:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests/python -v
```

## DeepMD DPA4 model adapter

`model.DPA4Model` delegates model construction and inference to DeepMD-kit, so
DPA4/SeZM model configurations supported by DeepMD do not need to be re-created
in this repository. It forwards DeepMD's public `DeepPot.eval` inputs and outputs
(energy, force, virial, and optional atomic outputs).

```python
from model import DPA4Model

with DPA4Model("water.pt") as model:            # stock DeepMD, untouched
    energy, force, virial = model.eval(coords, cells, atom_types)

with DPA4Model("water.pt", fused=True) as fused:  # bind the fused seams
    energy, force, virial = fused.eval(coords, cells, atom_types)
    print(fused.fused_ops)                       # hits / fallbacks / bound flags
```

Install DeepMD-kit separately and make this project's `python/` directory
importable (for example, put it first on `PYTHONPATH`). Without `fused=`, the
facade is a pass-through and DeepMD runs its own operators. `fused=` installs
this project's fused segmented operators into DeepMD's own dispatch seams via
`model.fused_deepmd`, which keeps a per-call fallback to the stock
implementation; `fused=("softmax",)` or `("envelope",)` select a single seam.
Neither the legacy `tilelang_ops` nor the removed `native_ops` selector is
accepted any more -- both raise with an explicit message instead of being
forwarded to DeepMD.

The fused seams require the HIP extension and a supported DCU; without them
`deepmd_patch` binds nothing and reports `bound_*=false`, so the run is simply
the DeepMD baseline (no silent partial state). These wrappers use PyTorch's
`device="cuda"` tensors on the HIP backend; they are not NVIDIA CUDA kernels.
Edge geometry, rotations, radial mixing, the SO(2) mixing stack and the force
assembly inside ``autograd`` still run in DeepMD; see
[docs/operators.md](docs/operators.md) for the exact
seams and the three sites that remain on DeepMD's path.

An optional integration test accepts a `.npz` containing `coords`, `atom_types`,
and optional `cells` arrays:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
DP_RUN_DPA4_MODEL_INTEGRATION=1 \
DP_DPA4_MODEL_FILE=/path/to/dpa4_pt_expt.pt \
DP_DPA4_INPUT_NPZ=/path/to/system.npz \
DP_DPA4_MODEL_NATIVE_OPS=gie,rmsnorm \
python -m unittest tests.python.test_dpa4_model -v
```

## Build and test on DTK/DCU

Package builds compile native RMSNorm, GIE and the fused segmented HIP operators by default and require HIP-enabled PyTorch and the DTK HIP compiler. Setup compiles native `.cpp` files with `hipcc` (override with `HIPCXX`). The deleted DUMMA GEMM source is not part of this extension.

```bash
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
MAX_JOBS=4 \
  python setup.py build_ext --inplace
python -m unittest discover -s tests/csrc -v
```

Native tests require a supported DCU; a non-DCU environment can run the Python oracle and host header tests but cannot verify the kernels.

## DCU megakernel LDS management

The DCU megakernel follows the HazyResearch Megakernels structure: a static LDS region holds `config::kStages` instruction pipeline slots (`InstructionState`: 32 instruction words, `pid_order`, per-instruction semaphores, scratch), while dynamic LDS is carved into `config::kNumPages` fixed 8 KiB pages. A controller wave fetches each instruction, permutes logical page ids into physical ids from the previous op's `release_lid` mapping, constructs the op's dynamic semaphores, and publishes the instruction. Consumer/loader/storer waves dispatch on the opcode and use per-page parity semaphores (`lds::Semaphore`, an mbarrier-compatible replacement) for backpressure. The 64 KiB DCU budget supports 6 pages: 10,448 B static + 49,152 B dynamic + 128 B alignment = 59,728 B, checked by `static_assert` and again at launch.

`csrc/dmk/config.h`, `lds.h`, `state.h`, `op.h`, `controller.h`, `noop.h`, `demo_op.h`, `gemm.h` and `embedding.h` implement the framework and concrete ops. Every op derives from `BaseOp<Globals>` (`op.h`) and implements `release_lid`, `init_semaphores`, `load`, `run` and `store`: the three data hooks describe one whole operator and know nothing about roles (load stages its first tile, run executes the operator's own tile loop, store writes the result out), the worker waves map `Role::Loader`/`Role::Consumer`/`Role::Storer` onto them, and the controller constructs the op for each instruction and dispatches through the virtual interface. `csrc/dmk/megakernel.h` and `csrc/megakernel.cpp` wire up the demo kernel (`NoOp`, `ElementwiseOp`, `GemmOp`, `EmbeddingOp`). Data movement lives in each op's own hooks (one instruction is one whole operator: which vectors are copied, the tail and the alignment fallback are op-specific), on top of the `copy::load_vec`/`copy::store_vec`/`copy::aligned` primitives in `csrc/dmk/copy.h`. The block runs four role classes (controller + loader + storer + 4 consumer waves, 448 threads); the Blackwell reference launcher wave is intentionally omitted because DCU DUMMA is wave-synchronous. The scaffold is not part of the `kernels._C` extension yet; the test runner compiles `csrc/megakernel.cpp` directly.

```bash
source /opt/dtk/env.sh
python tests/csrc/run_dmk_lds_test.py            # host logic + DCU device test
python tests/csrc/run_dmk_lds_test.py --host-only
```

Set `DMK_GPU_ARCH=gfx928` if the architecture cannot be detected through Torch. Design notes, page/semaphore protocol, the op interface and the H100-to-DCU mapping are in [docs/lds-management.md](docs/lds-management.md).

## Standalone GEMM and attention operators

Each operator can also be built and run on its own, before anything is fused into a megakernel: one kernel launch per operator, compared against a double-precision CPU reference.

| Operator | Entry point | Test |
| --- | --- | --- |
| GEMM `C = A * B` | `dmk::gemm::run_gemm_wave<T, K>` ([gemm.h](csrc/dmk/gemm.h)) | [test_dmk_gemm_standalone.cpp](tests/csrc/test_dmk_gemm_standalone.cpp) |
| Attention `softmax(Q K^T / sqrt(D)) V` | `dmk::mk::AttentionOp<Globals>` ([attention.h](csrc/dmk/attention.h)) | [test_dmk_attention.cpp](tests/csrc/test_dmk_attention.cpp) |

GEMM covers the DUMMA tile shapes from the manual (表 3.1), templated on the input dtype and the tile K: `float` m16n16k4 on every DCU, `float` m16n16k8 and `__half`/`__hip_bfloat16` m16n16k16 on gfx928/936/938. The accumulator is always `float`. The test runner enables the pairs its `--offload-arch` supports. Attention is a page-based op instead of a free kernel: one instruction is one whole query tile over all keys, so the loader wave stages Q and the first K/V tile while the consumer wave walks the remaining key tiles inside run(). The score matrix never reaches HBM. The element type is templated the same way (float, half, bfloat16 in and out, arithmetic in float). Both score tiles are 16 wide, exactly one DUMMA m16n16 tile, so the scalar QK^T and P*V loops can be replaced by fragments once the accumulator-rescaling question is answered on hardware. Operands are addressed as tiles: `csrc/dmk/tile.h` provides `TileShape`, `TileView`, `make_tile`/`make_tile_at`, `load_tile` and `copy_tile`, so a tile is a logical block (`tile_m x tile_n`, or `tile_b x tile_m x tile_n` for a rank-3 view), each operator picks its own tile extents (GEMM 16x16, attention 16xD), and one call moves exactly one tile with the shape clamped at the tensor edge.

```bash
python tests/csrc/run_dmk_ops_test.py                 # both, each in its own executable
python tests/csrc/run_dmk_ops_test.py --op gemm
python tests/csrc/run_dmk_ops_test.py --op attention
```

## DPA4 hotspot profiling

The three supported scenarios (single-card DeepMD inference, single-card LAMMPS,
and single-host multi-card LAMMPS) live in [tests/deepmd/profile](tests/deepmd/profile/README.md).
These profiles use original DeepMD operators. Direct inference calls
`deepmd.infer.DeepPot` without the project adapter or native HIP patches, and
remaps the bundled water types using the model type map. It defaults to
periodic scaling within 70% of available device
memory and five warmup calls. Multi-card LAMMPS accepts `--world-size` matching
`--gpu-ids`; its finite Kineto window requires the updated DeepMD PTExpt backend.
