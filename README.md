# DCU DPA4 Native Operators

The optional HIP extension in this checkout builds standalone RMSNorm, GIE (`zonal_scatter`) and the fused segmented operators (`segment_sum`, `segment_softmax`, `force_assembly`) for DCU. The C++/HIP DUMMA GEMM source is not present in `csrc`; the Python `kernels.gemm` wrapper, PyTorch oracle, and DUMMA comparison benchmark are retained, but this extension does not export `kernels._C.gemm`. For DPA4/SeZM inference, `python/model/dpa4.py` adapts DeepMD-kit rather than reimplementing its model graph; DeepMD owns checkpoint loading, neighbor-list handling, energy, forces, and virial. The adapter patches GIE and RMSNorm with the existing HIP/DCU operators by default for eager `pt_expt` `.pt` models, and can bind the fused segmented operators with `fused=`. TileLang has been removed from this project.

## Scope

- `python/kernels/`: native RMSNorm/GIE/fused-segment APIs and the retained Python GEMM wrapper with DCU extension checks.
- `python/reference/`: explicit PyTorch oracles, including the GEMM oracle; these are not imported as production fallbacks.
- `csrc/rmsnorm.cpp`, `csrc/gie.cpp`, `csrc/segment_sum.cpp`, `csrc/segment_softmax.cpp` and `csrc/force_assembly.cpp`: native HIP operator sources included by the optional extension build.
- `tests/`: Python contracts, host-side helper contracts, and opt-in DTK/DCU accuracy tests.
- `docs/kernel-model.html`: architecture overview for the native GIE/RMSNorm operators and the DeepMD adapter.

## Python API

```python
import torch
from kernels import gemm
from reference import gemm as reference_gemm

a = torch.randn(19, 7, device="cuda", dtype=torch.float32)
b = torch.randn(7, 23, device="cuda", dtype=torch.float32)
y = gemm(a, b)                 # requires a separately provided kernels._C.gemm entry point
expected = reference_gemm(a, b)  # explicit PyTorch oracle
```

The retained GEMM wrapper requires two contiguous 2D FP32 tensors on the same supported DCU, with matching inner dimensions. It is forward-only and does not support batching, transposes, bias, alpha/beta/C, or autograd. The oracle is available through `PYTHONPATH=python`; `kernels.gemm` never calls it implicitly. The local HIP extension build below does not provide the GEMM entry point.

## Adapter scope

The DPA4/SeZM `DPA4Model` adapter uses native HIP GIE/RMSNorm by default for eager `pt_expt` `.pt` inference. `fused=` additionally binds the fused segmented operators (`segment_sum`, `segment_softmax`) into DeepMD's own dispatch seams -- including the destination aggregation of attention-free SO2Convolution, which the removed `so2_envelope_reduce` selector used to cover; see [docs/deepmd-fused-dispatch.md](docs/deepmd-fused-dispatch.md). Edge geometry, rotations, radial mixing, and the SO(2) mixing stack remain in DeepMD. DeepMD-kit still owns the model graph and all unpatched stages. Training, higher-order force-loss gradients, stress/virial-specific kernels, and multi-rank ghost/halo exchange remain outside this native slice.

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

with DPA4Model("water_pt_expt.pt") as model:  # native GIE + RMSNorm by default
    energy, force, virial = model.eval(coords, cells, atom_types)

# Explicit DeepMD baseline, without native patches:
with DPA4Model("water_pt_expt.pt", native_ops=()) as baseline:
    energy, force, virial = baseline.eval(coords, cells, atom_types)
```

Install DeepMD-kit separately and make this project's `python/` directory
importable (for example, put it first on `PYTHONPATH`). By default, `DPA4Model`
patches matching GIE and plain RMSNorm modules per model instance using the
existing native HIP/DCU operators. Unsupported individual inputs retain
DeepMD's original implementation. Pass `native_ops=()` to explicitly use the
DeepMD baseline, or pass `native_ops=("gie",)` / `("rmsnorm",)` to select one
primitive. For an attention-free configuration, `fused=("envelope",)`
accelerates SO2Convolution's envelope destination-reduction seam; edge-message
construction and the rest of the convolution remain in DeepMD. TileLang dispatch was removed from
this project, so the legacy `tilelang_ops` argument now raises with an explicit
message instead of being forwarded to DeepMD.

Native patches require the HIP extension, a supported DCU, and an eager
`pt_expt` checkpoint ending in `.pt`; an unavailable extension, unsupported
backend, or missing target module raises clearly instead of silently falling
back for the whole model. These wrappers use PyTorch's `device="cuda"` tensors
on the HIP backend; they are not NVIDIA CUDA kernels. Native post-load patching
is not available for legacy `pt` or exported `.pt2`/`.pte` artifacts. Exported
artifacts keep their operator choices fixed at export; `native_ops=()` does not
rewrite an already-exported graph. The default replaces only plain RMSNorm and supported GIE scatter stages; the optional `fused` switch adds the fused segment operators, including the SO(2) envelope reduction. Other SeZM operations remain in DeepMD.

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

The DCU megakernel follows the HazyResearch Megakernels structure: a static LDS region holds `Config::kStages` instruction pipeline slots (`InstructionState`: 32 instruction words, `pid_order`, per-instruction semaphores, scratch), while dynamic LDS is carved into `Config::kNumPages` fixed 8 KiB pages. A controller wave fetches each instruction, permutes logical page ids into physical ids from the previous op's `release_lid` mapping, constructs the op's dynamic semaphores, and publishes the instruction. Consumer/loader/storer waves dispatch on the opcode and use per-page parity semaphores (`lds::Semaphore`, an mbarrier-compatible replacement) for backpressure. The 64 KiB DCU budget supports 6 pages: 10,416 B static + 49,152 B dynamic + 128 B alignment = 59,696 B, checked by `static_assert` and again at launch.

`csrc/dmk/config.h`, `lds.h`, `state.h`, `op.h`, `controller.h`, `noop.h`, `demo_op.h`, `gemm.h` and `embedding.h` implement the framework and concrete ops. Every op derives from `BaseOp<Config, Globals>` (`op.h`) and implements `release_lid`, `init_semaphores`, `loader`, `consumer`, `storer`; the controller and worker waves construct the op for each instruction and dispatch through the virtual interface. `csrc/dmk/megakernel.h` and `csrc/megakernel.cpp` wire up the demo kernel (`NoOp`, `ElementwiseOp`, `GemmOp`, `EmbeddingOp`). Data movement uses the vectorized HBM/LDS helpers in `csrc/dmk/copy.h` (`float4`/`float2` with alignment-checked scalar fallback and coalesced lane-to-vector mapping). The block runs four role classes (controller + loader + storer + 4 consumer waves, 448 threads); the Blackwell reference launcher wave is intentionally omitted because DCU DUMMA is wave-synchronous. The scaffold is not part of the `kernels._C` extension yet; the test runner compiles `csrc/megakernel.cpp` directly.

```bash
source /opt/dtk/env.sh
python tests/csrc/run_dmk_lds_test.py            # host logic + DCU device test
python tests/csrc/run_dmk_lds_test.py --host-only
```

Set `DMK_GPU_ARCH=gfx928` if the architecture cannot be detected through Torch. Design notes, page/semaphore protocol, the op interface and the H100-to-DCU mapping are in [docs/lds-management.md](docs/lds-management.md).

## DPA4 hotspot profiling

The three supported scenarios (single-card DeepMD inference, single-card LAMMPS,
and single-host multi-card LAMMPS) live in [tests/deepmd/profile](tests/deepmd/profile/README.md).
These profiles use original DeepMD operators. Direct inference calls
`deepmd.infer.DeepPot` without the project adapter or native HIP patches, and
remaps the bundled water types using the model type map. It defaults to
periodic scaling within 70% of available device
memory and five warmup calls. Multi-card LAMMPS accepts `--world-size` matching
`--gpu-ids`; its finite Kineto window requires the updated DeepMD PTExpt backend.
