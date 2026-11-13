# DCU DPA4 Native Operators

The optional HIP extension in this checkout builds standalone RMSNorm, GIE (`zonal_scatter`), and SO(2) envelope-reduction operators for DCU. The C++/HIP DUMMA GEMM source is not present in `csrc`; the Python `kernels.gemm` wrapper, PyTorch oracle, and DUMMA comparison benchmark are retained, but this extension does not export `kernels._C.gemm`. For DPA4/SeZM inference, `python/model/dpa4.py` adapts DeepMD-kit rather than reimplementing its model graph; DeepMD owns checkpoint loading, neighbor-list handling, energy, forces, and virial. The adapter patches GIE and RMSNorm with the existing HIP/DCU operators by default for eager `pt_expt` `.pt` models; TileLang remains standalone and is not dispatched by this adapter.

## Scope

- `python/kernels/`: native RMSNorm/GIE/SO(2) reduction APIs and the retained Python GEMM wrapper with DCU extension checks.
- `python/kernels/tilelang_ops/`: optional standalone TileLang operators for DPA4/SeZM, each with a PyTorch reference and (where needed) autograd wrapper.
- `python/reference/`: explicit PyTorch oracles, including the GEMM oracle; these are not imported as production fallbacks.
- `csrc/rmsnorm.cpp`, `csrc/gie.cpp`, and `csrc/so2_reduce.cpp`: native HIP operator sources included by the optional extension build.
- `tests/`: Python contracts, host-side helper contracts, and opt-in DTK/DCU accuracy tests.
- `doc/kernel-model.html`: architecture overview for standalone TileLang operators and the native GIE/RMSNorm DPA4 adapter.

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

## TileLang operator development

The DPA4/SeZM `DPA4Model` adapter uses native HIP GIE/RMSNorm by default for eager `pt_expt` `.pt` inference. The opt-in `so2_envelope_reduce` selector accelerates only the destination aggregation of attention-free SO2Convolution; edge geometry, rotations, radial mixing, and the SO(2) mixing stack remain in DeepMD. It does not dispatch TileLang operators; the standalone TileLang package remains available for separate kernel work. DeepMD-kit still owns the model graph and all unpatched stages. Fusion, training, higher-order force-loss gradients, stress/virial-specific kernels, and multi-rank ghost/halo exchange remain outside this initial native SO(2) slice.

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
primitive. For an attention-free configuration, the opt-in
`native_ops=("so2_envelope_reduce",)` accelerates only SO2Convolution's
native envelope destination-reduction seam; edge-message construction and the
rest of the convolution remain in DeepMD. The adapter does not mutate
`DP_TILELANG_*` environment variables;
use a DeepMD-kit checkout with the DPA4 TileLang dispatch removed. The legacy
`tilelang_ops` argument accepts only `None` or an empty iterable and rejects
non-empty selections. The standalone TileLang package remains available but is
not registered into the DeepMD DPA4 path.

Native patches require the HIP extension, a supported DCU, and an eager
`pt_expt` checkpoint ending in `.pt`; an unavailable extension, unsupported
backend, or missing target module raises clearly instead of silently falling
back for the whole model. These wrappers use PyTorch's `device="cuda"` tensors
on the HIP backend; they are not NVIDIA CUDA kernels. Native post-load patching
is not available for legacy `pt` or exported `.pt2`/`.pte` artifacts. Exported
artifacts keep their operator choices fixed at export; `native_ops=()` does not
rewrite an already-exported graph. The default replaces only plain RMSNorm and supported GIE scatter stages; the optional SO(2) selector replaces only the supported envelope-reduction seam. Other SeZM operations remain in DeepMD.

An optional integration test accepts a `.npz` containing `coords`, `atom_types`,
and optional `cells` arrays:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
DP_RUN_DPA4_MODEL_INTEGRATION=1 \
DP_DPA4_MODEL_FILE=/path/to/dpa4_pt_expt.pt \
DP_DPA4_INPUT_NPZ=/path/to/system.npz \
DP_DPA4_MODEL_NATIVE_OPS=gie,rmsnorm \
DP_DPA4_MODEL_EXTRA_NATIVE_OPS=so2_envelope_reduce \
python -m unittest tests.python.test_dpa4_model -v
```

## Build and test on DTK/DCU

Package builds compile native RMSNorm, GIE, and SO(2) envelope-reduction HIP operators by default and require HIP-enabled PyTorch and the DTK HIP compiler. Setup compiles native `.cpp` files with `hipcc` (override with `HIPCXX`). The deleted DUMMA GEMM source is not part of this extension.

```bash
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
MAX_JOBS=4 \
  python setup.py build_ext --inplace
python -m unittest discover -s tests/csrc -v
```

Native tests require a supported DCU; a non-DCU environment can run the Python oracle and host header tests but cannot verify the kernels.

## DPA4 hotspot profiling

The three supported scenarios (single-card DeepMD inference, single-card LAMMPS,
and single-host multi-card LAMMPS) live in [tests/deepmd/profile](tests/deepmd/profile/README.md).
These profiles use original DeepMD operators. Direct inference calls
`deepmd.infer.DeepPot` without the project adapter or native HIP patches, and
remaps the bundled water types using the model type map. It defaults to
periodic scaling within 70% of available device
memory and five warmup calls. Multi-card LAMMPS accepts `--world-size` matching
`--gpu-ids`; its finite Kineto window requires the updated DeepMD PTExpt backend.
