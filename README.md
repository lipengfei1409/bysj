# DCU DPA4 Native Operators

The optional HIP extension in this checkout builds standalone RMSNorm and GIE (`zonal_scatter`) operators for DCU. The C++/HIP DUMMA GEMM source is not present in `csrc`; the Python `kernels.gemm` wrapper, PyTorch oracle, and DUMMA comparison benchmark are retained, but this extension does not export `kernels._C.gemm`. For DPA4/SeZM inference, `python/model/dpa4.py` adapts DeepMD-kit rather than reimplementing its model graph; DeepMD owns checkpoint loading, neighbor-list handling, energy, forces, and virial, while selected TileLang operators or per-instance native GIE/RMSNorm patches can be enabled for eager `pt_expt` models.

## Scope

- `python/kernels/`: native RMSNorm/GIE APIs and the retained Python GEMM wrapper with DCU extension checks.
- `python/kernels/tilelang_ops/`: optional standalone TileLang operators for DPA4/SeZM, each with a PyTorch reference and (where needed) autograd wrapper.
- `python/reference/`: explicit PyTorch oracles, including the GEMM oracle; these are not imported as production fallbacks.
- `csrc/rmsnorm.cpp` and `csrc/gie.cpp`: native HIP operator sources included by the optional extension build.
- `tests/`: Python contracts, host-side helper contracts, and opt-in DTK/DCU accuracy tests.
- `doc/kernel-model.html`: architecture overview for the TileLang operators and DPA4 model adapter.

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

The current model-integration milestone is single-card DPA4/SeZM inference with **standalone, non-fused TileLang operators** and explicit native HIP GIE/RMSNorm patches. TileLang operators live in `python/kernels/tilelang_ops/`; native operators live in `csrc/` and `python/kernels/`. Dispatch is reached through DeepMD-kit `pt_expt`, whose unported stages remain on ordinary PyTorch/kernels. The optional `DPA4Model` facade is in `python/model/dpa4.py`. Fusion, training, stress/virial-specific work, and multi-rank ghost/halo exchange remain outside this inference adapter.

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

with DPA4Model(
    "water_pt_expt.pt",
    tilelang_ops=("radial_basis", "edge_geometry", "edge_degree"),
) as model:
    energy, force, virial = model.eval(coords, cells, atom_types)
```

Install DeepMD-kit separately and make this project's `python/` directory
importable (for example, put it first on `PYTHONPATH`). `tilelang_ops=None`
leaves the caller's `DP_TILELANG_*` environment unchanged; `tilelang_ops=()`
forces the DeepMD baseline; an explicit sequence enables just those operator
gates while DeepMD constructs the model. The `"dpa4"` selector sets the global
`DP_TILELANG_DPA4` fast path and is restricted by DeepMD to its supported water
configuration. For other DPA4 scenarios, prefer individual operator selectors;
unsupported operators/configurations fail through DeepMD rather than silently
pretending TileLang was used. TileLang dispatch requires DeepMD's `pt_expt`
backend; legacy `pt` checkpoints remain usable as a baseline but do not dispatch
TileLang. Use an eager `pt_expt` training checkpoint (typically `.pt`) to
select operators at load time; `.pt2`/`.pte` artifacts have their kernels
fixed at export time and should be loaded without TileLang selectors.

The standalone HIP/DCU GIE zonal-scatter and RMSNorm operators can also be
selected explicitly for an eager `pt_expt` `.pt` model. These patches are
per-model, disabled by default, and leave unsupported module inputs on
DeepMD's original implementation:

```python
with DPA4Model(
    "water_pt_expt.pt",
    native_ops=("gie", "rmsnorm"),
) as model:
    energy, force, virial = model.eval(coords, cells, atom_types)
```

Build the HIP extension and run on a supported DCU. These wrappers use PyTorch's
`device="cuda"` tensors on the HIP backend; they are not NVIDIA CUDA kernels.
Native post-load patching is not available for legacy `pt` or exported
`.pt2`/`.pte` artifacts. Only plain RMSNorm and the supported GIE scatter path
are replaced; other SeZM modules remain in DeepMD.

An optional integration test accepts a `.npz` containing `coords`, `atom_types`,
and optional `cells` arrays:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
DP_RUN_DPA4_MODEL_INTEGRATION=1 \
DP_DPA4_MODEL_FILE=/path/to/dpa4_pt_expt.pt \
DP_DPA4_INPUT_NPZ=/path/to/system.npz \
DP_DPA4_MODEL_TILELANG_OPS=radial_basis,edge_geometry \
python -m unittest tests.python.test_dpa4_model -v
```

## Build and test on DTK/DCU

Package builds compile the native RMSNorm and GIE HIP operators by default and require HIP-enabled PyTorch and the DTK HIP compiler. Setup compiles native `.cpp` files with `hipcc` (override with `HIPCXX`). The deleted DUMMA GEMM source is not part of this extension.

```bash
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
MAX_JOBS=4 \
  python setup.py build_ext --inplace
python -m unittest discover -s tests/csrc -v
```

Native tests require a supported DCU; a non-DCU environment can run the Python oracle and host header tests but cannot verify the kernels.
