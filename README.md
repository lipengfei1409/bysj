# DCU DPA4 GEMM

The current production GEMM operator is a contiguous, 2D FP32 `A[M,K] @ B[K,N]` implemented with DTK DUMMA. The native path is DCU-only and has no CPU, rocBLAS, or scalar-kernel fallback. A separate PyTorch oracle lives in `python/reference/` for explicit tests and comparisons. For DPA4/SeZM inference, `python/model/dpa4.py` adapts DeepMD-kit rather than reimplementing its model graph; DeepMD owns checkpoint loading, neighbor-list handling, energy, forces, and virial, while selected standalone TileLang operators can be enabled in its `pt_expt` backend.

## Scope

- `python/kernels/`: production GEMM API and DCU extension checks.
- `python/kernels/tilelang_ops/`: optional standalone TileLang operators for DPA4/SeZM, each with a PyTorch reference and (where needed) autograd wrapper.
- `python/reference/`: explicit PyTorch GEMM oracle; not imported as a production fallback.
- `csrc/gemm/gemm.cpp`: DUMMA FP32 16x16x4 kernel with masked M/N/K tails.
- `include/`: `.h` headers for the four DUMMA API wrappers, manual-based MMA shape selectors, and shared tile/LDS/cooperative-load/synchronization helpers.
- `tests/`: Python contracts, host-side helper contracts, and opt-in DTK/DCU accuracy tests.
- `doc/`: GEMM kernel and tile/LDS execution notes.

See [DMK current kernels](doc/dpa4-kernels.md) for implemented kernels and TileLang inference-development status, and [DeepMD-kit DPA4 operators](doc/megakernel.md) for framework dataflow, operator definitions, fused CUDA-op precedents, and distributed halo exchange.

## Python API

```python
import torch
from kernels import gemm
from reference import gemm as reference_gemm

a = torch.randn(19, 7, device="cuda", dtype=torch.float32)
b = torch.randn(7, 23, device="cuda", dtype=torch.float32)
y = gemm(a, b)                 # requires a supported DCU and built DUMMA extension
expected = reference_gemm(a, b)  # explicit PyTorch oracle
```

The first operator contract requires two contiguous 2D FP32 tensors on the same supported DCU, with matching inner dimensions. It is forward-only and does not support batching, transposes, bias, alpha/beta/C, or autograd. The oracle is available through `PYTHONPATH=python`; `kernels.gemm` never calls it implicitly.

## TileLang operator development

The current model-integration milestone is single-card DPA4/SeZM inference with **standalone, non-fused TileLang operators**. Operators live in `python/kernels/tilelang_ops/`; their dispatch is reached through DeepMD-kit `pt_expt`, whose unported stages remain on ordinary PyTorch/kernels. The optional `DPA4Model` facade is in `python/model/dpa4.py`. Fusion, training, stress/virial-specific work, and multi-rank ghost/halo exchange remain outside this inference adapter.

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

The extension requires HIP-enabled PyTorch, the DTK HIP compiler, and the external `<du_mma.h>` header from a compatible DTK installation. The FP32 K=4 DUMMA shape is enabled for gfx926, gfx928, gfx936, and gfx938. `gemm.cpp` contains HIP syntax despite the `.cpp` suffix; setup compiles native `.cpp` files with `hipcc` (override with `DPA4_HIP_CXX`). The project does not vendor DUMMA.

```bash
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
DPA4_BUILD_HIP=1 DPA4_ENABLE_DUMMA=1 DPA4_HIP_CXX=hipcc MAX_JOBS=4 \
  python setup.py build_ext --inplace
python -m unittest discover -s tests/csrc -v
```

For the direct DUMMA instruction test, set `DPA4_RUN_INSTRUCTION_TESTS=1` and `DPA4_TEST_DUMMA=1` when running the C++ test suite. Native tests require a supported DCU; a non-DCU environment can run the Python oracle and host header tests but cannot verify the kernel.
