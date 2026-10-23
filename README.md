# DCU DPA4

DPA4 operators and an experimental DPA4-Neo OMol25 model path for ROCm/DTK. The project also has a generic Python model-adapter framework and optional HuggingFace text generation. The independent DPA4 model is not yet fully validated (`complete_model=False`); no complete DPA4 or Qwen MegaKernel speedup is claimed.

## Layout

```text
include/           DTK/HIP tile, wave, sync, DUMMA, and BLAS interfaces
csrc/              HIP primitives, persistent edge-feature worker, pybind bindings
python/dcu_dpa4/   reference/HIP operators, autograd, DeePMD adapters
python/model/      generic inference API and canonical model/dpa4 package
examples/          separate DPA4 E/F/V and Qwen text-generation entry points
tests/             Python, native-operator, and opt-in full-model suites
doc/               two canonical guides
```

- [DPA4 kernels and DTK design](doc/dpa4-kernels.md): operator contracts, ThunderKittens translation boundaries, DUMMA/linear path, build and native verification.
- [Model adapters and tests](doc/models.md): DPA4/DeePMD, Qwen text adapter, model-level accuracy and latency.

## Quick start

Pure-Python model/framework checks (requires the project's Python dependencies):

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests/python -v
python -m model operators
```

DPA4 E/F/V through an installed DeePMD runtime:

```bash
PYTHONPATH=python python examples/dpa4/infer.py \
  --checkpoint /path/to/model.pt --system /path/to/deepmd-npy-system --device cpu
```

Optional Qwen3 text generation:

```bash
pip install -e '.[qwen]'
PYTHONPATH=python python examples/qwen3/infer.py \
  --family qwen3 --model-id Qwen/Qwen3-0.6B \
  --prompt 'Explain equivariant message passing.'
```

Qwen3.5 is conditional on the installed Transformers model class and the included adapter supports text only. It is not a Qwen serving backend.

## Build on DTK/DCU

Use an existing DTK PyTorch environment and its installed toolkit; this project does not replace system dependencies:

```bash
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
DPA4_BUILD_HIP=1 MAX_JOBS=4 python setup.py build_ext --inplace
python -c 'from dcu_dpa4.native import native_available; assert native_available()'
python -m unittest discover -s tests/csrc -v
```

The DUMMA-tiled FP32 linear path is opt-in and requires a supported gfx target with `du_mma.h`:

```bash
DPA4_BUILD_HIP=1 DPA4_ENABLE_DUMMA=1 MAX_JOBS=4 \
  python setup.py build_ext --inplace
```

rocBLAS is a separate optional host-launched path (`DPA4_BUILD_HIP=1 DPA4_BUILD_ROCBLAS=1`). It is not called inside the persistent device worker.

## Current implementation boundary

The current HIP extension includes geometry, radial features, edge-feature fusion, ordinary RMSNorm, linear, segment sum and force/virial assembly. `DPA4Model` still has incomplete checkpoint/VJP/native coverage. The persistent worker dispatches only fused edge features or `edge_geometry → radial_basis`; the static task planner is not a full-model executor. The newly added DUMMA linear path, native runtime, model accuracy and DCU timing require compilation and verification on the target device before use or performance claims.
