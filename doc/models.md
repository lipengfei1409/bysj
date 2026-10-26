# Model definitions and runtime

## Python package boundary

This repository is DCU-focused. The Python packages separate the model's dataflow from the code that loads and runs it; there is no `python/dcu_dpa4/` package.

| Package | Responsibility | Representative modules |
|---|---|---|
| `python/model/` (`import model`) | DPA4 model definitions, configuration, tensor contracts, parameterized components, and operator-level dataflow. No runner, CLI, checkpoint/system I/O, registry, or numerical execution backend. | `model/dpa4/modeling/{config,types,components,dpa4}.py` |
| `python/runtime/` (`import runtime`) | Model execution: generic task/family registry, runners, adapters, checkpoint/system loading, CLI, DeePMD integration, numerical backend, and megakernel planning/dispatch. | `runtime/inference/`, `runtime/dpa4/` |
| `python/kernels/` (`import kernels`) | Reusable operator APIs, reference implementations, HIP-extension loader, BLAS, and operator catalogue. | `kernels/{ops,geometry,radial,reduce,native,catalog}.py` |

The dependency direction is **runtime → model definitions and kernels**; model definitions depend on the operator/dataflow contract rather than a runner or concrete numerical backend. The compiled extension is `kernels._C`. The distribution remains named `dcu-dpa4`, which is separate from its Python import namespaces. Package discovery is configured in [`setup.py`](../setup.py).

## DPA4 and DeePMD

For established full E/F/V inference, use the runtime adapter around DeePMD's `DeepPot`. The independent `DPA4Model` in `model.dpa4.modeling` defines the project-owned dataflow but remains experimental and is not yet a validated replacement for the complete DeePMD model. Checkpoint loading is in `runtime.dpa4.loading`, not on the model class.

Configure the device before importing/building the DeePMD model:

```python
from runtime.dpa4.deepmd import configure_eager, install_operators
configure_eager("cpu")  # use "dcu" in a supported ROCm/DTK runtime
from deepmd.infer import DeepPot

model = DeepPot("model.pt", no_jit=True)
with install_operators(model.get_model(), backend="reference") as handle:
    energy, force, virial = model.eval(coord, cell, atype)
    print(handle.report())
```

Operator APIs and compiled-extension loading are in [`python/kernels`](../python/kernels/); instance-local DeePMD patching and eager configuration are in [`python/runtime/dpa4/deepmd`](../python/runtime/dpa4/deepmd/). `backend="hip"` requires the built native extension and supports only the listed FP32 inference primitives. Unsupported DeePMD modules retain their upstream implementations. The adapters do not alter LAMMPS/Kokkos, `dp test`, exported models, MPI ghost communication, or training/second-order gradients.

DPA4 E/F/V example:

```bash
PYTHONPATH=python python examples/dpa4/infer.py \
  --checkpoint /path/to/model.pt --system /path/to/deepmd-npy-system --device cpu
```

## Qwen text generation

The optional Transformers adapter is a runtime adapter, not a model definition or serving engine. It uses `AutoTokenizer` and `AutoModelForCausalLM`; no custom Qwen HIP kernel, continuous batching, paged KV cache, or SGLang/vLLM integration is provided.

```bash
pip install -e '.[qwen]'
PYTHONPATH=python python examples/qwen3/infer.py \
  --family qwen3 --model-id Qwen/Qwen3-0.6B \
  --prompt 'Explain equivariant message passing.'
```

`trust_remote_code` is off by default. Qwen3.5 text generation works only if the installed Transformers version provides a compatible model class. Multimodal requests require a separate runtime adapter.

## Accuracy and latency

The full-model tests are opt-in and use the OMol25 Neo checkpoint by default; override with `DPA4_MODEL_CHECKPOINT`. Accuracy compares the independent runtime path with DeePMD/DeepPot on the same checkpoint and a deterministic H₂O-shaped smoke input. This is **implementation parity**, not chemical accuracy against OMol25 labels.

```bash
DPA4_RUN_MODEL_ACCURACY_TESTS=1 PYTHONPATH=python:. \
  python -m unittest tests.model.test_accuracy -v
```

Optional finite-difference force check:

```bash
DPA4_RUN_MODEL_ACCURACY_TESTS=1 DPA4_RUN_MODEL_FINITE_DIFFERENCE=1 \
  PYTHONPATH=python:. python -m unittest tests.model.test_accuracy -v
```

The DCU-only latency benchmark measures complete independent `DPA4Model` calls; it does not exercise the persistent worker as a full-model executor:

```bash
DPA4_RUN_MODEL_LATENCY_TESTS=1 PYTHONPATH=python:. \
  python -m tests.model.benchmark_latency --atoms 32 --warmup 5 --batches 9 --iterations 5
```

Run accuracy before interpreting latency. Opt-in tests that skip or have not been run provide no correctness or performance evidence. The current worker covers only selected edge-feature tasks; full-checkpoint, VJP, DCU, and end-to-end speed results still require target-environment verification.
