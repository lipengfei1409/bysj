# Model adapters and model-level checks

## Separate family/task contracts

`python/model/inference/` provides a lazy family/task registry and runner lifecycle. It does not force atomistic DPA4 requests and text prompts into one input schema. DPA4-specific Python code lives under `python/model/dpa4/`; low-level primitives remain in `python/dcu_dpa4/`, `csrc/`, and `include/`. Legacy model import paths remain compatibility shims.

A new model family needs an explicit adapter. Registering a family name does not make arbitrary checkpoints or tasks compatible.

## DPA4 and DeePMD

For validated full E/F/V inference, `model.dpa4` wraps the installed DeePMD `DeepPot` path and preserves its checkpoint, topology, and model-specific logic. The independent `DPA4Model` is still experimental and reports `complete_model=False` until its checkpoint and E/F/V path are verified.

Configure the device before importing/building the DeePMD model:

```python
from dcu_dpa4.deepmd import configure_eager, install_operators
configure_eager("cpu")  # use "dcu" in a supported ROCm/DTK runtime
from deepmd.infer import DeepPot

model = DeepPot("model.pt", no_jit=True)
with install_operators(model.get_model(), backend="reference") as handle:
    energy, force, virial = model.eval(coord, cell, atype)
    print(handle.report())
```

`backend="hip"` requires the built native extension and currently supports only its listed FP32 inference primitives. Unsupported DeePMD modules retain their upstream implementations. Adapters are instance-scoped; they do not alter LAMMPS/Kokkos, `dp test`, exported models, ghost communication, or training/second-order gradients.

Example CLI:

```bash
PYTHONPATH=python python examples/dpa4/infer.py \
  --checkpoint /path/to/model.pt --system /path/to/deepmd-npy-system --device cpu
```

## Qwen text generation

The optional Transformers adapter uses `AutoTokenizer` and `AutoModelForCausalLM`. It is a text-generation adapter, not a serving runtime; no custom Qwen HIP kernel, continuous batching, paged KV cache, or SGLang/vLLM integration is provided here.

```bash
pip install -e '.[qwen]'
PYTHONPATH=python python examples/qwen3/infer.py \
  --family qwen3 --model-id Qwen/Qwen3-0.6B \
  --prompt 'Explain equivariant message passing.'
```

`trust_remote_code` is off by default. Qwen3.5 text generation works only if the installed Transformers version provides a compatible model class. Multimodal requests require a separate processor adapter.

## Accuracy and latency

The full-model tests are opt-in and use the OMol25 Neo checkpoint by default; override with `DPA4_MODEL_CHECKPOINT`. Accuracy compares the independent reference path with DeePMD/DeepPot on the same checkpoint and a deterministic H₂O-shaped smoke input. This is **implementation parity**, not chemical accuracy against OMol25 labels.

```bash
DPA4_RUN_MODEL_ACCURACY_TESTS=1 PYTHONPATH=python:. \
  python -m unittest tests.model.test_accuracy -v
```

Optional finite-difference force check:

```bash
DPA4_RUN_MODEL_ACCURACY_TESTS=1 DPA4_RUN_MODEL_FINITE_DIFFERENCE=1 \
  PYTHONPATH=python:. python -m unittest tests.model.test_accuracy -v
```

The DCU-only latency benchmark measures full `DPA4Model` calls; it is not a primitive timing and does not exercise the persistent worker as a full-model executor:

```bash
DPA4_RUN_MODEL_LATENCY_TESTS=1 PYTHONPATH=python:. \
  python -m tests.model.benchmark_latency --atoms 32 --warmup 5 --batches 9 --iterations 5
```

Run accuracy before interpreting latency. Opt-in tests that skip or have not been run provide no correctness or performance evidence. The current worker only covers selected edge-feature tasks; all full-checkpoint, VJP, DCU and end-to-end speed results still require the target environment.
