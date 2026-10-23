# Generic model inference framework

## Package boundaries

`python/model/inference/` supplies a lazy family/task registry, a common runner lifecycle, and adapter/request protocols. It deliberately does not force different model families into one tensor contract.

DPA4-specific Python model, checkpoint, DeepMD runner, system, neighbor bridge, and stage code now live under `python/model/dpa4/` and import through `model.dpa4`. The old `model.modeling`, `model.runner`, `model.system`, `model.checkpoint`, `model.neighbor_bridge`, and `model.stages` imports remain compatibility shims. The lower-level HIP/reference primitives and compiled extension stay in `python/dcu_dpa4/`, `csrc/`, and `include/`.

The registry supports multiple models through explicit adapters. It does not automatically make arbitrary checkpoints compatible; a new family/task needs an adapter that implements its own input, output, and lifecycle contract.

## Generic API

```python
from model import create_runner
from model.dpa4.system import load_deepmd_system

system = load_deepmd_system("/path/to/deepmd-npy-system")
with create_runner(
    "dpa4", "energy-force-virial",
    checkpoint="/path/to/model.pt",
    device="cpu",
) as runner:
    prediction = runner.predict(system)
    print(prediction.energy.shape, prediction.force.shape, prediction.virial.shape)
```

`model` and `model.inference` imports do not load Torch, DeePMD, or Transformers. Adapters are loaded only when `create_runner` selects a family/task. Third-party families can be registered with a factory or lazy `module:attribute` entrypoint.

## DPA4 example

The DPA4 adapter wraps the existing DeePMD `DeepPot` path for complete E/F/V inference. The separate project-owned `DPA4Model` remains experimental and continues to report `complete_model=False` until full-checkpoint E/F/V validation.

```bash
PYTHONPATH=python python examples/dpa4/infer.py \
  --checkpoint ../DPA4-OMol/DPA4-Neo-OMol25-100M-v20260820.pt \
  --system /path/to/deepmd-npy-system --device cpu
```

## Qwen examples

Install the optional Transformers extra in an environment with a compatible PyTorch build:

```bash
pip install -e '.[qwen]'
PYTHONPATH=python python examples/qwen3/infer.py \
  --family qwen3 --model-id Qwen/Qwen3-0.6B \
  --prompt 'Explain equivariant message passing.'
```

The built-in adapter is for text causal-LM generation using `AutoTokenizer` and `AutoModelForCausalLM`. `trust_remote_code` defaults to false. Qwen3.5 can be requested with `--family qwen3.5 --model-id ...` only when the installed Transformers version recognizes a compatible text-generation class. Multimodal inputs require a separate processor adapter and are rejected by this text-only path. The local Mirage registration alone is not evidence that a Qwen3.5 checkpoint works end-to-end.

`examples/qwen3/infer.py --help` does not import Transformers or download weights; actual inference loads the requested local/HuggingFace model explicitly.
