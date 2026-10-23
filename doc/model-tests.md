# Full-model accuracy and latency tests

These checks target the **DPA4 model**, not individual primitive kernels. They use the OMol25 Neo checkpoint by default:

```text
../DPA4-OMol/DPA4-Neo-OMol25-100M-v20260820.pt
```

Override the path with `DPA4_MODEL_CHECKPOINT`. The repository currently has no OMol25 evaluation-system fixture, so the accuracy test uses a deterministic, non-periodic H2O-shaped input and compares the independent model with DeePMD/DeepPot using the **same checkpoint**. This is numerical implementation parity, not chemical accuracy against OMol25 labels. A real labeled evaluation set should be added when available.

The shared input helper makes a complete directed graph for the tiny non-periodic structure, maps `O`/`H` by symbol into the checkpoint type map, and uses the checkpoint default charge/spin. This avoids calling the independent backend's unimplemented numerical `neighbor_graph` operator.

## Accuracy

Requires Python ≥3.10, PyTorch, DeePMD-kit with compiled libraries, and the Neo checkpoint. The suite is opt-in so ordinary test discovery does not need those external dependencies. Once opted in, missing dependencies or checkpoint files fail rather than silently skip.

```bash
DPA4_RUN_MODEL_ACCURACY_TESTS=1 \
PYTHONPATH=python:. \
python -m unittest tests.model.test_accuracy -v
```

It compares energy `[F,1]`, force `[F,N,3]`, and virial `[F,3,3]` between `DPA4Model.from_checkpoint(..., primitive_backend="reference")` and `DPA4Runner`/DeepPot. Initial parity budgets are `rtol=2e-4`, energy `atol=1e-4 eV`, force `atol=2e-4 eV/Å`, and virial `atol=2e-3 eV`; override the per-output absolute budgets with `DPA4_MODEL_ENERGY_ATOL`, `DPA4_MODEL_FORCE_ATOL`, and `DPA4_MODEL_VIRIAL_ATOL`, or the shared relative budget with `DPA4_MODEL_RTOL`. These are provisional gates until a real Neo run establishes appropriate numerical tolerances.

An optional central-difference force check is controlled separately:

```bash
DPA4_RUN_MODEL_ACCURACY_TESTS=1 DPA4_RUN_MODEL_FINITE_DIFFERENCE=1 \
PYTHONPATH=python:. python -m unittest tests.model.test_accuracy -v
```

## Latency

The benchmark measures full `DPA4Model` calls on a deterministic synthetic, non-periodic graph. Model/checkpoint loading, graph construction, and host-to-device transfer happen before timing. It reports energy-only and E/F/V calls separately, with warm-up and current-stream device events. It has no machine-dependent speed threshold; it reports median, p90, min/max, and individual batch samples. This measures the current full-model launch path, **not** the persistent edge-feature frontend in isolation and not a full persistent MegaKernel.

Build the HIP extension on a DTK/DCU system, then run:

```bash
source /opt/dtk/env.sh
DPA4_BUILD_HIP=1 MAX_JOBS=4 python setup.py build_ext --inplace
DPA4_RUN_MODEL_LATENCY_TESTS=1 PYTHONPATH=python:. \
python -m tests.model.benchmark_latency --atoms 32 --warmup 5 --batches 9 --iterations 5
```

The benchmark requires `torch.version.hip`, a working `dcu_dpa4._C`, and the checkpoint. `--atoms` changes the synthetic workload; the script records actual atom and edge counts, model/checkpoint, device, torch/HIP versions, and timing scope in JSON on stdout. For representative production latency, rerun with the target workload's atom/edge distribution; synthetic numbers alone are not a performance claim.

## Interpretation boundary

- A skipped opt-in test is **not** a pass.
- Accuracy compares two implementations of the same model/checkpoint; it does not establish agreement with experimental labels.
- Latency is only meaningful after the full E/F/V accuracy gate passes on the same DCU.
- The current persistent runtime supports only selected edge-feature tasks / the geometry→radial pair. It is not the full DPA4 graph; do not label this benchmark as persistent-MegaKernel latency.
