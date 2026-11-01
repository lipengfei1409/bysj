# TileLang DPA4 operators

This package holds standalone TileLang operators for the single-card DPA4/SeZM
energy-and-force inference path. Operators are not fused. Each module owns its
forward kernel, backward kernel(s), `torch.autograd.Function`, and differentiable
PyTorch reference. The autograd functions save the inputs needed by backward
with `ctx.save_for_backward`.

## Operator modules

| Module | DPA4 stage | Backward inputs |
| --- | --- | --- |
| `radial_basis.py` | Raw Bessel radial basis | Distance; no frequency-parameter gradient |
| `cutoff_envelope.py` | C3 radial/edge envelope | Distance |
| `edge_geometry.py` | Regularized edge lengths and unit vectors | Edge vectors |
| `edge_quaternion.py` | Smooth two-chart edge-aligned quaternion | Edge vectors, optional cached lengths |
| `wigner_l1.py` | Packed l=0/l=1 Wigner-D pair | Quaternion |
| `zonal_scatter.py` | Geometric initial embedding scatter | Zonal coupling, radial features, degree normalization |
| `environment_matrix.py` | Build `[s, s*r_hat]` geometry channels | Edge vectors, envelope |
| `env_seed_scatter.py` | Environment-seed outer-product scatter | Geometry features, edge features, degree normalization |
| `env_seed_gram.py` | Invariant environment Gram projection | Aggregated environment matrix |
| `so2_rotate_to_local_l1.py` | Source gather and local-frame projection | Node features, Wigner matrix |
| `so2_radial_scale.py` | Per-degree local radial modulation | Local features, radial features |
| `so2_linear_l1.py` | Block-diagonal SO2Linear for lmax=mmax=1 | Local features; frozen weights |
| `so2_rotate_back_l1.py` | Local-to-global projection | Local features, Wigner matrix |
| `segment_softmax.py` | Envelope-gated destination segment attention | Logits, envelope |
| `attention_aggregate.py` | Attention-weighted destination aggregation | Edge values, attention weights, head gate |
| `grid_product.py` | Pointwise product of S2-projected grid fields | Both grid operands |

Generic neural-network affine layers, fitting, graph construction, and unsupported
DPA4 options continue through the existing PyTorch/DeepMD paths. The TileLang
SO(2) rotation/linear kernels currently target `lmax=mmax=1`; the first full
model target is the water configuration in
`deepmd-kit/examples/water/dpa4/lmp/input.json`. Spin, bridging, stress/virial,
MPI exchange, and training are not part of this opt-in path.

## Opt in

Set `DP_TILELANG_DPA4=1` to select all implemented standalone operators for the
supported water inference configuration, or set individual controls to test an
operator in isolation:

- `DP_TILELANG_RADIAL_BASIS=1`
- `DP_TILELANG_C3_ENVELOPE=1`
- `DP_TILELANG_EDGE_GEOMETRY=1`
- `DP_TILELANG_EDGE_QUATERNION=1`
- `DP_TILELANG_WIGNER_L1=1`
- `DP_TILELANG_ZONAL_SCATTER=1`
- `DP_TILELANG_ENV_SEED_MATRIX=1`
- `DP_TILELANG_ENV_SEED_SCATTER=1`
- `DP_TILELANG_ENV_SEED_GRAM=1`
- `DP_TILELANG_SO2_ROTATION_L1=1`
- `DP_TILELANG_SO2_RADIAL_SCALE=1`
- `DP_TILELANG_SO2_LINEAR_L1=1`
- `DP_TILELANG_ATTENTION_SOFTMAX=1`
- `DP_TILELANG_ATTENTION_AGGREGATE=1`
- `DP_TILELANG_GRID_PRODUCT=1`

The gates are resolved when the `pt_expt` modules are constructed. Enabled paths
are inference-only and require float32 HIP/DCU tensors. Unsupported devices or
modes raise clearly; they never silently fall back. TileLang is imported lazily,
so importing `kernels` or `kernels.tilelang_ops` does not require TileLang.

## Validation

Each module has a CPU-runnable PyTorch-reference test and a TileLang forward /
backward parity test that runs only when both TileLang and HIP/DCU are present.
The current development host has neither TileLang nor HIP/DCU, so successful
CPU reference tests do **not** establish that the JIT kernels compile or that a
full DPA4 energy/force run passes on the target device. Keep that limitation
visible in validation reports.
