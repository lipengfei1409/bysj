# 独立 DPA4 numerical modeling（增量实现）

当前仍区分三条路径：`ShapeBackend` 只追踪 meta shape；`NumericalBackend` + `DPA4Model` 负责项目自有数值路径；`DPA4Runner` 仍可调用 DeePMD 原模型作为 E/F/V oracle。

**独立模型的 OMol25 Neo 全路径尚未验证。** 目标 checkpoint 是 `DPA4-OMol/DPA4-Neo-OMol25-100M-v20260820.pt`；loader 报告 loaded/deferred 权重并继续保留 `complete_model=False`。当前源码已经接入 charge/spin、环境 FiLM、Wigner/GIE、SO(2)/attention、SO(3) grid、output FFN 和 fitting head 的 reference 组件；当前 Python 环境没有 PyTorch，尚未安全地用项目 loader 读取 `.pt` tensor 或运行端到端模型。因此这表示代码接线范围，不表示完整 checkpoint 已可运行、数值正确或已加速。

## 数据流与状态

N：展平原子数；E：有向边容量；F：帧数；C：通道数；R：radial 数；D0=(L0+1)²。

| 阶段 | 输入 → 输出 | 数值状态 |
|---|---|---|
| geometry | positions[N,3], edge[2,E], shift[E,3], cell[F,3,3] → vec/unit[E,3], distance[E,1] | 已接入 geometry，派生值保持边向量 VJP |
| radial basis | distance、freq[1,R]、mask → rbf[E,R], env[E,1] | Bessel + C3，reference / HIP |
| type/pair | atype、table[T+1,C]、edge → type[N,C], pair[E,C] | 真实参数 gather、src+dst、padding 归零 |
| degree | env、dst → degree[N], inverse[N,1,1] | segment_sum(env²)，版本相关 floor |
| radial MLP | rbf、matrix[K,C] → radial[E,L0+1,C] | Linear/可选普通 RMSNorm/Activation，末层仅 Linear |
| fuse | radial、pair → fused[E,L0+1,C] | pair 广播加到 degree 轴 |
| Wigner/FiLM/GIE | geometry/type/radial/degree → D/Dt, film, seed[N,D0,1,C] | reference modules and checkpoint mapping are present; not yet numerically validated on OMol25 |
| blocks | seed、D/Dt、fused → x[N,D,1,C] | focus/SO(2)/radial mixer/gating/attention 与 message-node/FFN SO(3) grid reference code 已接线；尚未在 OMol25 上验证 |
| readout/fitting | x → descriptor[N,C] → atom_energy[N,1] | output FFN、GLU fitting head 与 per-type energy bias mapping 已写；端到端路径未验证 |
| energy/force/virial | atom_energy、边向量梯度、拓扑 → E[F,1], force[N,3], virial[F,3,3] | 基础汇总可执行，完整 VJP 取决于未实现模块 |

状态查询：`model.backend.report()`。grid/depth/cartesian 等接口也在 catalogue 中，但主模型尚未展开所有可选分支，不能据此宣称 Neo/DPA4c 完整兼容。

## 运行真实 checkpoint 前端

工程根目录执行（小系统 smoke test，不是完整模型能量测试）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python3 - <<'PY'
import torch
from model.dpa4.modeling import AtomicBatch, EdgeGraph, DPA4Model
model = DPA4Model.from_checkpoint('../deepmd-kit/examples/water/dpa4/lmp/pretrained.pt')
batch = AtomicBatch(torch.tensor([[0.,0.,0.], [.96,0.,0.], [-.24,.93,0.]]),
    torch.tensor([0,1,1]), torch.zeros(3,dtype=torch.int64), 1)
edge = torch.tensor([[1,2,0,2,0,1], [0,0,1,1,2,2]])
graph = EdgeGraph(edge, torch.zeros(6,3,dtype=torch.int64),
    torch.zeros(6,dtype=torch.int64), torch.ones(6,dtype=torch.bool))
features = model.prepare_edges(batch, graph)
for name, tensor in vars(features).items():
    print(name, tuple(tensor.shape), tensor.dtype)
print('loaded:', model.checkpoint_report['loaded_keys'])
print('deferred tensor count:', len(model.checkpoint_report['deferred_keys']))
print('complete_model:', model.checkpoint_report['complete_model'])
PY
```

water 示例的 radial `[E,2,16]`、rbf `[E,6]` 只用于展示前端 API，不是 Neo 维度。只接收可信 `.pt`，用 `weights_only=True`；多 head 必须指定 `head`。前端 loader 当前支持 Bessel、charge/spin embedding、environment seed/FiLM 和所接入的 Wigner/GIE/SO(2)/attention 子模块；Gaussian、未折叠 LoRA、exclude_types、inner clamp 等仍明确拒绝或留作 deferred。完整模型的活跃 grid/readout/fitting 权重尚未全部映射，不能据此报告 `complete_model=True`。

## DCU 与参数管理

```python
model = DPA4Model.from_checkpoint(checkpoint,
    primitive_backend='hip', linear_backend='rocblas').to('cuda')
features = model.prepare_edges(batch, graph)
```

`primitive_backend` 控制 geometry/radial/norm/scatter；`linear_backend` 单独选 reference/hip/rocblas。构建见 `doc/blas.md`，显式 native 请求不降级。DCU float tensor 当前要求 FP32，同设备；索引 INT64、mask bool。reference 支持 `.double()`，输入和模型必须显式同步移动。

这是**混合 eager 基线，不是全 native/megakernel**：type/pair、激活、mask、几何派生用 Torch，energy_reduce 明确用 Torch FP64。HIP force/virial 汇总是一阶推理边界，输入 detach；FP32 汇总后的 virial 只转换为 FP64 输出，不声称 FP64 accumulation。force-loss/二阶导训练尚不支持。本机没有 DCU，HIP/rocBLAS 编译、精度与性能待实机验证。

顶层仅拥有一个 backend，blocks 借用，不重复保存参数。state_dict 保留 descriptor version，恢复不能丢失 v1.1 envelope/floor 语义。config/backend 仍由构造器提供，项目 state_dict 不是 DeePMD checkpoint 导出格式。

## 逐项补算子

`NumericalBackend(nn.Module).register(operator, module, stage=None)`：模块 `forward(**inputs)` 返回 Tensor 字典，须满足 stage 输出名/shape/dtype/device。stage 实现优先，同一位置重复添加报错；学习参数必须注册为 nn.Parameter。

```python
# 这两个类为后续自行编写的真实实现，不是项目已有实现。
backend.register('wigner_rotation', YourWignerOp(config))
backend.register('so2_linear', YourSO2Op(block_weights), stage='blocks.0.so2_linear.0')
```

不同 block/层的 SO2 权重通常不同，不能用同一套权重冒充多层。签名可从 DPA4Model 或 shape trace 的 inputs 读取；补全了 environment_seed 的 atype/vec/env/mask，GIE 的 inverse/mask，以及 attention/reduce 的 degree/mask。数学合同见 `doc/dpa4-operators.md`；普通 RMSNorm 不替代 equivariant_norm。

几何采用 `src + shift@cell - dst`，盒是行向量布局；cells=None 仅允许零 shift。有效边须同帧，所有容量槽索引合法（不允许 -1）。mask=False 的 vec 为 canonical +z，rbf/env/pair 归零；距离为上游 `sqrt(sum(vec²)+eps²)`。独立 primitive 合同不变，SeZM 语义在 modeling 层实现。

native geometry 的输出是 autograd 兄弟节点，因此从 vec 重算 distance/unit，确保 `dE/dvec` 包含 radial/angular 路径。计算力可在外层 no_grad 下重新启用梯度，不修改调用方 requires_grad；不要使用 inference_mode 计算力。admission 检查会读取 device scalar 并可能同步，不是未来 device 调度性能路径。

## 验证与下一步

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python3 -m unittest tests.python.test_numerical_modeling -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python3 -m unittest discover -s tests -v
```

在真实 DCU 上编译扩展后，单独验证前端数值及坐标一阶梯度（可用 `DPA4_WATER_CHECKPOINT` 指定权重路径）：

```bash
DPA4_RUN_MODEL_NATIVE_TESTS=1 PYTHONPATH=python python -m unittest tests.csrc.test_model_frontend -v
# 已构建 rocBLAS 时同时测 library linear：
DPA4_RUN_MODEL_NATIVE_TESTS=1 DPA4_RUN_ROCBLAS_TESTS=1 PYTHONPATH=python python -m unittest tests.csrc.test_model_frontend -v
```

显式开启实机测试但没有扩展/DCU/权重时会失败，不用 skip 假装实机通过。

测试覆盖真实 water 前端权重、upstream RadialMLP/MLPLayer/RMSNorm forward 合同、一阶 gradcheck、版本 roundtrip；测试专用注入数值图用有限差分检查 E/F 接线，**不代表 DPA4 物理模型**。两个 native 边界探针只模拟拓扑/输入合同，不是 native 精度证据。

后续先在可用 PyTorch/DeePMD 环境验证 OMol25 权重键、grid projector、readout/fitting 与完整 E/F/V；修正数值/参数映射问题后，再补全一阶 VJP 的逐层对照。reference 阶段过 Gate 后才实现/验证 native HIP fusion 和 device task runtime。当前未运行这些验证。
