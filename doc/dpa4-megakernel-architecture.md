# DPA4-Neo OMol25 DCU MegaKernel 架构说明

> 本文记录 `myproject` 当前架构、OMol25 Neo 的活跃分支，以及后续设备调度的目标边界。它是设计/实现说明，不代表模型已完整运行、已通过 DCU 精度验证或已有加速结果。

## 1. 目标与边界

首期目标是独立 `myproject` 推理路径，接受原子批次和调用方提供的边图，输出能量、力和 virial；用 DeePMD 原模型作数值 oracle。首期不修改 DeePMD eager adapter、LAMMPS/Kokkos ABI 或 MPI/ghost 通信代码，也不把 DPA4C 当作 DPA4。

模型目标固定为：

- Checkpoint：`../DPA4-OMol/DPA4-Neo-OMol25-100M-v20260820.pt`
- 配置：`../DPA4-OMol/DPA4-Neo-OMol25-100M-v20260820.json`
- 模型类型：SeZM；全周期表 H–Og；`rcut=6 Å`
- `channels=32`、`n_radial=16`、`lmax=3`、`mmax=1`、2 个交互块
- 2 个 focus、1 个 attention head；`radial_so2_mode=degree_channel`、rank 1
- charge/spin embedding、message-node SO(3) grid、SO(3) grid FFN 均启用

以上配置来自随 checkpoint 提供的 JSON；权重 tensor-level 加载、逐层数值对照尚未在本环境验证。不要用 water 或 OMat24 Neo 的维度替代这份模型的配置。

## 2. 输入与数据合同

`myproject/python/model/modeling/types.py` 定义独立数值路径的输入：

- `AtomicBatch.positions[N,3]`、`atype[N]`、`frame_index[N]`、可选 `cells[F,3,3]`
- OMol25 条件 `charge_spin[F,2]`，也接受 `[2]` / `[1,2]` 并按帧广播；默认值由 checkpoint 的 `[charge, multiplicity]` 提供
- `EdgeGraph.edge_index[2,E_capacity]`，其中第 0 行是 `src`、第 1 行是 `dst`；另有周期 `shifts[E,3]`、`frame_index[E]` 和 `mask[E]`

几何约定为行向量晶胞：

```text
d[e] = pos[src[e]] + shifts[e] @ cells[edge_frame[e]] - pos[dst[e]]
```

masked capacity 槽的索引仍必须有效，不能用 `-1` 表示 padding。首期沿用显式 `EdgeGraph` 输入；`NumericalBackend` 的 `neighbor_graph` 目前没有数值实现，所以不应把自动邻居搜索说成已支持。

## 3. 当前数值路径

```text
AtomicBatch + EdgeGraph
  ├─ type embedding + frame charge/spin embedding
  ├─ edge geometry → Bessel radial basis + cutoff envelope
  ├─ pair embedding / envelope-squared degree reduction
  ├─ environment seed → FiLM
  ├─ Wigner-D rotation + GIE geometric seed
  └─ interaction blocks
       ├─ degree-wise focus projections
       ├─ gather source features / rotate to edge frame
       ├─ radial degree-channel mixing
       ├─ SO(2) linear stack + gated activation + focus competition
       ├─ inverse rotation
       ├─ Q/K attention + envelope-gated null-mass softmax
       ├─ destination message reduction / output head gate
       ├─ message-node SO(3) grid product
       └─ residual / equivariant norm / SO(3)-grid FFN
  → scalar readout → fitting energy → frame energy
  → dE/d(edge vector) → force / virial assembly
```

各项的数学/张量合同维护在 [`dpa4-operators.md`](dpa4-operators.md)。`DPA4Model.prepare_edges` 是目前可单独使用的前端边界；checkpoint loader 会报告已加载与 deferred 的键，且不会把部分权重误报为完整模型。

### 目前已经接线的独立组件

- OMol25 配置字段和 `charge_spin` 输入：`python/model/modeling/config.py`、`types.py`
- charge/spin embedding、environment seed、FiLM、GIE：`python/model/modeling/components.py`
- 参考 Wigner-D、focus/SO(2)、radial mixer、gated attention、SO(3) grid projector/product/FFN：`python/model/modeling/wigner.py`、`components.py`、`so3_grid.py`
- checkpoint 参数映射及 loaded/deferred 报告：`python/model/modeling/loading.py`
- 常见算子 dispatch、reference primitive 和一阶力路径：`python/model/modeling/numerical.py`

这些实现目前主要是 PyTorch reference/组合算子；已有 geometry、radial、fused edge-features、RMSNorm、linear、reduce 和 force/virial primitive 可按 backend 选择。融合边特征 kernel 位于 `csrc/parallel/edge_features.cu`，Python HIP 路径也将 scalar feature VJP 接回 edge vector；它尚未 HIP 编译或在 DCU 上验证。上述新增 model-stage 代码尚未在目标 checkpoint 上完整加载或数值运行。

### 明确未完成的计算

当前代码已包含 OMol25 活跃分支的 PyTorch reference 模块和 checkpoint 参数映射：message-node SO(3) grid、block FFN SO(3) grid、scalar output FFN 与 fitting head。它们尚未在可用 PyTorch 环境中加载这份 `.pt` 并执行，因此 `checkpoint_report['complete_model']` 仍保持 `False`，也不能据源码接线认定完整 E/F/V 正确。完整一阶 VJP 的数值对照仍待验证。

另一个独立缺口是完整 DCU MegaKernel 执行：新增了 opt-in HIP persistent worker 和 selected-task runtime plan，但 dispatcher 当前只支持 `edge_features` opcode；完整 DPA4 DAG 仍不能交给 worker 执行。runtime/kernel 尚未编译、上机或测量，也没有完整模型 DCU 执行或速度数据。

## 4. MegaKernel 任务图与同步边界

`python/dcu_dpa4/megakernel/planner.py::build_dpa4_task_graph` 仍生成**静态任务模板 DAG**；`PersistentTaskPlan` 可把显式选中的 `edge_features` 及其依赖 closure 降为 runtime ABI。当前 worker 不会执行其他 DPA4 operator。每个静态任务记录：

- operator、edge/node/frame work domain、shape bucket 工作量和 tile size
- dependency names / IDs、epoch、variant 和下游 completion fanout
- `edge_capacity` 是存储上限，动态有效边数仍需由设备输入/mask 控制

规划将边特征、node/attention reductions、message passing、frame reduction 和反向/force 拆成显式 epoch。`DPA4Model.plan_execution()` 暴露静态 DAG，`to_json()` 可导出审计用任务/依赖表；`PersistentTaskPlan.from_task_graph()` 为显式选中的任务及其依赖 closure 做 v1 lowering，不改变 planner v1 JSON。当前 native executor 将只接受 edge-feature payload、单任务或线性同算子依赖链；未支持 operator 在 launch 前拒绝。

首个 tile-local `geometry + radial basis/cutoff` 融合已写入 `csrc/parallel/edge_features.cu`，由 `dcu_dpa4.ops.edge_features` 暴露，包含 vec/distance/unit/rbf/env 的一阶 VJP；HIP 模型路径可调用它。planner 的 `fuse_edge_features=True` 会用一个 `edge_features` task 替代 `edge_geometry` 与 `radial_basis`。`execute_persistent_edge_features(task_graph, target_tasks=...)` 是独立 opt-in task API，只执行选中的 edge-feature 依赖 closure 并返回输出与异步 status；`check_runtime_status()` 会同步读取该状态并对错误抛异常。`DPA4Model.forward()` 仍走直接 HIP kernel，不会把完整模型误报成 persistent execution。`fuse_edge_features=False` 可选择未融合 geometry/radial 基线。融合 kernel 和 persistent worker 均尚未编译或上机，梯度回归测试也尚未运行。跨 tile destination scatter、attention 分段归一化、frame reduction 和 force scatter 不能靠 device fence 或“全 grid 同步”替代真实依赖；这些操作要么使用有界 task/event 协议，要么保留独立 launch/epoch。

参考来源及迁移边界：

- Mirage：借鉴 task DAG、依赖事件、producer/consumer 调度；不复用它的 CUDA runtime。
- ThunderKittens：借鉴 loader/consumer/storer 角色、tile 生命周期和 pipeline 组织；不移植 warp/TMA/WGMMA/cluster 专属指令。
- DTK/HIP：实际 gfx 型号与编译器确定后再选择 wave、同步、LDS 和 DUMMA 形状；使用 PyTorch 当前 stream。host rocBLAS/hipBLASLt 不可从 device-side persistent kernel 调用。

## 5. 当前阻塞与后续实现顺序

1. 在可用 PyTorch/DeePMD 环境加载 OMol25 权重，核对 state-dict shape、model params、head 和所有活跃分支；当前工作站 Python 缺 PyTorch，尚不能执行 `.pt` tensor-level mapping。
2. 完成 `message_node_so3`、SO(3)-grid FFN、scalar readout、fitting 和所有活跃权重映射；继续保持未支持分支 fail-closed。
3. 补齐完整一阶 VJP，尤其是所有局部 message/grid 运算到 edge vector 的梯度，再验证 force/virial 组装。
4. 在完整 Torch reference oracle 后，针对真实活跃路径补齐 HIP/DTK native operators，并验证已写的 edge-feature fusion island；大 GEMM 首期继续走 host-launched 框架/库路径。
5. 将 worker 的 typed dispatch 从 `edge_features` 扩展到其他已具 device-callable 实现的 task；持续保留版本化 descriptor、有界队列与 generation/error 协议，并用实际占用边界避免 producer 尚未调度而所有 CTA 自旋的死锁。
6. LAMMPS/Kokkos 和 DeePMD eager adapter 是后续独立集成目标，不属于当前 standalone runtime。

## 6. 后续验证（当前未执行）

依次进行：Python reference/loader 回归；OMol25 与上游相同 topology 的逐阶段/E/F/V 对照；力有限差分、旋转/置换与 virial 检查；真实 DTK/DCU 的 HIP build/load、native 精度和 stream 检查；最后比较普通 launches、HIP Graph 与 persistent scheduler 的完整 E/F/V 延迟及 scheduler 开销。当前未运行测试、编译、上机或 benchmark；不据此文档宣称模型支持或加速。

现有项目命令和硬件验证边界见 [`dtk-native.md`](dtk-native.md)、[`numerical-modeling.md`](numerical-modeling.md) 与 [`verification.md`](verification.md)。
