# DeepMD-kit DPA4/SeZM 算子与推理参考

## 文档范围

本文按 DeepMD-kit 的 DPA4/SeZM descriptor 数据流，整理主要算子、张量语义和上游融合 GPU op。它描述的是 DeepMD-kit 框架设计，不表示这些算子已在当前 DMK 仓库实现。DMK 当前的 production native kernel 是 DUMMA GEMM；另有一个尚未在 DCU 上验证的 TileLang Bessel radial-basis inference op，见 [DMK 当前 Kernel](dpa4-kernels.md)。

梳理依据是相邻的 DeepMD-kit checkout，revision `f7e7b9e6`。核心入口位于 `deepmd/dpmodel/descriptor/dpa4.py` 的 `DescrptDPA4.call` / `call_graph` / `_run_graph`；PyTorch 实验路径及其融合 op 位于 `deepmd/pt_expt/descriptor/`、`deepmd/pt_expt/kernels/cuda/dpa4/` 和 `source/op/pt/dpa4/`。DPA4 与 SeZM 是该 descriptor 家族的注册名称。不同 backend/config 会选择不同实现或跳过可选分支，下面将数学/框架算子与自定义 CUDA 融合算子分开说明。

## DMK 的 TileLang 单卡推理开发阶段

第一阶段不做融合，也不实现整个模型的 megakernel；目标是单卡 DPA4/SeZM inference，首阶段验收能量与力（stress/virial 暂不包含）。把该推理配置需要的 TileLang 算子分别实现、单独验证，尚未替换的算子继续使用普通 PyTorch/kernel。TileLang 开发代码放在 `python/kernels/tilelang_ops/`，按每个算子一个模块组织 forward/backward 与 PyTorch autograd wrapper。首个独立算子是 raw Bessel radial basis，C3 envelope 仍是单独 PyTorch 操作；默认不启用 TileLang，也不增加默认依赖。其他算子和模型路径验证通过后，再评估局部融合。

本文件的算子表和 `pt_expt` fused CUDA op 是 DeepMD-kit 的框架参考，不是 DMK 的实现清单。可供后续评估的局部候选包括 `dpa4_zonal_scatter`（edge message 生成、destination CSR reduction、degree normalization）及 `dpa4_so2_conv`（本地邻居特征访问、SO(2) mixing、attention 与 aggregation）；目前没有选定或实现其中任何一个。

第一阶段限定单卡推理，不实现 `border_op`/MPI ghost-row exchange。单卡 neighbor gather/scatter/CSR aggregation 属于可单独验证的图算子；跨 rank 同步还涉及 ownership、ordering 和 backward 语义，不应与本阶段混为一谈。文档后续列出的 fused CUDA op 是 DeepMD-kit 的上游实现参考，不代表 DMK 当前的实现计划。

## 主数据流

```text
coordinates + types + neighbor list / NeighborGraph
  -> edge geometry + type/condition embeddings
  -> edge cache (cutoff, radial features, smooth degree, Wigner rotations)
  -> optional environment FiLM + geometric initial embedding (GIE)
  -> repeated SeZM interaction blocks
       SO(2) edge message/convolution + aggregation
       equivariant FFN + residual / optional AttnRes
  -> l=0 / optional SO(3)-grid readout
  -> per-atom energy fitting
  -> frame energy reduction; force / virial by energy derivatives
```

`NeighborGraph.edge_index` 采用 `[src, dst]`，其中 `src` 是邻居、`dst` 是中心；`edge_vec[e] = r_src - r_dst`，周期镜像位移已包含在扩展坐标中。`edge_mask` 标识有效边；稠密 neighbor-list adapter 会将 padding 或无效 mapping 变为无效边。

## 框架算子定义

| 阶段 | 算子 | 定义 / 作用 |
|---|---|---|
| 邻居图与几何 | `NeighborGraph`、edge geometry | 从坐标和邻居图得到 `src/dst`、`edge_vec[E,3]`、`edge_mask[E]`；距离由 `r_e = sqrt(sum(edge_vec[e]^2) + eps^2)` 得到，并用于构造 edge-aligned quaternion。图可来自 dense nlist 转换，也可由 graph-native 路径传入。 |
| 类型嵌入 | `SeZMTypeEmbedding` | `h_i = E[type_i]`，生成节点的 l=0 标量通道。 |
| 条件嵌入（可选） | `ChargeSpinEmbedding`、native `SpinEmbedding` | 将 frame 级 charge/spin 或 node spin 注入标量嵌入、初始 l=1 backbone 或邻居 l=1 消息；关闭相应模型选项时不执行。 |
| edge cache | `_edge_cache_from_arrays` / `EdgeCache` | 对每条边缓存端点、向量、距离相关 radial basis、C3 cutoff envelope、edge type feature、Wigner `D/Dᵀ` 或压缩表示。`edge_type_feat[e] = h_src + h_dst`。平滑度数 `deg[i] = Σ_{dst[e]=i} edge_env[e]^2`，归一化 `inv_sqrt_deg[i] = 1/sqrt(deg[i] + floor)`。可选 SFPG source gate 作为额外边权。 |
| 径向特征 | `RadialBasis`、`RadialMLP` | Bessel 或 Gaussian basis 将标量距离展开为 `rbf[E,R]`，并乘以 C3 cutoff envelope；径向 MLP 将其映射到每个 degree/channel 使用的 edge feature。参数配置可决定频率是否可训练。 |
| 环境条件（可选） | `EnvironmentInitialEmbedding` + FiLM | 对边的径向、类型和方向特征作加权聚合，得到每节点 scale/shift；`x0' = x0 * (1 + exp(a) * tanh(norm(scale))) + exp(b) * tanh(norm(shift))`。仅 `use_env_seed` 等开关启用时执行。 |
| 几何初始嵌入 | `GeometricInitialEmbedding` (GIE) | 以 edge quaternion 的 zonal/Wigner 系数与相应 degree 的 radial feature 构造非标量消息：`m[e,r,c] = zonal[e,r] * radial[e,slot(r),c]`；按 `dst` scatter-sum，再乘节点平滑度归一化。l=0 行保持为零；输出加到初始 SO(3) backbone。 |
| SO(2) 消息 | `SO2Convolution` | 对节点系数做 channel/focus mixing；用 edge Wigner `D` 将源节点表示转到局部 m-major basis；施加 degree-aware radial modulation 和 SO(2) mixing/gated activation；再用 `Dᵀ` 转回全局系数。 |
| 消息聚合 | envelope scatter 或 attention | 无 attention head 时，边消息乘 cutoff envelope（和可选 source gate），按 `dst` scatter-sum 并做 degree normalization。启用 attention 时，先生成 edge/focus/head logits，再按 destination 做 envelope-gated softmax，使用正的 null mass `zeta = softplus(z_bias) + eps`，最后施加 output head gate。 |
| 交互块 | `SeZMInteractionBlock` | 一个 SO(2) convolution branch（可选 pre/post norm）和一个或多个 equivariant FFN subblock；baseline 用 residual shortcut。可选 block/unit-level AttnRes 对历史表示作 depth-wise 加权汇聚。模型的 `l_schedule` 可逐 block 缩小 degree。 |
| 可选等变乘积 | `GridPair` / S2、SO3 grid、Cartesian tensor product | 网格乘积的核心为 `from_grid(to_grid(left) * to_grid(right))`；可用于节点或 message-node 交互。Cartesian tensor-product 分支是另一种按张量阶数耦合特征的路径。均为配置可选项。 |
| Readout | `DescrptDPA4._apply_readout` | `so3_readout="none"` 直接取 l=0；其他 readout 配置可在 SO(3) grid/readout FFN 中将高阶系数折入标量，再输出节点 descriptor。 |
| 能量拟合 | `SeZMEnergyFittingNet` / `GLUFittingNet` | 对每节点 descriptor（以及按配置拼接的 frame/atom 参数）作 fitting。GLU 隐层为 `[v,g] = xW+b`、`x' = v * activation(g)`，输出投影得到 atom energy `E_i`。 |
| 总能量、力、virial | `DPA4EnergyModel` + `EnergyModel` | 总能量 `E = Σ_i E_i`；力 `F_i = -∂E/∂r_i`；virial 按模型导数定义从坐标/盒子导数得到。`DPA4EnergyModel` 是 DPA4/SeZM 类型注册包装，能量/导数语义继承通用 `EnergyModel`。 |

## 等变子算子定义

下列模块是交互块/读出常见的可组合部件；具体使用受模型配置、`lmax/mmax` 和 backend 影响。

| 子算子 | 定义 |
|---|---|
| `SO3Linear` | 对每个 degree `l` 使用一个 channel 权重矩阵，并在该 degree 的全部 `m` 分量间共享；只在 l=0 标量分量可加 bias。它不混合不同 degree/m，因此保持 SO(3) 等变。 |
| `SO2Linear` | 在 edge-aligned 的 m-major 表示上作 block-diagonal mixing。m=0 可在 degree 间混合；`|m|>0` 的实数 2x2 块按复数乘法结构耦合 `(-m,+m)`，保证绕局部 z 轴的 SO(2) 等变性。 |
| `GatedActivation` / `EquivariantFFN` | 标量 l=0 走配置的 SiLU/GELU 等激活；l>0 只接收由标量通道产生的 gate。典型 FFN 为 `SO3Linear -> gated activation (可用 GLU/SwiGLU 形式) -> SO3Linear`。 |
| `RMSNorm` / `EquivariantRMSNorm` | 普通 RMSNorm 沿 channel 轴归一化；等变版本对 degree 做平衡后计算共享 RMS，再用 per-degree scale（bias 仅作用于标量 slice），避免不同 `2l+1` 维数改变各 degree 的统计权重。 |
| `WignerDCalculator` | 将 edge quaternion 映射为按 degree 分块对角的 Wigner `D` 和 `Dᵀ`；用于 global/local frame 旋转。融合路径可改用 packed Wigner runs。 |
| `DepthAttnRes` | 从各历史表示的 l=0 标量通道得到 query/key 和 depth 权重，再将相同权重广播到完整等变张量后聚合；权重只看不变量，因而不破坏表示结构。 |
| `GridProduct` / `GridMLP` | 先将 SO(3)/S2 系数投影到网格，在每个网格点做乘积/小型 MLP/门控混合，再投回系数空间；用于配置启用的高阶非线性交互。 |

## DeepMD-kit PyTorch 路径中的融合 CUDA 算子

这些是 `pt_expt` CUDA 路径中注册的重点 fused op；它们不是所有配置都会启用，也不等于上表全部框架算子。

| 注册算子 | 定义 / 输入输出 | 主要作用 |
|---|---|---|
| `deepmd::dpa4_edge_radial` (+ backward) | 输入 edge length、keep mask、basis frequencies 和 envelope 系数；输出 `edge_env[E,1]` 与 `edge_rbf[E,R]`。 | 单次边遍历融合 cutoff envelope 与径向 basis，反向对 edge length 求梯度。 |
| `deepmd::dpa4_wigner_dense` (+ backward) | edge quaternion -> block-diagonal Wigner `D[E,D,D]` 和 `Dᵀ`。 | 为普通 dense rotation path 构造并缓存 edge frame rotation。 |
| `deepmd::dpa4_wigner_runs` (+ backward) | quaternion 和稀疏多项式表 -> packed Wigner rows/runs。 | 为融合 SO(2) convolution 生成压缩旋转表示；`SO2ConvCuda` 也复用其中 l,m=0 行作为 GIE zonal coupling。 |
| `deepmd::dpa4_zonal_scatter` (+ backward) | `zonal[E,R]`、`radial[E,L,C]`、destination CSR 与 node scale -> `(N,R+1,C)`。非标量行按 `zonal * radial` 生成并对 destination 归约，标量行留零。 | 融合 GIE 边消息生成、CSR destination reduce 与 degree normalization，避免物化 `(E,R,C)` 中间量。 |
| `deepmd::dpa4_so2_conv` (+ backward) | 节点特征、src/dst CSR、Wigner runs、radial coefficients、SO(2) weights、attention query/key/null logits、envelope/head gate -> gated destination aggregate。 | 推理时将局部旋转、SO(2) value mixing、attention 和聚合融合到每边的 CUDA operator span。 |
| `deepmd::sezm_so2_value_fwd/bwd/bwd2` | 输入节点特征、source CSR、Wigner runs、radial degree kernel、SO(2) weights 和 focus-competition 参数；输出局部 value features；提供一阶及二阶反向路径。 | 训练路径融合 SO(2) value mixing/focus competition，并保留训练所需导数。 |
| `deepmd::dpa4_grid_pair` (+ backward) | 系数场 `left/right[N,P,C]` 与投影矩阵 -> `from_grid(to_grid(left) * to_grid(right))`。 | 避免完整 materialize grid 中间张量的网格乘积。 |

对应 Python/C++ 文件集中在 `deepmd/pt_expt/kernels/cuda/dpa4/`、`source/op/pt/dpa4/`；标准组合路径仍使用 PyTorch/array-API 的 Linear、scatter、norm、activation 等基本算子。

## 单卡与多 rank / 多卡通信

- **单卡**：一个 `NeighborGraph` 包含本 rank 的节点及本地邻接；edge cache 和 interaction block 在当前设备上执行。按 `dst` 的 scatter/CSR 归约在本地完成。
- **多 rank 的图路径**：邻居可映射到 ghost/halo 节点。SO(3) 表示在共享全局坐标框架中，因此 ghost 节点特征可直接从 owner rank 复制。需要远端邻居特征的 interaction block 在 SO(2) cross-node convolution 前调用 `deepmd_export::border_op` 刷新 ghost rows；block 0 在初始特征可本地重建时可跳过 exchange。
- **通信反向**：`border_op_backward` 将 ghost 行梯度累加回 owner 并清理 ghost 梯度，随后 `border_op` 再同步完成值；这样跨 rank 的 message/energy 梯度可以沿相同图路径回传。
- **通信元数据**：`comm_dict` 携带 `send_list`、`send_proc`、`recv_proc`、`send_num`、`recv_num`、communicator、`nlocal`、`nghost` 等边界交换信息。实际通信由 DeepMD-kit 的 `border_op` 自定义 op/MPI 层负责，不是 DUMMA GEMM 的功能。
- **重要实现差别**：`DescrptDPA4.call()` 的 dense nlist lower 明确拒绝非空 `comm_dict`；graph-native `call_graph()` 才把通信往 block 传。`dpmodel` 的通用 `exchange_ghost_features` leaf 会报 `NotImplementedError`；相邻 checkout 的 `pt_expt` PyTorch override 实际调用 `torch.ops.deepmd_export.border_op`。因此不能把 graph/pt_expt 的多 rank 能力误写成 dense lower 或所有 backend 都支持。

## 代码索引

- Descriptor orchestration：`deepmd/dpmodel/descriptor/dpa4.py`
- Edge cache、degree normalization、edge type feature：`deepmd/dpmodel/descriptor/dpa4_nn/edge_cache.py`
- Embeddings、radial、SO2、block、attention、Wigner、grid：`deepmd/dpmodel/descriptor/dpa4_nn/`
- Energy wrapper/fitting：`deepmd/dpmodel/model/dpa4_model.py`、`deepmd/dpmodel/fitting/dpa4_ener.py`、`deepmd/dpmodel/model/ener_model.py`
- PyTorch distributed graph exchange：`deepmd/pt_expt/descriptor/dpa4.py`、`deepmd/pt_expt/descriptor/dpa4_nn/block.py`
- Fused CUDA operator wrappers/kernels：`deepmd/pt_expt/kernels/cuda/dpa4/`、`source/op/pt/dpa4/`
