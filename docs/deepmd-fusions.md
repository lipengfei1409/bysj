# deepmd-kit 已有的算子融合清单

上游（`deepmd-kit`，本 checkout 的 `42641a6e`）在 `deepmd/pt_expt/kernels/` 下自带一整套融合算子：
**5 套实现层、约 20 个融合点**。最深的一个 `deepmd::dpa4_so2_conv` 把一条 per-edge 的 SO(2) 路径
（attention logits → 门控 softmax → 旋转 → radial → mixing → 逆旋转 → 加权 → 目标归约 → head gate）
压成一个算子，中间量一律不落 HBM。

- 层：native CUDA（`source/op/pt/*.cu`）、CuTe-DSL、cuTile、Triton、Python 参考层。
- 开关：`DP_CUDA_INFER`(0–2)、`DP_TRITON_INFER`(0–3)、`DP_CUTE_INFER`、`DP_CUTILE_INFER`、`DP_CUDA_TRAIN`。
  全部在**模型构造时**读取，因此在 `make_fx` 图里是编译期常量。
- 两套后端都绑定：`pt_expt/descriptor/dpa4_nn/*` 与 `pt/model/descriptor/sezm_nn/*` 是镜像的。

> 全部 native 算子只在 `USE_CUDA_TOOLKIT AND DEEPMD_TORCH_HAS_CUDA` 下进入构建
> （`source/op/pt/CMakeLists.txt`），所以 DCU 构建里 `torch.ops.deepmd.*` 不存在，dispatch 静默回退
> 稠密参考路径。Triton 那套理论上跨厂商，但本机 gcvm（LLVM 17）编不了 Triton 3.5 的 IR。

## 1. SeZM / DPA4（本 checkpoint 这条线）

| # | 融合算子 | 层 | 融合了哪些算子 | 省掉的访存 | 调用点 | 开关 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | `dpa4_so2_conv`（fwd/bwd，12+12 个 `(CF,L)` 特化） | native | attention logits + envelope 门控分段 softmax + Wigner 旋转 + radial degree mixer + 门控 mixing stack + 逆旋转 + attention 加权 + dst 归约 + head gate | 整段 per-edge 中间量不下 HBM；被替代的组合每次卷积搬 **≈3.6 GB** 的 `(E, ROW)` 激活[^1] | `dpa4_nn/so2.py:299`（Step 14）／`sezm_nn/so2.py:1711` | `DP_CUDA_INFER>=2` + `_is_supported` + `_profitable` |
| A2 | `sezm_so2_value_fwd` / `_bwd` | native | value 流到 attention 之前：按结构非零元 gather Wigner、radial mixing、cross-focus competition、全部 gated 层、末端 identity 层；反向一个算子重算 | 窄布局下旋转后输入与各层激活驻留 smem，只有 3 个 anchor 跨 dispatcher 边界 | `so2.py:228`（Step 15） | `DP_CUDA_TRAIN=1` |
| A3 | `dpa4_wigner_dense` | native | 四元数 → `(D_full, Dt_full)` 打包 Wigner 对 | 组合路径 5 趟全尺寸 pass（monomial basis / GEMM / 补零 / block scatter / 转置拷贝）→ 读四元数 + 2 次写 | `descriptor/dpa4.py:286` | `DP_CUDA_INFER>=1` |
| A4 | `dpa4_edge_radial` | native | cutoff envelope + radial basis 一次算 | 组合式下这段 96 MB 的 pass 会被**每个消费者**内联重算；算子边界后只做一次 | `descriptor/dpa4.py:283`／`sezm.py:1097` | `DP_CUDA_INFER>=1` |
| A5 | `dpa4_grid_pair` | native | `from_grid(to_grid(l)·to_grid(r))`，不物化 grid field | grid field 是系数操作数的 **39 倍**（还顺带省掉每个乘法的全尺寸 layout copy） | `dpa4_nn/grid_net.py:35` | `DP_CUDA_INFER>=1` |
| A6 | `dpa4_zonal_scatter` | native | 初嵌入的 per-edge message + 目标归约（gather → scatter → degree 归一） | 参考组合要把 message 写成 `(E, R, C)`，生产尺寸 **1.3 GB**；融合后 message 留寄存器、按 dst CSR 走，且**无原子**、求和顺序确定 | `dpa4_nn/embedding.py:62` | `DP_CUDA_INFER>=1` + `supported()` |
| A7 | `sezm/forward`+`backward`（CuTe） | CuTe | gather → rotate_to_local → radial mix → 3 层 (SO2Linear + GatedActivation + residual) → `x_local`，一个 bucketed kernel；反向在同一核里重算 | `E×D_m×C` 中间量全部留在片上；显存比 eager 低 **1.68×**、比编译路径低 0.5/0.8 GB[^2] | `so2.py:309` | `DP_CUTE_INFER` |
| A8 | `so2_rotate_mix`（cuTile） | cuTile | gather + 块对角 Wigner 旋转（只走结构非零元）+ radial degree mixing | 旋转后的 pre-mix 中间量不物化；旋转+mix 只占 mixing stack 乘加的 **1.5%** | `so2.py:320` | `DP_CUTILE_INFER` |
| A9 | `so2_mixing_stack`（cuTile） | cuTile | 整个 stack（`n_layers-1` 个门控层 + identity 末层）一个核 | 反向重放恢复 pre-activation，省掉每 block **2.11 GB** 的保存量 | `so2.py:320` | `DP_CUTILE_INFER` |
| A10 | `flash_atten`（cuTile） | cuTile | 逆旋转 + degree rescale + envelope 门控 softmax 权重 + dst 归约，四合一 | 旋转回来的 message 与加权后的 value 都不写 DRAM；一 block 一目标节点走 CSR，无原子 | `so2.py:256` | `DP_CUTILE_INFER` |
| A11 | `so2_value_path`（Triton） | Triton | `so2_rotate_mix` + `so2_mixing_stack` 两个 functional op | 同上（旋转中间量 + 各层激活不出片）；反向用 `segment_sum` 替代 `index_add_` | `so2.py:341`（Step 16） | `DP_TRITON_INFER>=1` 且 `hidden_channels>=128` |
| A12 | `so2_rotation`（Triton） | Triton | `rotate_to_local` / `rotate_back`：把 `D_to_m`、`x[src]` 的 gather 折进 GEMM | `lmax=10, E=100k` 时组合式要物化 **≈9 GB** scratch（写了立刻再读），融合后一次不建 | `so2.py` 旋转分支 | `DP_TRITON_INFER>=1` |
| A13 | `so2_block_gemm`（Triton） | Triton | 块对角 `SO2Linear`：每个 `|m|` 块一次 launch | 省掉 block 拼接的新张量 + `permute` 的 strided 权重；`N=192` 不再按 256 补齐（省 ~25%） | `so2.py:101/128` | `DP_TRITON_INFER>=1` |
| A14 | `so2_stack_fp16x3`（Triton） | Triton | mixing stack 的 GEMM 走 fp16 tensor core（Ootomo 两段分裂，fp32 累加） | 不改访存，改**算力**：fp32 的 3 个 fp16 乘积 | `DP_TRITON_INFER>=3`（按 `(focus_dim, lmax)` 查表） |
| A15 | `segment_softmax`（Triton） | Triton | 门控分段 softmax 的 fwd/bwd/二阶，各一个 CSR 段算子 | 参考 ATen 链是由 `scatter_reduce/scatter_add/index_select` 组成的十几个物化面；融合后按 dst 段走一遍 | `so2.py:375`（Step 18） | `DP_TRITON_INFER>=1` 且 `F·H<=16`、fp32 |
| A16 | `radial_mix`（Triton） | Triton | `degree_channel` 分支的 `Σ_r channel_basis[r,c]·Σ_i K_r[e,o,i]·x[e,i,c]` | 不物化 `K_r` 与中间 `(E, RED, C)` | `so2.py:170` | `DP_TRITON_INFER>=1` |
| A17 | `wigner_monomials`（Triton/cuTile） | Triton / cuTile | 四元数单项式基（`l>=2`） | 参考链每 degree 物化 3 个 `(4,P+1,E)` 中间量、反向还要 `cumprod` 扫描；融合后指数表编译期常量、单项式全在寄存器 | `dpa4_nn/wignerd.py:191` | `DP_TRITON_INFER>=1` / `DP_CUTILE_INFER` |
| A18 | `gated_activation`（Triton，二阶） | Triton | 单个门控层非线性核的二阶导 | 不是省访存，是省"二阶导只能展开成物化面"的代价 | `dpa4_nn/activation.py:70` | `DP_TRITON_INFER>=1`（训练/力损失） |
| A19 | `grid_pair`（Triton，训练） | Triton | `to_grid·to_grid → from_grid` 的 GEMM-逐点-GEMM 三明治 | grid field 在 forward / backward / 二阶都不落盘；每个收缩都上 tensor core | `grid_net.py:45` | `DP_TRITON_INFER>=1` |
| A20 | `force_assembly`（Triton/cuTile） | Triton / cuTile | 力/维里装配：`F_k=Σ_{dst}k g − Σ_{src}k g`、`W_k=Σ_{src}k(−g⊗v)` | 3 次 `index_add` 原子散射 + `(E,9)` 外积物化 → 2 次 CSR 段归约；fp64 累加 | `model/transform_output.py:306`／`edge_transform_output.py` | `DP_TRITON_INFER>=1` / `DP_CUTILE_INFER` |

## 2. DPA1（`se_atten`）与 DPA4C

| # | 融合算子 | 层 | 融合了哪些算子 | 省掉的访存 / 收益 | 调用点 | 开关 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | `dpa1_graph_descriptor` | native | 整个 attention-free 描述符体：env matrix + 3 层 embedding MLP + strip type-pair gate + moment 归约 + `G^T G`，两个 megakernel（fwd/bwd） | 描述符中间量不再逐个物化；反向走注册的 `dpa1_graph_descriptor_backward` | `pt_expt/descriptor/dpa1.py` | `DP_CUDA_INFER>=1`（freeze 时选定） |
| B2 | `dpa1_graph_compress` | native | 压缩版：env + 五次样条表查 + strip gate + moment + `G^T G` | 同上，且把样条表查折进核 | 同上（geo-compress 分支） | `DP_CUDA_INFER>=1` |
| B3 | `dpa1_graph_energy_force` | native | 描述符 + 拟合 + 解析力/维里，**一个 graph node** | 去掉 autograd tape、逐分量 grad 循环、算子间的数组胶水；力作为返回值，数值与 level-1 一致 | `descriptor/dpa1.py` | `DP_CUDA_INFER>=2` |
| B4 | `dpa1c` / `dpa4c` `graph_compress` | native | DPA4C 整段描述符：radial 分支 + PairFiLM + 一次目标归约同时给出 envelope 质量与各 degree moment + readout + 输出标定 | 解析反向（一次 node-readout VJP + 一次 edge 重算扫描） | `descriptor/dpa4c.py` | `DP_CUDA_INFER>=1` |
| B5 | `dpa1/se_conv`、`dpa1/edge_conv` | Triton | 最后一层 embedding + resnet + type-pair gate + cutoff + 外积 + `segment_sum`（dense / graph 两种输入） | 把"最后一层 + 门 + 外积 + 归约"并成一段；graph 版把 neighbor 轴归约换成 `segment_sum` | `descriptor/dpa1.py`、`env_mat.py` | `DP_TRITON_INFER>=1` |
| B6 | `dpa1/gemm_fp16x3` | Triton | embedding 的 GEMM 走 fp16x3 | 省算力（到 tensor core），不改访存 | `descriptor/dpa1.py` | `DP_TRITON_INFER>=3` |

## 3. 跨描述符（通用）

| # | 融合算子 | 层 | 融合了哪些算子 | 省掉的访存 | 调用点 | 开关 |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | `edge_force_virial` | native（CUDA + CPU） | force / 每原子 virial / 每帧 virial 装配 | 替换 API 链 `index_add`×3 + 外积 + `segment_sum`；只读 CSR 两次，`(E,9)` 不物化 | `pt/model/model/transform_output.py:306`、`pt_expt/model/edge_transform_output.py` | `DP_CUDA_INFER>=1` |
| C2 | `graph_fitting` | native（CUDA + CPU） | 整个能量拟合网：GEMM + bias/激活/残差 epilogue | 每层激活不留到下一趟；fp64 输出 | `pt_expt/fitting/ener_fitting.py` | 与 `edge_force_virial` 同级 |
| C3 | `env_mat`（Triton） | Triton | 所有 `se` 族共用的平滑环境矩阵 | 一次算 `env` + `diff` + `sw`，不再每个消费者重算 | `descriptor/dpa1.py:118`、`dpa2/dpa3/se_r` 等 | `DP_TRITON_INFER>=1` |
| C4 | `aggregate` / `segment_sum` / `edge_csr` | Python | 分段归约 + CSR 视图（其余融合的共同基建） | `index_add_` 的原子散射换成段归约 | `network/utils.py:43`、`edge_transform_output.py:332` | 始终可用（参考层） |

## 4. 同一个 value path 的四套实现

| 实现 | 粒度 | 覆盖到哪一步 | 中间量 | 开关 | 已知性能 |
| --- | --- | --- | --- | --- | --- |
| Triton | 2 个算子（rotate_mix + mixing_stack）+ 旋转/块 GEMM/flash atten | 到 attention 加权之前 | 旋转中间量不物化，层间激活仍落盘 | `DP_TRITON_INFER>=1/2/3` | 上游的基线路径 |
| CuTe | 1 个 bucketed kernel + 重算反向 | 到 focus competition | 全部片上 | `DP_CUTE_INFER` | 在 H20 上比 Triton+flash **慢 2.8×**（489/724 vs 174/262 ms @2000/4000 原子），显存低 1.68×[^2] |
| cuTile | 3 个算子（rotate_mix / mixing_stack / flash_atten） | 到目标归约 | 全部片上 | `DP_CUTILE_INFER` | 无公开数字 |
| native CUDA | 1 个算子（含 softmax 与归约） | 到 head gate | 全部寄存器 | `DP_CUDA_INFER>=2` | PRO 6000 上：mini **+43%**、neo −4%、air **−9%**；H20 **0.79×**[^3] |

## 5. 收益来源只有五类

1. **去中间量**（最大的一类）：A1 3.6 GB/conv、A6 1.3 GB、A9 2.11 GB/block、A12 9 GB scratch、A5 39× 场。
   判据是"这段的 `字节/带宽` 明显小于实测时间"，与 [`execution-model.md`](execution-model.md) §6 同一条。
2. **去原子散射**：A6、A10、A15、A20、C1 —— 行原子在每原子约 10² 条边相撞时只有百 GB/s 级，
   换 CSR 段归约后既快又**求和顺序固定**（可复现）。
3. **去 gather / layout 物化**：A3（5 趟→3 趟）、A12（`D_to_m` 与 `x[src]` 不建）、A13（strided 权重 + 补齐浪费）。
4. **上 tensor core**：A14、B6（fp16x3 两段分裂补偿，误差 ~2⁻²²）。
5. **去 autograd**：B3（力作为值返回）、A15/A18 的闭式二阶、A2 的融合反向。

   "fwd/bwd 融合"的准确含义是**一个 autograd 边界 + 两个 kernel**，不是一次 launch：
   `torch.library.register_autograd("deepmd::dpa4_so2_conv", _backward, setup_context=_setup_context)`
   把 fwd kernel 和 bwd kernel 绑成图上的一个节点（`so2_conv.py:539`），于是
   (a) 段内中间量既不为前向也不为反向 materialize；
   (b) `_setup_context` 只 `save_for_backward` 全部小输入 + 三个 aux 输出
   （`alpha`/`pre_gate`/`z_all`），并用 `set_materialize_grads(False)` 免掉 aux 的零填充；
   (c) 二阶导不再把整条链展开一遍，而由 `second_order.py` 用多线性公式拼。
   三种实现存/算的取舍不同：CUDA 存 `z_all`；CuTe 只存两个门控层 pre-activation 并在同一核里重放；
   Triton 的 `so2_rotate_mix` 反向连旋转都重算（"nothing is saved besides the operator inputs"）。

## 6. 门与互斥

| 环境变量 | 取值 | 作用 |
| --- | --- | --- |
| `DP_CUDA_INFER` | 0/1/2 | 0 关；1 = 单点算子（wigner_dense、edge_radial、grid_pair、zonal_scatter、edge_force_virial、DPA1 描述符）；2 = 加上 `dpa4_so2_conv` 与 `dpa1_graph_energy_force` |
| `DP_TRITON_INFER` | 0/1/2/3 | 1 = 通用核 + 旋转/radial/块 GEMM/flash atten/force；2 = 加上查表的 `so2_value_path`；3 = 加上 fp16x3 |
| `DP_CUTE_INFER` / `DP_CUTILE_INFER` | bool | 与 Triton 三条路径**互斥**，同时开会构造期报错 |
| `DP_CUDA_TRAIN` | bool | A2 训练形态 |

## 7. 与本项目（myproject）的关系

| 我们的算子 | 上游等价物 | 结论 |
| --- | --- | --- |
| `force_assembly` | A20 / C1 | 同数学、同接口；我们的是 HIP 版 |
| `segment_softmax` | A15 | 同数学；且已被 A1 吸收 |
| `segment_sum` | C4 `aggregate`/`segment_sum` | 同数学 |
| `slot_pack_scale` | `GatedActivation` 的 gate 展开（没有独立上游算子，被 A1/A2/A9 吸收进核） | 上游没有对应"单点算子"，我们做的是把这段单独拉出来 |
| CSR `order` | `edge_csr`（A10/A15/A20 共用） | 同 |

也就是说：**除了 `slot_pack_scale`，我们的每个算子在上游都有同名实现**；价值不在算法，
而在"让它们在 legacy `pt` 后端、HIP 上可运行"。段级 megakernel（A1）的结论见
[`execution-model.md`](execution-model.md) §4.5 —— 本 checkpoint 是 `air` 形状，
上游自己的标定说它在这里**慢 9%**。

---

[^1]: `source/op/pt/dpa4/so2_conv_kernel.cuh` 文件头：*"The path they replace moves about 3.6 GB of
      `(E, ROW)` activation per convolution at the reference shape, which pins it to the DRAM roof at an
      arithmetic intensity of 24 FLOP/byte against a ridge point near 78."*
      自检：`E=181838, ROW=640, fp32` → 每次 pass 465 MB，约 8 个 pass ≈ 3.7 GB，与该数字吻合。
[^2]: `deepmd/pt_expt/kernels/cute/sezm/__init__.py` 的 "Performance" 一节（H20 / fp32）。
[^3]: `deepmd/pt_expt/kernels/cuda/dpa4/so2_conv.py:845-862`（`_MAX_PROFITABLE_LAYER_FMA` 与 `_RIDGE_REF`）。
