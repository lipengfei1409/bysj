# 优化项状态表

本表是"做了哪些优化、解决了什么、到哪一步、值多少"的唯一索引。每完成一项就更新对应行；
收益一律区分**实测**（真机 A/B 跑出来的数）与**估算**（按 trace 的池子 / roofline 折算的区间）。

口径与来源：

- 池子、候选排序、收益预算： [`task2-hip-fusion-plan.md`](task2-hip-fusion-plan.md)
- 每个算子的公式与 DeepMD 原位代码对照： [`fused-operators.md`](fused-operators.md)
- 接缝、前置条件、回退规则： [`deepmd-fused-dispatch.md`](deepmd-fused-dispatch.md)
- 前向 + 反向的算子时间占比、每个融合各吃了哪几行： [`op-time-share.md`](op-time-share.md)
- persistent 调度 vs megakernel 的选择依据与分档计划： [`execution-model.md`](execution-model.md)
- 数据基线：3000 原子 / 181838 边，2614 launch。**两个数要分开看**：仓库内 trace
  （`dpa4_inference_trace.json`）设备合计 1218.58 ms（前向 436.21 / 反向 780.38），
  真机 A/B 的基线是 964.4 ms（同一个 launch 数，机器状态不同）；占比用 trace，倍数用真机 A/B

状态图例： ⬜ 未做 · 🟡 已实现待真机验证 · ✅ 已实现且真机验证过 · ❌ 已验证无效（保留结论）

---

## 1. 判定与工具

| # | 优化项 | 解决的问题 | 方案 | 状态 | 收益 |
| --- | --- | --- | --- | --- | --- |
| A1 | Triton 可行性判定（D0） | 不确定是否该在 Triton 上调优 | 冒烟 + IR 分析：前端 LLVM 22 发 `getelementptr inbounds nuw`，gcvm（LLVM 17）解析失败并 SIGSEGV | ✅ 完成（判负） | 避免整周投入不可用路线 |
| A2 | trace 归因工具 | 不知道时间花在哪 | `analyze_trace.py`、`scan_fusion_patterns.py`（launch → aten op → 形状） | ✅ 完成 | 得出"64% 在反向、74.9% edge-major" |
| A3 | A/B 对比工具 | 融合后没有可信的端到端判据 | profile CLI `--fused` + `compare_fused_runs.py`（设备时间 / launch 数 / 逐 kernel 增量） | ✅ 完成 | 使"接缝是否生效"可观测 |
| A4 | 融合接缝框架 | 算子写好但 DeepMD 不调用 | `deepmd_patch()`：绑 DeepMD 自己的接缝，逐调用前置检查 + 回退 + 命中计数 | ✅ 完成 | 端到端接入的载体 |

## 2. 已实现的融合算子

| # | 优化项 | 解决的问题 | 方案 | 状态 | 收益 |
| --- | --- | --- | --- | --- | --- |
| B1 | `segment_sum`（分段归约） | `fill_`+`index_add_`+`mul_` 三次 launch；行原子只有 177 GB/s | 一 block 一段，CSR 遍历，累加器在寄存器；权重/归一作为 `edge_scale`/`node_scale` 进核 | 🟡 算子完成；接 `envelope`（本模型不走）与 `aggregate`（本模型走） | 算子级**未测**；trace 池 = 前向 E→N 散射 11.2 ms（3×3.7），630 GB/s 下 fusion 约 1.2 ms → **−10 ms**（原估 −15…−25 偏乐观） |
| B2 | `segment_softmax`（门控分段 softmax） | max + exp + 2 次散射 + 除，共 6 次 launch | 段内 max 与分母留寄存器；反向只读已存的 `α` | ✅ 完成并接入；端到端已测 | 算子级 **9.09×**（C=64）/ **5.00×**（C=256）；端到端 **≈0**（964.4 → 963.8 ms，−150 launch）。trace 佐证：本 trace 里根本没有 `_softmax` 行 |
| B3 | `force_assembly`（力/维里装配） | 2 次原子散射 + `(E,9)` 物化；源端维里外积 | 两次只读 CSR 遍历，double 累加，5 原子/block 填满 wave64 | 🟡 算子完成并接上上游 `edge_force_assembly`；**端到端价值已判负** | 算子级 **5.53×**（1.006 → 0.182 ms）；但 trace 里真实池只有 **1.7 ms**（`index_add [3000,3]`×2 + `[3000,9]` + 外积）→ 端到端 **−1.5 ms**，原估 −60…−85 是把 gate/槽位 gather 的反向误算进来了 |
| B4 | CSR 置换 `order` 支持 | DeepMD 边表不按位置有序，否则每个算子前要先物化排序结果 | kernel 内 `edge = order[slot]`，输出仍写自然序；`force_assembly` 另加 `dst_order` | 🟡 已实现（三算子 + 主机测试） | 省掉每步 2 趟 gather（约 1.2 ms/步） |

**注**：B1–B4 在上游都有同名实现 —— `aggregate`/`segment_sum`（Python）、
`triton/sezm/segment_softmax`、`triton/sezm/force_assembly` + `cuda/edge_force_virial`、
`triton/sezm/gated_activation`、以及 `edge_csr`。它们只在 CUDA-only 构建或 Triton 里可用，
所以我们的增量是"把它们接到 legacy `pt` 后端、跑在 HIP 上"，不是新算法。
上游全清单见 [`execution-model.md`](execution-model.md) §4.5。

## 3. 前向 / 反向热点（未实现）

| # | 优化项 | 解决的问题 | 方案 | 状态 | 收益（估算） |
| --- | --- | --- | --- | --- | --- |
| C1 | 槽打包 + 缩放（`slot_pack_scale`） | `GatedActivation.forward`（`activation.py:213`）里 `index_select`+`mul`+`copy_` 三次 E-major 遍历：`index_select [1,E,3,64] idx[9]` 20.5 ms、`mul [1,E,9,64]` 17.3 ms、`copy_` 15.6 ms（各 9 次） | 映射已确认：`gates = gating_scalars.index_select(2, expand_index)`，随后 `out[:,1:] = x[:,1:] * gates`。`slot_pack_scale(gating_scalars, expand_index, scale=x[:,1:])` 一趟完成"按 gate 组展开 + 逐元素乘" | 🟡 **已接线**（`fused=("pack",)`）：只替换 `fndc` 布局那一条分支（本模型就是它），另两条 layout 与非 fp32 直接回退原方法 | 估算 **−20…−25 ms**（末端 `narrow.copy_` 与 `x0` 拷贝仍在，只有 gather+mul 被合掉）；trace 池 53.4 ms；待真机 |
| C1b | 同一条链的**反向** | `index_select` 的反向：`index_add_ [1,E,3,64] ⬅ [1,E,9,64]` 32.7 ms（9 次）+ `fill_ [1,E,3,64]` 2.6 ms，跑成行原子 | 逐边独立、无竞争：一个 kernel 里按 9 个 slot→group 的映射把 `grad_gates` 累加进 `grad_gating_scalars` | ⬜ 未做（同一映射已有，`slot_pack_scale` 的反向可直接写） | 估算 **−20…−25 ms**（35.3 → 约 1.0 ms/次的读+写） |
| C2′ | 10 槽加权归约（`slot_reduce`） | `SO2Linear.forward`（`so2.py:812`）`rank>0` 分支的 `sum(dim=2)`：`(mixed * channel_basis).sum(dim=2)`，权重依赖 `(o, r, c)` | 原以为可映射成 `Σ_r w[e,r]·src[e,r,c]`；实际是 `Σ_r kernel[e,o,r]·x[e,r,c]·channel_basis[o,r,c]`，**权重依赖输出槽 o** | 🟡 算子完成，但**与真实调用点不匹配**（需要 rank-K 收缩 + 逐通道基底的 epilogue） | 本行不再计收益；真实可做的是把它归入 C7 的"rank-K 收缩 + epilogue"（见 C7） |
| C3 | GEMM + 残差 epilogue | 两个 E-major GEMM 仅 8.1 TFLOP/s（roofline 13.7），残差 `add [E,10,64]` 又是一趟 | DMK 分块 GEMM，epilogue 吸收残差 | ⬜ 未做（属 megakernel 线） | −30…−35 ms |
| C4 | 小轴归约重写 | `sum [E,10,1,64]` 有效带宽 56 GB/s（约 10× 浪费），单行 24.9 ms、该组合计 36.5 ms | 小轴映射到 lane，`__shfl` 归约，或并进 C2 | ⬜ 未做 | −20…−22 ms |
| C5 | readout 瘦 GEMM | readout 组 5 个 bmm 合计 63.9 ms，其中 `[1,152,48]×[1,48,576000]` 24.2 ms（K=48 算力空转） | `float4` 向量化 + 融合 scale/add epilogue | ⬜ 未做 | −6…−10 ms |
| C6 | 边流逐元素链融合 | E-major 逐元素/memcpy/init/cat 合计 **488.1 ms（40.1%）**，同一条 3–4 核链重复 9 次 | 项表驱动的多输入 FMAD，k 趟合 1 趟 | ⬜ 仅设计 | −180…−220 ms |
| C7 | SO(2) 消息链段级融合（含 rank-K 收缩） | **`G2 槽位段` 580.0 ms / 47.6%**：`[E,3/4/9/10/15/16,64]` 的展开、残差、gate，加 6 个 per-edge 小 `bmm` 与 `sum(dim=2)`；中间量全部落 HBM | **上游已有等价实现**：`deepmd::dpa4_so2_conv`（`source/op/pt/dpa4/so2_conv_{fwd,bwd}_c{32,64}_l{1..6}.cu`），寄存器驻留 activation、`__shfl` 广播、权重 LDS staging、在线 softmax；legacy `pt` 的 dispatch 也已就位（`sezm_nn/so2.py` Step 14，`DP_CUDA_INFER=2`） | 🟡 **不写新 kernel**；先做 1 周的单特化 hipify A/B 决定生死（理由见 [`execution-model.md`](execution-model.md) §4.5） | **未定**：本 checkpoint 由 trace 反解为 `CF=64, lmax=3`（`ROW=640`、`M0=256`、`M1=384`、`GATE=192` 四处吻合），per-edge FMA = 225.3 k，正是上游标定里 `air` 那个点，那里 fused 比 Triton 组合**慢 9%** |

## 4. 工程与质量

| # | 优化项 | 解决的问题 | 方案 | 状态 | 收益 |
| --- | --- | --- | --- | --- | --- |
| D1 | 仓库去冗余 | TileLang 21 模块、`so2_reduce` 全家、`kernels.gemm`、不可达 C++ 文件 | 删除 + 旧参数显式报错，`kernels.__all__` 收敛到 11 项 | ✅ 完成（约 −1800 行） | 维护成本，非性能 |
| D2 | 真机 bug 与守卫 | 6 个只在 DCU 暴露的 bug | 每个补主机可拦的测试；新增"DCU 测试陈旧 API 守卫" | ✅ 完成（主机 94 项） | 减少"只能在集群发现"的返工 |

---

## 5. 收益汇总

按 trace 口径（1218.58 ms 设备时间）逐项加总，**不是**各估算的简单相加：

| 阶段 | 内容 | 覆盖池子 | 收益 | 状态 |
| --- | --- | ---: | --- | --- |
| 已实测 | B2 `softmax` 接缝 | ~1 ms（0.1%） | **≈0**（−150 launch，时间不变） | ❌ 已证明在本 checkpoint 无效 |
| 待实测 | B3 `force` | 1.7 ms（0.1%） | **−1.5 ms**（原估 −60…−85 作废） | 🟡 代码已接，端到端价值已判负 |
| 待实测 | B1 `aggregate` | 11.2 ms（0.9%） | **−10 ms** | 🟡 代码已接，等真机 |
| 待实测 | C1 `slot_pack_scale`（前向） | 53.4 ms（4.4%） | **−20…−25 ms** | 🟡 接缝已挂上，等真机 |
| 未实现 | C1b 同一条链的反向 | 35.3 ms（2.9%） | −20…−25 ms | ⬜ 映射已有，可直接写 |
| 待重新设计 | C2′ `slot_reduce` | 0 | 0（不匹配真实调用点） | 🟡 并入 C7 的 rank-K 收缩 epilogue |
| 未实现 | C3 + C5 | 407 ms（33.4%） | −36…−45 ms | ⬜ 需 DMK GEMM |
| 未实现 | C6 + C7（全图池子） | 618.9 ms（50.8%） | −240…−300 ms | ⬜ 设计阶段 |

**已接线的四个接缝合计覆盖 66 ms / 5.4%，可实现 −32…−37 ms（约 1.03×）**；
把 C1b 补上到 −52…−62 ms（约 1.05×）。这正是"为什么端到端只看得到 1.0x"的原因：
真正的大头（40.1% 的 E-major 逐元素链 + 28.2% 的 GEMM）还在 C6/C7/C3 里。

## 6. 待验证清单（填 🟡 → ✅）

```bash
cd ~/private_data/lipengfei/bysj/myproject
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
MAX_JOBS=4 python3 setup.py build_ext --inplace

# 1. 算子级：三个算子的精度/梯度（含 force_assembly 的 dst_order 路径）
python3 -m unittest discover -s tests/csrc/accuracy -p 'test_segment_ops*' -v

# 2. 接缝 parity：与 DeepMD 自身实现对照（目前覆盖 softmax / envelope）
DP_RUN_FUSED_DEEPMD_PARITY=1 \
  python3 -m unittest discover -s tests/csrc/accuracy -p 'test_fused_deepmd_parity.py' -v

# 3. 端到端 A/B（--fused on = softmax + envelope + force + aggregate）
OUT=/root/private_data/lipengfei/test_result
python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
  --model-file "$MODEL" --input-npz "$NPZ" --fixed-input \
  --expected-dcu-name K500SM_AI --output-dir "$OUT/base"
python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
  --model-file "$MODEL" --input-npz "$NPZ" --fixed-input \
  --expected-dcu-name K500SM_AI --fused on --output-dir "$OUT/fused"
python3 tests/deepmd/profile/compare_fused_runs.py \
  --baseline "$OUT"/base/dpa4_*/dpa4_inference_profile_summary.json \
  --patched  "$OUT"/fused/dpa4_*/dpa4_inference_profile_summary.json
```

第 3 步的期望计数：`aggregate_hits = 3`、`force_hits = 1`、`softmax_hits = 3`，所有 `*_fallbacks = 0`。

## 7. 更新日志

| 日期 | 变更 |
| --- | --- |
| 2026-11-19 | 建表。A1–A4、B2（算子 + 接缝）、D1、D2 标记完成；B2 端到端实测 ≈0 并标注为"本 checkpoint 无效"；B1/B3/B4 标 🟡 待真机；C1–C7 为未实现或设计阶段 |
| 2026-11-19 | C1 落地为 `slot_pack_scale`、C2 修正归因后拆为 C2′（`slot_reduce`，已实现）+ 10 槽小矩阵乘（并入 C7）。原 C2 把 `bmm [E,10,10]` 归给 attention 是错的：它来自 `so2.py:848` 的 `einsum("eoic,eic->eoc")`，属 SO(2) mixing；attention 的 logits 是 `(q_edge*k_edge).sum(-1)`，本来就是 elementwise |
| 2026-11-19 | C1 接线完成：`deepmd_patch(pack=True)` 替换 `GatedActivation.forward`（只接管 `fndc` 布局），`fused=("pack",)` 可单独开关；主机侧用 `fndc` 假模块与 stock 方法逐位对照（3 个用例），另有 layout 不匹配与非 fp32 的回退用例 |
| 2026-11-19 | 用 launch→region 归因钉死两个真实站点：C1 的 `index_select`+`mul`+`copy_` 全在 **`GatedActivation.forward`**（`activation.py:213`，映射 `gates = gating_scalars.index_select(2, expand_index)`），不是 `so2_message`；C2′ 的 `sum [E,10,1,64]` 在 **`SO2Linear.forward`**（`so2.py:812`）的 `(mixed * channel_basis).sum(dim=2)`，权重依赖输出槽 → `slot_reduce` 不匹配。C1 估算由 −45…−55 下调到 −20…−25，C2′ 收益归零并并入 C7 |
| 2026-11-23 | 查清上游已有实现：`pt_expt/kernels` 下五套融合（native CUDA / CuTe / cuTile / Triton / Python），native 一律只编 `USE_CUDA_TOOLKIT AND DEEPMD_TORCH_HAS_CUDA`，DCU 构建里 `dpa4_so2_conv` 不存在 → 静默回退稠密路径；legacy `pt` 的 dispatch 反而已就位。由 trace 反解本 checkpoint 为 `CF=64/lmax=3`，正好是上游 `air` 标定点（225.3 kFMA），故 C7 的收益改为"待单特化 A/B 实测" |
| 2026-11-23 | 新增 [`execution-model.md`](execution-model.md)：按张量家族把 1218.6 ms 分成 G1–G9，只有 `G2 槽位段`（580.0 ms / 47.6%）值得做段级 megakernel（4 KB/边驻留、无跨 block 依赖），全模型 megakernel 不做；C7 的池子由 66 ms 修正为 580 ms |
| 2026-11-23 | 新增 `op_time_share.py` 与 [`op-time-share.md`](op-time-share.md)：按 `nn.Module: ModelWrapper_0` 作用域（用 **launch 时间**过滤）给出前向 + 反向全部算子的时间占比，2614/2614 kernel、1218.58 ms 全部在作用域内。**据此修正两处估算**：B3 的真实池只有 1.7 ms（原 −60…−85 ms 作废，那 95 ms 里是 gate/槽位 gather 的反向），B1 上限 −10 ms。同时新增 C1b（同一 gate 链的反向，35.3 ms 池） |

## 8. 边界说明

1. **B1/B3 的端到端收益已按 trace 重新标定**：`segment_sum` 的算子级延迟至今没有测到（首次 latency 运行被校验器 bug 中断，
   之后未重跑），它的池子是 11.2 ms，所以上限是 −10 ms；`force_assembly` 的算子级 5.53× 是真的，
   但池子只有 1.7 ms，端到端最多 −1.5 ms —— **继续在它上面投入端到端验证不划算**，算子保留作 CSR 基建。
2. **B2 的 ≈0 不是 bug**：该 checkpoint `n_atten_head=1`，被替换的是 `[3000,C]` 量级小张量，成本由 launch 主导；
   这条接缝的价值要在 attention 宽度更大的模型上才体现。trace 侧佐证：整张 trace 没有一行 `_softmax`。
3. **C6/C7 的收益是"池子"不是"承诺"**：按 trace 元素池子折算，落地后要减去融合核的新成本，因此一律给区间。
4. **C1 的映射已确认、接缝不在方法边界上**：`GatedActivation.forward` 里 `gates = gating_scalars.index_select(2, expand_index)`
   之后紧跟 `out.narrow(...).copy_(x.narrow(...) * gates)`，`slot_pack_scale(gating_scalars, expand_index, scale=x[:,1:])`
   语义完全对应；但该方法内部有 3 条 layout 分支（`fndc` / `ndfc` / `nfdc`），接线只能 fork，约 35 行 DeepMD 逻辑，
   换 **−20…−25 ms**（2.6%）。更划算的做法是让上游加一行 `self._gate_expand_fn` 钩子（类似 `_segment_softmax_fn`），
   我们只绑定——那需要 DeepMD 侧 2 行改动，已记录为待决。
   同一条链的**反向**（C1b，35.3 ms）比前向还便宜可做：每个边的 9 个 slot 互不冲突，不需要原子。
5. **C2′ 的 `slot_reduce` 没有对应站点**：真实调用是 `Σ_r kernel[e,o,r]·x[e,r,c]·channel_basis[o,r,c]`，
   权重随输出槽 `o` 变化，不是按 `(e, r)` 的加权和。要么把它重写成 rank-K 收缩（并入 C7），要么删掉这个算子。
6. **两份数的关系**：trace 1218.58 ms 与真机 A/B 基线 964.4 ms 是同一份计算（都是 2614 launch），
   差在机器状态/时钟。**占比一律用 trace，倍数一律用真机 A/B**，不要互相折算。

