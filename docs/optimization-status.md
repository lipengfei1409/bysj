# 优化项状态表

本表是"做了哪些优化、解决了什么、到哪一步、值多少"的唯一索引。每完成一项就更新对应行；
收益一律区分**实测**（真机 A/B 跑出来的数）与**估算**（按 trace 的池子 / roofline 折算的区间）。

口径与来源：

- 池子、候选排序、收益预算： [`task2-hip-fusion-plan.md`](task2-hip-fusion-plan.md)
- 每个算子的公式与 DeepMD 原位代码对照： [`fused-operators.md`](fused-operators.md)
- 接缝、前置条件、回退规则： [`deepmd-fused-dispatch.md`](deepmd-fused-dispatch.md)
- 数据基线：3000 原子 / 181838 边，设备总时间 964.4 ms / 2614 launch（`pt` 后端，`n_atten_head=1`）

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
| B1 | `segment_sum`（分段归约） | `fill_`+`index_add_`+`mul_` 三次 launch；行原子只有 177 GB/s | 一 block 一段，CSR 遍历，累加器在寄存器；权重/归一作为 `edge_scale`/`node_scale` 进核 | 🟡 算子完成；接 `envelope`（本模型不走）与 `aggregate`（本模型走） | 算子级**未测**；`aggregate` 估算 −15…−25 ms |
| B2 | `segment_softmax`（门控分段 softmax） | max + exp + 2 次散射 + 除，共 6 次 launch | 段内 max 与分母留寄存器；反向只读已存的 `α` | ✅ 完成并接入；端到端已测 | 算子级 **9.09×**（C=64）/ **5.00×**（C=256）；端到端 **≈0**（964.4 → 963.8 ms，−150 launch） |
| B3 | `force_assembly`（力/维里装配） | 2 次原子散射 + `(E,9)` 物化，约 95 ms 池 | 两次只读 CSR 遍历，double 累加，5 原子/block 填满 wave64 | 🟡 算子完成并接上上游 `edge_force_assembly`，端到端待测 | 算子级 **5.53×**；端到端估算 **−60…−85 ms** |
| B4 | CSR 置换 `order` 支持 | DeepMD 边表不按位置有序，否则每个算子前要先物化排序结果 | kernel 内 `edge = order[slot]`，输出仍写自然序；`force_assembly` 另加 `dst_order` | 🟡 已实现（三算子 + 主机测试） | 省掉每步 2 趟 gather（约 1.2 ms/步） |

## 3. 前向 / 反向热点（未实现）

| # | 优化项 | 解决的问题 | 方案 | 状态 | 收益（估算） |
| --- | --- | --- | --- | --- | --- |
| C1 | 槽打包 + 缩放（`slot_pack_scale`） | `GatedActivation.forward`（`activation.py:213`）里 `index_select`+`mul`+`copy_` 三次 E-major 遍历：`index_select [1,E,3,64] idx[9]` 20.5 ms、`mul [1,E,9,64]` 17.3 ms、`copy_` 15.6 ms（各 9 次） | 映射已确认：`gates = gating_scalars.index_select(2, expand_index)`，随后 `out[:,1:] = x[:,1:] * gates`。`slot_pack_scale(gating_scalars, expand_index, scale=x[:,1:])` 一趟完成"按 gate 组展开 + 逐元素乘" | 🟡 **已接线**（`fused=("pack",)`）：只替换 `fndc` 布局那一条分支（本模型就是它），另两条 layout 与非 fp32 直接回退原方法 | 估算 **−20…−25 ms**（末端 `narrow.copy_` 与 `x0` 拷贝仍在，只有 gather+mul 被合掉）；待真机 |
| C2′ | 10 槽加权归约（`slot_reduce`） | `SO2Linear.forward`（`so2.py:812`）`rank>0` 分支的 `sum(dim=2)`：`(mixed * channel_basis).sum(dim=2)`，权重依赖 `(o, r, c)` | 原以为可映射成 `Σ_r w[e,r]·src[e,r,c]`；实际是 `Σ_r kernel[e,o,r]·x[e,r,c]·channel_basis[o,r,c]`，**权重依赖输出槽 o** | 🟡 算子完成，但**与真实调用点不匹配**（需要 rank-K 收缩 + 逐通道基底的 epilogue） | 本行不再计收益；真实可做的是把它归入 C7 的"rank-K 收缩 + epilogue"（见 C7） |
| C3 | GEMM + 残差 epilogue | 两个 E-major GEMM 仅 8.1 TFLOP/s（roofline 13.7），残差 `add [E,10,64]` 又是一趟 | DMK 分块 GEMM，epilogue 吸收残差 | ⬜ 未做（属 megakernel 线） | −30…−35 ms |
| C4 | 小轴归约重写 | `sum [E,10,1,64]` 有效带宽 56 GB/s（约 10× 浪费），24.9 ms | 小轴映射到 lane，`__shfl` 归约，或并进 C2 | ⬜ 未做 | −20…−22 ms |
| C5 | readout 瘦 GEMM | `bmm [1,152,48]×[1,48,576000]`，K=48 算力空转，17.3 ms | `float4` 向量化 + 融合 scale/add epilogue | ⬜ 未做 | −6…−10 ms |
| C6 | 边流逐元素链融合 | `add_/mul/copy_/fill_` 587 ms 池，同一条 3–4 核链重复 9 次 | 项表驱动的多输入 FMAD，k 趟合 1 趟 | ⬜ 仅设计 | −180…−220 ms |
| C7 | SO(2) 消息链全融合（含 rank-K 收缩） | rotate → radial → `SO2Linear` 的 `bmm(E,10,16)` + `*channel_basis` + `sum(dim=2)`（合计约 66 ms）→ rotate-back 的中间量落 HBM | 旋转保持寄存器；把 `einsum + 逐通道基底 + 归约` 合成一个 rank-K 收缩 kernel，走 `_flash_atten_fn` / 自建接缝 | ⬜ 仅设计 | −60…−80 ms（含 rank-K 收缩的 −35…−45） |

## 4. 工程与质量

| # | 优化项 | 解决的问题 | 方案 | 状态 | 收益 |
| --- | --- | --- | --- | --- | --- |
| D1 | 仓库去冗余 | TileLang 21 模块、`so2_reduce` 全家、`kernels.gemm`、不可达 C++ 文件 | 删除 + 旧参数显式报错，`kernels.__all__` 收敛到 11 项 | ✅ 完成（约 −1800 行） | 维护成本，非性能 |
| D2 | 真机 bug 与守卫 | 6 个只在 DCU 暴露的 bug | 每个补主机可拦的测试；新增"DCU 测试陈旧 API 守卫" | ✅ 完成（主机 94 项） | 减少"只能在集群发现"的返工 |

---

## 5. 收益汇总

| 阶段 | 内容 | 收益 | 状态 |
| --- | --- | --- | --- |
| 已实测 | B2 `softmax` 接缝 | **≈0**（−150 launch，时间不变） | ❌ 已证明在本 checkpoint 无效 |
| 待实测 | B3 `force` + B1 `aggregate` | **−75…−110 ms**（1.08–1.12×） | 🟡 代码已接，等真机 |
| 待实测 | C1 `slot_pack_scale`（`fndc` 分支已接线） | **−20…−25 ms** | 🟡 接缝已挂上，等真机 |
| 待重新设计 | C2′ `slot_reduce` | 0（不匹配真实调用点） | 🟡 算子存在但无对应站点；真实站点需要 rank-K 收缩 epilogue（并入 C7） |
| 未实现 | C3 + C5 | −36…−45 ms | ⬜ 需 DMK GEMM |
| 未实现 | C6 + C7（全图池子） | −240…−300 ms | ⬜ 设计阶段 |

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

## 8. 边界说明

1. **B1/B3 的端到端收益仍是估算**：`segment_sum` 的算子级延迟至今没有测到（首次 latency 运行被校验器 bug 中断，之后未重跑）；
   `force` 接缝刚接上，还没在真机验证过。
2. **B2 的 ≈0 不是 bug**：该 checkpoint `n_atten_head=1`，被替换的是 `[3000,C]` 量级小张量，成本由 launch 主导；
   这条接缝的价值要在 attention 宽度更大的模型上才体现。
3. **C6/C7 的收益是"池子"不是"承诺"**：按 trace 元素池子折算，落地后要减去融合核的新成本，因此一律给区间。
4. **C1 的映射已确认、接缝不在方法边界上**：`GatedActivation.forward` 里 `gates = gating_scalars.index_select(2, expand_index)`
   之后紧跟 `out.narrow(...).copy_(x.narrow(...) * gates)`，`slot_pack_scale(gating_scalars, expand_index, scale=x[:,1:])`
   语义完全对应；但该方法内部有 3 条 layout 分支（`fndc` / `ndfc` / `nfdc`），接线只能 fork，约 35 行 DeepMD 逻辑，
   换 **−20…−25 ms**（2.6%）。更划算的做法是让上游加一行 `self._gate_expand_fn` 钩子（类似 `_segment_softmax_fn`），
   我们只绑定——那需要 DeepMD 侧 2 行改动，已记录为待决。
5. **C2′ 的 `slot_reduce` 没有对应站点**：真实调用是 `Σ_r kernel[e,o,r]·x[e,r,c]·channel_basis[o,r,c]`，
   权重随输出槽 `o` 变化，不是按 `(e, r)` 的加权和。要么把它重写成 rank-K 收缩（并入 C7），要么删掉这个算子。
