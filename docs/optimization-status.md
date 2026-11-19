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
| C1 | 槽打包 + 缩放（`slot_pack_scale`） | `index_select`+`mul`+`copy_`+`fill_` 共 4 趟全量 E-major，前向 75.7 ms | `out[e,r,c] = slot[r]<0 ? 0 : src[e,slot[r],c]·scale[e,r,c]`，一趟写完（保留行由负 slot 直接置零，省掉 `fill_`） | 🟡 算子 + 参考实现 + 测试完成（`csrc/slot_ops.cpp`），未接 DeepMD | 估算 −45…−55 ms（`cat` 那 22.3 ms 仍需 GEMM 支持双源，未含在内） |
| C2′ | 10 槽加权归约（`slot_reduce`） | `sum [E,10,1,64]` 有效带宽 56 GB/s（约 10× 浪费），3 次 24.9 ms | `out[e,c] = Σ_r w[e,r]·src[e,r,c]`，一 block 一边，10 行留在寄存器 | 🟡 算子完成，未接 DeepMD | 估算 −20…−22 ms（按 630 GB/s 折算 3 次共约 2.2 ms） |
| C3 | GEMM + 残差 epilogue | 两个 E-major GEMM 仅 8.1 TFLOP/s（roofline 13.7），残差 `add [E,10,64]` 又是一趟 | DMK 分块 GEMM，epilogue 吸收残差 | ⬜ 未做（属 megakernel 线） | −30…−35 ms |
| C4 | 小轴归约重写 | `sum [E,10,1,64]` 有效带宽 56 GB/s（约 10× 浪费），24.9 ms | 小轴映射到 lane，`__shfl` 归约，或并进 C2 | ⬜ 未做 | −20…−22 ms |
| C5 | readout 瘦 GEMM | `bmm [1,152,48]×[1,48,576000]`，K=48 算力空转，17.3 ms | `float4` 向量化 + 融合 scale/add epilogue | ⬜ 未做 | −6…−10 ms |
| C6 | 边流逐元素链融合 | `add_/mul/copy_/fill_` 587 ms 池，同一条 3–4 核链重复 9 次 | 项表驱动的多输入 FMAD，k 趟合 1 趟 | ⬜ 仅设计 | −180…−220 ms |
| C7 | SO(2) 消息链全融合（含 10 槽小矩阵乘） | rotate → radial → GEMM → rotate-back 的中间量落 HBM；`bmm [E,10,10]`/`[E,16,10]`/`[E,10,16]`（31.5 ms）也在这里 | 旋转保持寄存器，走 `_flash_atten_fn` 接缝；10 槽按段进寄存器 | ⬜ 仅设计 | −60…−80 ms（含 10 槽 bmm 的 −25…−30） |

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
| 待实测 | C1 `slot_pack_scale` + C2′ `slot_reduce` | **−65…−77 ms** | 🟡 算子已落地，未接线、未测 |
| 未实现 | C4 小轴归约（若与 C2′ 合并则只剩一半） | −20…−22 ms | ⬜ 需新 kernel |
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

## 8. 边界说明

1. **B1/B3 的端到端收益仍是估算**：`segment_sum` 的算子级延迟至今没有测到（首次 latency 运行被校验器 bug 中断，之后未重跑）；
   `force` 接缝刚接上，还没在真机验证过。
2. **B2 的 ≈0 不是 bug**：该 checkpoint `n_atten_head=1`，被替换的是 `[3000,C]` 量级小张量，成本由 launch 主导；
   这条接缝的价值要在 attention 宽度更大的模型上才体现。
3. **C6/C7 的收益是"池子"不是"承诺"**：按 trace 元素池子折算，落地后要减去融合核的新成本，因此一律给区间。
4. **C1/C2′ 尚未接线**：两个算子在 `csrc/slot_ops.cpp`，参考实现与主机测试齐备，但没有像 B1–B3 那样挂到 DeepMD 的接缝上；
   接线需要先确认 `so2_message` 里 3→9→10 的 slot 映射与 scale 来源（`radial_feat` 的哪一维），否则宁可先不接。
   算子级延迟用 `DPA4_RUN_SLOT_LATENCY=1` 测，跑的是真实尺寸（181838 边 × 10 行 × 64 通道）。
