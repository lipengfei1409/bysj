# 任务一：一周执行计划（Triton 算子优化）

> 口径：单卡、单次推理、给定原子数；不做 LAMMPS 端到端。
> 前提：**所有验证都在 DCU 机器上做**（本机 macOS 只能改代码、不能跑）。

---

## 0. 本周交付定义

### 0.1 硬交付（无论中途撞到什么，周内必须交出来）

| 编号 | 交付物 | 说明 |
| --- | --- | --- |
| A1 | `docs/week1-env.md` | DCU/Torch/Triton/DTK 版本矩阵 + **Triton 在该 DCU 上能否出 kernel 的结论**（含 TTGIR 片段、是否发 MFMA、LDS 用量） |
| A2 | `baseline.csv` + 基线表 | `DP_COMPILE_INFER{0,1} × DP_TRITON_INFER{0,1,2,3}` 共 8 格 × 2 个原子数的 eval 延迟（同一口径，含 3 次重复） |
| A3 | `hotspots.md` | 每次 eval 的算子级时间表：kernel 名 / 调用次数 / µs / 占比 / 是否回落 default 配置 / 相对 roofline 的百分比 |
| A4 | DCU 设备表 | `tile_config_data.py` 里新增 DCU 设备 key 的配置表（由 sweep 生成，不手写） |
| A5 | 改动 patch | 隔离分支 `feat/dcu-triton-tuning`，按 commit 分开、可单独回滚；NVIDIA 路径零行为变化 |
| A6 | `docs/week1-triton-report.md` | 数据 + 结论 + 下周清单 |

### 0.2 目标值（相对"同口径 level 0 基线"）

| 指标 | 底线 | 期望 |
| --- | --- | --- |
| 单次 eval（3000 原子，`--no-profile` 中位数） | **1.5x** | 2.5x |
| top-5 热点 kernel 的 roofline 百分比 | 全部有测量值并记录 | 其中 ≥2 个达 40%+ |
| 数值 | energy/force/virial 对齐 baseline；全部单测通过 | 同 |

> 1.5x 是按"level 0 是稠密参考实现、1650 个 elementwise kernel 被融合掉"估的保守值，
> 拿到 A2 后立刻用真实数据修正。

### 0.3 本周明确不做

- fp16x3 / 3×TF32 的**新** sweep 与精度验证（需要 fp64 exactness 全扫，至少 2 天，放第二周）
- 结构性重写（BLOCK 重排、跨算子融合、手写 kernel）——这些是第二周 P2/P3
- LAMMPS、多卡、`pt → pt_expt` 迁移、任务二的 DMK 集成

---

## 1. 先固化度量口径（不统一口径，后面所有数字都不可比）

| 项 | 取值 |
| --- | --- |
| 输入 | 已有的 3000 原子 npz（profile 里用的那份）+ 12000 原子扩胞版 |
| 模式 | `--fixed-input`（禁止自动扩容，保证每次 shape 一致） |
| 延迟口径 | `--no-profile --iterations 10`，冷跑 3 次取中位数 |
| 热点口径 | `--profile` 跑 1 次，`--warmup 5` |
| 环境 | 显式写死 `DP_TRITON_INFER` / `DP_COMPILE_INFER` / `HIP_VISIBLE_DEVICES`；不开第二个 profiler |
| 记录 | torch / triton / DTK / `gcnArchName` / 驱动版本，写进每个 summary（现有 summary 已记录大部分） |
| 判据 | 只比同一口径下的相对值；绝对 ms 仅作参考 |

**注意**：现在 profile 脚本没有 `DP_TRITON_INFER` 开关，所以 D1 的第一件事就是加它，
否则"到底跑的哪一级"永远说不清。

---

## 2. 逐日计划

### D0（半天）：环境与可行性冒烟 —— 决定整周路线

**目的**：在投入任何改动之前，先回答"Triton 在这台 DCU 上到底能不能跑"。

**动作**

1. 环境采集（10 min）：
   ```bash
   source /opt/dtk/env.sh
   rocminfo | grep -m1 -E 'gfx[0-9a-f]+'
   python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)"
   python -c "import triton;print('triton', triton.__version__)"
   python -c "import deepmd;print(deepmd.__file__)"
   ```
2. **Triton 冒烟脚本**（`tests/deepmd/profile/triton_smoke.py`，**D0 判负后已删除**，结论见
   `docs/task2-hip-fusion-plan.md` 附录）：一个 pointwise
   kernel + 一个 64×64×64 `tl.dot`，然后打印：
   - `kernel.metadata.shared`（LDS 字节数，对照 64 KB 上限）
   - `kernel.asm['ttgir']` 里是否出现 `mfma` / `buffer_load` / `ds_write`，`tl.dot` 是否退化
   - `kernel.n_regs`（若可用）
   
   ```bash
   TRITON_ALWAYS_COMPILE=1 python tests/deepmd/profile/triton_smoke.py 2>&1 | tee /tmp/wk/smoke.log
   ```
3. 试一次 `sweep` 的 `--model` 模式能否加载现有模型（10 min，不跑全量）：
   ```bash
   python -c "from deepmd.pt_expt.kernels.triton.sezm.sweep_tile_configs import _load_model_params; print(_load_model_params('<MODEL>').keys())"
   ```
   失败不影响本周（见 D3 的退路），但早知道早改方案。

**产出**：A1（`docs/week1-env.md`）

**完成判据**：能回答三个问题 —— ① Triton 是否可用；② LDS 预算是多少、冒烟用掉多少；
③ `tl.dot` 有没有走到矩阵指令。

**卡住怎么办**：若冒烟直接失败 → 不必等 D1，立即跳到 §3 的 Plan B 分支评估。

---

### D1：打开 Triton，拿到真基线（本周最关键的一天）

**目的**：把"Triton 关着"的现有 profile 换成"Triton 开着"的可比基线。

**动作**

1. 改 `tests/deepmd/profile/test_dpa4_inference_profile.py`：
   - `dpa4` 子命令加 `--triton-infer {0,1,2,3}`、`--triton-train {0,1}`（默认 None = 不改环境）；
   - 在设置 `DP_COMPILE_INFER` 的同一处（约 L663）做 set/restore，**必须早于模型构造**；
   - summary 里记录 `triton_infer_level` / `triton_train_level` / `compile_infer`。
2. 改 `deepmd/pt_expt/kernels/triton/sezm/tile_configs.py`：
   - 在 `_lookup()`（L187）加 env-gated 打印（`DP_TRITON_CONFIG_LOG=1`），格式
     `family / key / device / config / source(表|default|runtime)`，每个 key 只打一次。
   这一步让"回落 default"从静默变可见，是 D3 的前提。
3. 跑 8 格矩阵（每格：1 次 profile + 3 次 no-profile）：
   ```bash
   for L in 0 1 2 3; do for C in 0 1; do
     FLAG=""; [ "$C" = 1 ] && FLAG="--compile"
     DP_TRITON_INFER=$L python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
       --model-file "$MODEL" --input-npz "$NPZ3000" --fixed-input \
       --device cuda:0 --expected-dcu-name '' $FLAG \
       --no-profile --iterations 10 --output-dir /tmp/wk/np_lv${L}c${C}
   done; done
   ```
   （profile 版把 `--no-profile --iterations 10` 换成默认 profiler 参数）
4. 用 `DP_TRITON_CONFIG_LOG=1` 重跑一次 level 2，把 resolve 日志存下来。

**产出**：A2、A3 的原始数据

**完成判据**（G1 见 §3）：能画出 `level 0→1→2→3` 的 eval 时间曲线；trace 里能看到 triton kernel
的名字与次数；resolve 日志里能看到至少 1 个 family 落到 default。

**卡住怎么办**：如果 level ≥1 时 trace 里**仍然没有任何 triton kernel**：
① 先确认 `DP_TRITON_INFER` 在模型构造前生效（打印一次 `triton_infer_level()`）；
② 再确认 `TRITON_AVAILABLE` 为 True（`from deepmd.pt_expt.kernels.triton.sezm import TRITON_AVAILABLE`）；
③ 都正常但 kernel 没出现 → 检查 trace 的 kernel 名过滤条件（triton kernel 名不含 `at::native::`，
形如 `xxx_kernel`），排除"看漏了"。仍然没有 → Plan B。

---

### D2：归因 —— 把时间落到 op 和配置上

**目的**：知道每一毫秒花在哪个 kernel、离 roofline 差多少，从而决定 D3/D4 的投入顺序。

**动作**

1. 从 trace 聚合全量 kernel（现有 summary 只给 top-40，这里要全量）：
   读 `dpa4_inference_trace.json` 里 `cat == "kernel"` 的事件，按 name 聚合 calls / dur，
   分三类统计：`triton（名字匹配 *_kernel 且不含 at::native）` / `aten elementwise&index` /
   `Tensile(GEMM)`。写成 `tests/deepmd/profile/analyze_trace.py`（约 50 行，可复用）。
2. 对 top-5 热点按 shape 估 roofline，算出"当前是 roofline 的百分之几"：
   - 访存型：`bytes = 读入 + 写出`（按 (E, C_wide) / (N, C) 推），除以实测有效带宽
     （D0 的拷贝微基准，或临时用 ~1 TB/s 估）；
   - GEMM 型：`2·M·N·K / 13.7 TFLOPS`（rocBLAS 实测值）。
   这一步的输出直接就是 `hotspots.md` 的核心列。
3. **并行支线（半天）**：用 `analyze_trace.py` 的 `cast_report` 把 `aten::to` / `_to_copy` /
   `copy_` 按"输入 dtype + 维度"分组，找可省的逐层冗余拷贝（fp32→fp32、同 shape 的
   `copy_`/`clone`/`contiguous`）。
   **已修正的前提**：现有 trace 里那 1.0 s 的 `aten::to` 其实是**一次阻塞式拷贝**
   （fp64 `[1,3000,1,1]`，24 KB，单次 1023 ms）——它是 GPU 追赶的同步点，不是转换成本。
   所以这条支线的真实收益是"消掉 10 ms 级的小拷贝"，不要按 1 s 估收益。

**产出**：`hotspots.md` + `analyze_trace.py` + cast 候选清单

**完成判据**（G2 见 §3）：能排出"本周最值得动"的 2–3 项，并给出每项的预期收益上限。

---

### D3：DCU 专用 launch config（本周确定性最高的一项）

**目的**：让 DCU 从"回落 default"变成"有实测最优配置"。

**动作**

1. 全量 sweep（会按 checkpoint 自动收集所有 shape key，输出可直接贴进 `tile_config_data.py` 的片段）：
   ```bash
   python -m deepmd.pt_expt.kernels.triton.sezm.sweep_tile_configs \
     --model "$MODEL" --level 2 --device cuda:0 2>&1 | tee /tmp/wk/sweep_l2.log
   ```
2. 把输出的 fragment 作为新设备 key（`K500SM_AI`）合并进
   `deepmd/pt_expt/kernels/triton/sezm/tile_config_data.py`；匹配规则是"精确名优先、其次最长前缀"，
   所以只要 key 写对就不会影响 NVIDIA。
3. 加 **LDS 守卫**（在 sweep 的候选过滤处）：`BLOCK_* × dtype_bytes × num_stages ≤ 64 KB`，
   超限直接剔除。没有这层保护，扫出来的"最优"可能在 DCU 上根本没跑到（OutOfResources 后静默回落）。
4. 重跑 D1 的 level 2（和 level 3，如果精度门禁允许），对比前后。

**产出**：A4 + 前后对比表

**完成判据**：至少 2 个 family 的配置与 default 不同，且对应 kernel 的 µs 下降 ≥10%；
`hotspots.md` 里"回落 default"的行消失或大幅减少。

**卡住怎么办**：`--model` 模式加载失败（现有模型是 legacy `pt` checkpoint，可能没有
`_extra_state`）→ 退回手工模式，用 D1/D2 里日志收集到的 key 逐个扫：
```bash
python -m deepmd.pt_expt.kernels.triton.sezm.sweep_tile_configs \
  --cf <Cf> --lmax <lmax> --focus <F> --heads <H> --device cuda:0
```
（`Cf/lmax/F/H` 从 `collect_model_shape_keys` 或模型 config 里拿。）

---

### D4：2–3 个结构类快赢（按 D2 的排序取，不要贪多）

候选清单，按 DCU 上的典型收益排序，**只做 D2 数据支持的那几个**：

| # | 动作 | 落点 | 风险 |
| --- | --- | --- | --- |
| 1 | **消掉冗余 cast / 中间 clone**（支线 3 的结果） | 模型侧 Python，不在 triton kernel 里 | 低，纯等价替换 |
| 2 | **索引宽度 int32**（`E×C_wide < 2^31` 时） | `so2_value_path.py` / `so2_rotation.py` / `force_assembly.py` 的 offset 计算 | 中，需要守卫 + 守卫失败回退 int64 |
| 3 | **`num_warps` / `num_stages` 扩展**（wave64 + 64 KB LDS 下重扫） | `segment_softmax.py`(L455/502/552 固定 `num_warps=1`)、`force_assembly.py`(L182/195) | 低，只改 launch schedule |
| 4 | `tl.dot` 精度路径（仅当 D0 确认 `ieee` 在 DCU 上退化成 FMA） | `so2_rotation.py` / `so2_block_gemm.py` / `gated_activation.py` 的 `input_precision` | **高**，改数值，本周只在通过 op 级对照后尝试 |

**每项的闭环**：改 → 单测（`tests/python` 里对应 op 的 reference 对照）→ 整模 energy/force/virial
对照 baseline → 计时。每项一个独立 commit。

**产出**：A5 + 每项的前后数字

**完成判据**（G3 见 §3）：至少 2 项落地且合计贡献 ≥10%；周目标达成度可判定。

---

### D5：复测、报告、收口

**动作**

1. 全量复测：2 个原子数 × level{0, 最优 level} × 5 次，出最终表。
2. 回归：`python -m unittest discover -s tests/deepmd -p 'test*.py'` + `PYTHONPATH=python python -m unittest discover -s tests/python`。
3. 写 A6：改了什么 / 前后数字 / 没做成什么 / 下周 P2-P3 清单。
4. 整理分支：`feat/dcu-triton-tuning`，commit 拆分，`git log --oneline` 能一眼看出每一步。

**完成判据**：报告能让第三个人按命令复现所有数字。

### D6（缓冲）：没达标的补测，或提前启动第二周的结构类改造

---

## 3. 三个 Gate 与 Plan B

### G1（D1 末）：Triton 是否可用

| 结果 | 判断 | 下一步 |
| --- | --- | --- |
| level1 的 eval 时间 ≤ 0.8×level0 | 通过 | 按 D2 继续 |
| triton kernel 出现，但 level1 比 level0 **更慢** | 弱通过（配置/后端问题） | 继续 D2-D3，D3 的权重加大 |
| 没有任何 triton kernel / 崩溃 / 编译失败 | **不通过** | 立刻转 Plan B |

### G2（D2 末）：本周收益在哪

- 若"回落 default 的 family ≥ 3 个" → D3 占 D3-D4 的 70%。
- 若"roofline% 普遍 < 30% 且配置正常" → 说明是 kernel 结构问题，D3 收益有限，
  改为把 D4 的 cast/索引项做完，并把结构问题写进第二周清单。

### G3（D4 末）：是否达标

- 达到 1.5x → D5 只做复测与报告。
- 未达到 → D5 **不再加新改动**，把差距与原因写清楚（这比超期硬调更有价值）。

### Plan B（G1 不通过时，本周改交什么）

分三档，按严重程度：

| 档 | 现象 | 本周改交 |
| --- | --- | --- |
| B1 | 部分 family 不可用 | 跳过该 family，其余照常调 config；把不可用的证据（编译错误 / TTGIR）写进报告，转给任务二的 DMK 接手 |
| B2 | Triton 在该 DCU 上完全不可用 | 本周转向：① 消逐层冗余拷贝（fp32→fp32 的 copy_/clone/contiguous，10 ms 级；注意 CPU 侧那 1 s 是同步点，不是拷贝成本）；② 用 `csrc` 现有 HIP 能力把 **top-1 热点算子**写成原生实现，走 `python/model/dpa4.py` 的 patch 机制替换；③ 输出每个 op 的 shape/时间/roofline 清单，作为任务二的输入 |
| B3 | 环境层面阻塞（驱动/Torch/Triton 版本不匹配） | 本周只交 A1 + A2 的 level 0 基线 + 环境问题定位，并给出升级/降级方案 |

**Plan B 下的周目标**：从"1.5x"下调为"接口开销项落地 + 1 个热点算子被替换 + 归因清单完整"。

---

## 4. 本周改动文件清单（预估）

| 文件 | 改动 | 规模 |
| --- | --- | --- |
| `tests/deepmd/profile/test_dpa4_inference_profile.py` | 加 `--triton-infer` / `--triton-train` 透传 + summary 字段 | ~30 行 |
| `deepmd/pt_expt/kernels/triton/sezm/tile_configs.py` | `_lookup` resolve 日志 | ~10 行 |
| `deepmd/pt_expt/kernels/triton/sezm/tile_config_data.py` | 新增 DCU 设备表（由 sweep 生成） | 表数据 |
| `deepmd/pt_expt/kernels/triton/sezm/sweep_tile_configs.py` | LDS ≤64 KB 候选守卫 | ~15 行 |
| ~~`tests/deepmd/profile/triton_smoke.py`~~（已删除） | Triton 冒烟 + TTGIR/LDS 检查 | D0 判负后移除 |
| `tests/deepmd/profile/analyze_trace.py`（新） | trace 全量聚合 + roofline 计算 | ~50 行 |
| `so2_value_path.py` / `so2_rotation.py` / `segment_softmax.py` / `force_assembly.py` | D4 选中的 2–3 项 | 每项 ~10–30 行 |
| `docs/week1-env.md` / `week1-triton-report.md`（新） | 报告 | — |

---

## 5. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
| --- | --- | --- | --- |
| Triton 在 gfx928 上不可用 | 中 | 整周路线要改 | D0 冒烟提前暴露；Plan B 三档已备 |
| Triton 可用但 `tl.dot` 退化 FMA | 中高 | GEMM 类算子天花板低 | D0 看 TTGIR；本周不改精度，留给第二周的 fp16x3/3×TF32 |
| 扫描出的配置在 DCU 上跑不到（LDS 超限静默回落） | 高 | 调了等于没调 | D3 的 LDS 守卫 + resolve 日志双保险 |
| 现有模型是 legacy `pt`，sweep `--model` 加载不了 | 中 | D3 退化为手工扫 | 已有退路（`--cf/--lmax`），D0 先试加载 |
| 数值不达标（MMA 累加顺序 / 精度路径） | 中 | D4 第 4 项做不了 | 该项标记为"仅当通过 op 对照才做"，否则本周不动 |
| 只有一个人 + 只有一台 DCU | 高 | 串行排队 | D2 的 cast 支线可与主线和别人并行；sweep 是长时间单机任务，安排在夜间跑 |

---

## 6. 我这个环境能先做的（不需要 DCU）

D1 的两处代码改动（`--triton-infer` 透传、`_lookup` resolve 日志）、D0 的 `triton_smoke.py`、
D2 的 `analyze_trace.py`，以及 D3 的 LDS 守卫，都是纯 CPU 可写、可 review 的改动。
提前写好，D1 就只剩"跑命令 + 看结果"，一天能压到半天。

---

## 附：GEMM 对照基准（Triton vs rocBLAS）

`tests/dcu/test_triton_gemm_compare.py` 用同一份张量、同一个 stream，把 `tl.dot` 的
tiled GEMM 与直接 `rocblas_sgemm` 逐形状对比：ms、TFLOP/s、`triton/rocblas` 比值，
以及每个形状的数值对照（防止"快但算错"的配置胜出）。覆盖五类：

| `--kinds` | 覆盖 |
| --- | --- |
| `square` | 128 … 8192 的 `N×N×N` |
| `rect` | 长宽比极端（tall-skinny / wide-short）与奇数尺寸 |
| `ktail` | `K` 落在 tile 边界附近（1、3、31、33、100、1000、4097） |
| `dpa4` | 模型实际发出的 `(E,C)×(C,C)`：E ∈ {3000, 36000, 181838}，C ∈ {32…512} |
| `batch` | `bmm`，batch 8 / 64 / 512 |

```bash
python tests/dcu/test_triton_gemm_compare.py --kinds all --precision ieee \
  --output-json /tmp/triton-vs-rocblas.json
# 只看某几个形状，或与 tf32 对照
python tests/dcu/test_triton_gemm_compare.py --cases 2048,2048,2048;181838,128,128
python tests/dcu/test_triton_gemm_compare.py --kinds dpa4 --precision tf32
```

用途：D2 的 roofline 列可以直接用它给出的"同一形状下厂商库能跑多快"当分母，避免用
2048³ 的峰值去套所有形状；`--precision ieee` 与 `tf32` 的差值是 `tl.dot` 是否走到
矩阵指令的量化证据（也正是任务二 GEMM 要跨过的那道坎）。
