# 下一步计划：Triton 算子优化 + DCU Megakernel 框架

本文基于对 `myproject`、`deepmd-kit`、`dtk_doc`、`Megakernels` 四个目录的代码走读，以及已有
profile 结果（`dpa4_inference_profile_summary.json`、`dpa4_inference_trace.json`）写成。

---

## 0. 先对齐现状（从代码与日志确认的事实）

| # | 事实 | 依据 |
| --- | --- | --- |
| 0.1 | 已有 profile 是 **legacy `pt` 后端 + Triton 未生效**。trace 里 2614 个 kernel 全是 `at::native::*` 与 Tensile GEMM，没有任何 triton kernel；profile CLI 也没有 `DP_TRITON_INFER` 开关。 | `dpa4_inference_profile_summary.json`（`backend: pt`）、`tests/deepmd/profile/test_dpa4_inference_profile.py` |
| 0.2 | 3000 原子单次 eval：设备时间 **1279.5 ms**，2614 次 launch，峰值显存 **30.7 GB**。其中 elementwise / reduce / index / gather 类 **≈850 ms（1650 次）**，Tensile SGEMM **≈327 ms（165 次）**。 | 同上 `trace_analysis` |
| 0.3 | CPU 侧 `aten::copy_` / `aten::to` / `aten::_to_copy` 各约 1.0 s（互相重叠），但追下去是**一次阻塞式拷贝**（fp64 `[1,3000,1,1]` 单次 1023 ms）：它是 GPU 追赶的同步点，不是 fp64↔fp32 的转换成本。可省的是逐层 fp32→fp32 冗余拷贝（10 ms 级）。 | `top_cpu_ops` + `analyze_trace.py` 的 `cast_report` |
| 0.4 | 结论：当前 66% 的设备时间在**访存型 elementwise/index**，26% 在 GEMM（且 GEMM 已由 rocBLAS 跑到接近峰值）。加速空间主要在**融合**，而不是把 GEMM 写得更快。 | 0.2 的换算 |
| 0.5 | Triton 的 launch config 表只收 NVIDIA 设备名，DCU（`K500SM_AI` / `K100_AI`）会**静默回落到 conservative default**，等于没有调优。 | `deepmd-kit/deepmd/pt_expt/kernels/triton/sezm/tile_config_data.py` |
| 0.6 | 多个 kernel 固定 `num_warps=1/2`、`BLOCK_M` 偏小（pointwise/softmax/force assemble），是按 32 线程 warp 与 228 KB LDS 的假设写的；DCU 是 wave64 + 64 KB LDS + 32 bank×4B。 | `triton/sezm/segment_softmax.py`、`force_assembly.py`、`so2_value_path.py` |
| 0.7 | `DP_TRITON_INFER` 分 4 级（0 关 / 1 通用 kernel / 2 查表 tile config / 3 fp16x3），`DP_TRITON_TRAIN` 独立；这是现成的分级灰度开关。 | `deepmd-kit/deepmd/pt_expt/kernels/utils.py` |
| 0.8 | DMK GEMM 现状：16×16×4 单 wave 一块、k 步长 4、**每批 12 个 k（3 个 stage）一次 `__syncthreads`**（k=2048 时单块 171 次屏障、512 次 MMA）、标量 4B 访存、`tile_indices()` 里每元素做 **64-bit 除/取模**、epilogue 也是标量；`ShmemSwizzleLayout` 已实现，但取片仍是 row-major 裸 `ld_matrix`，没有接 swizzle。 | `csrc/dmk/gemm.h`、`csrc/dmk/tensor.h`、`csrc/dmk/copy.h` |
| 0.9 | DUMMA FP32 支持 `m16n16k4`（gfx926/928/936/938）与 **`m16n16k8`（gfx928/936/938）**；独立 `GemmOp<T,K>` 现在把 K 作为模板参数（float k4/k8、half/bf16 k16），page 版 `GemmOp` 仍是标量参考。 | `dtk_doc/DTK 26.04 DUMMA使用手册.pdf` 表 3.1 |
| 0.10 | megakernel 脚手架仍是空壳：`megakernel.cpp` 里 `uint` 未定义、`WARP_SZIE` 拼写错误、`launch_mk()` 空实现；`rmsnorm.h` / `attention.h` 语法错误不能编译；`embedding.h` 为空；`setup.py` 不编译 dmk；`bindings.cpp` 无 dmk 入口。 | `csrc/megakernel.cpp`、`csrc/dmk/rmsnorm.h`、`csrc/setup.py` |
| 0.11 | 已有可复用测试设施：GEMM 自带 rocBLAS 对照的 C++ 用例、hipcc 直编 runner。 | （随 DUMMA 版 GEMM 一起删除；`GemmOp` 现在只有持久化形态） |
| 0.12 | 参考实现是五角色（controller / loader / consumer / storer / launcher）+ instruction pipeline 的持久化 megakernel；DMK 的 `BaseOp<Globals>` 现在提供 release_lid / init_semaphores / load / run / store 五个多态 hook（`load/run/store` 与角色无关，由 kernel 把 role 映射过去），controller 与 worker 直接虚调用。 | `Megakernels/include/megakernel.cuh`、`csrc/dmk/op.h` |
| 0.13 | DCU 侧硬约束：无 TMA / mbarrier → 双缓冲与信号量要自己做；`hipLaunchCooperativeKernel` 在 DTK 26.04 手册中有，可作为 grid 同步候选；DUSHMEM 是多卡 PGAS，**不能**用来做单卡 CTA 间同步。 | `dtk_doc` 各手册 |

> 关于 0.1：不能只凭 summary 判断。第一步要用显式 `DP_TRITON_INFER=1/2/3` 复测一次，排除
> "环境变量没设 / 级别被别处覆盖" 的可能。如果设了还是没出现 triton kernel，那就是
> **Triton 在这台 DCU 上根本没跑起来**，任务一的技术路线要立刻调整（见风险 R1）。

---

## 1. 任务一：优化 deepmd-kit 已有 Triton 算子

**目标口径**：单卡、单次推理、给定原子数（3000 / 12000 / 36000），暂不做 LAMMPS 端到端。
**主指标**：单次 `DeepPot.eval` 的设备时间（`--no-profile` 墙钟 + `--profile` 设备时间双口径）、
每个 triton kernel 的 µs 与它相对 memory roofline 的百分比。

### P0（1–2 天）把基线做成"可信、可比、可归因"

1. 给 profile CLI 加两个透传开关，**注入点必须在构造模型之前**（和 `DP_COMPILE_INFER` 同一处）：
   `--triton-infer {0,1,2,3}` → `DP_TRITON_INFER`，`--triton-train {0,1}` → `DP_TRITON_TRAIN`；
   并把解析结果写进 summary（`triton_infer_level`），避免以后再出现"不知道跑的是哪一级"。
2. 固定口径：`--fixed-input` + fp32 输入 + 固定原子数；`--no-profile` 取延迟中位数（≥10 次），
   `--profile` 只取一次做热点归因。
3. 跑 2×4 矩阵：`DP_COMPILE_INFER ∈ {0,1}` × `DP_TRITON_INFER ∈ {0,1,2,3}`，产出表格。
   - level 1→2 的差值 = launch config 表的潜在收益（P1 的上限）；
   - level 2→3 的差值 = fp16x3 的收益，但要先过精度门禁；
   - `compile=1` 在 HIP 上是否可用，顺手验证（README 说明上游要求 torch 2.11/CUDA ≥12.6）。
4. 量化 0.3 的接口开销（已修正：那 1.0 s 是同步点，不是转换成本）：用 `analyze_trace.py`
   的 `cast_report` 按 dtype/维度分组，只挑 fp32→fp32 的同 shape 冗余拷贝作为可省项。
5. 冒烟检查（如果 P0.3 里 level≥1 仍然没有 triton kernel）：写一个最小脚本，只做一次
   `tl.dot` + 一次 pointwise，确认 Triton 能否为该 DCU 出 kernel、LDS 是否够、`num_warps`
   语义是否按 64 线程解释。这一步失败会改变整个任务一的路线。

**产出**：`docs/dpa4-triton-baseline.md` + `baseline.csv`；一张"热点算子 → 时间 → 占比"的表。

### P1（3–5 天）DCU 专用 launch config（低风险、收益确定）

1. 用现成的 sweep 基础设施在 DCU 上按 shape key 扫：`tune_missing_configs`，把赢家写进
   `tile_config_data.py` 的新设备表（key 用 `K500SM_AI` / `K100_AI`，遵循现有"精确名优先、
   其次最长前缀"的规则）。
2. 加 **LDS 预算守卫**：候选 config 必须满足 `BLOCK_* × dtype_bytes × num_stages ≤ 64 KB`，
   不满足直接剔除。不这么做的后果是 `OutOfResources` 后静默回落到保守配置，扫描结果看起来
   "差不多"，实际什么都没调到。
3. 复核 `num_warps`：AMD 一个 warp 是 64 线程，把 `num_warps=1` 的 pointwise / segment_softmax /
   force_assembly 扩到 2/4 重扫，确认在 64 KB LDS 与 64 lane 下是否更优。
4. 增加**可观测性**：config resolve 时打印一行 `family/key/device → config(来源: 表/默认)`，
   这样"回落"是可见的。
5. 门禁：数值不变（同一 kernel 只改 launch schedule），单测 + 整模 energy/force/virial 全过。

### P2（1–2 周）结构性改造（热点 kernel 逐个过）

按 P0 的热点排序处理，每个 kernel 都要走"改 → 单测 → 整模对照 → 性能记录"闭环：

- **索引宽度**：`E×C` 是 1e8 量级，`E*C < 2^31` 时全部走 int32 offset（需加编译期/运行期守卫），
  这是 DCU 上很典型的一刀。
- **`tl.dot` 的精度路径**：先确认 Triton 在 gfx928 上对 `input_precision="ieee"` 是发 MFMA 还是
  退化成 FMA 循环（用 `hipprof` 看指令成分）。如果是退化，就要评估 3×TF32 或 fp16x3 拆分；
  **任何精度降级都必须先过 fp64 exactness sweep 再落到表里**，不是手改。
- **去冗余**：`tl.trans` 换布局、`tl.sum` 沿 edge 维的 cross-lane reduce 分块、`num_stages=2`
  （64 KB LDS 下 3 级放不下）。
- **小 kernel 合并**：把 gate/activation、radial mix、value path 之间能合的合成一个。

### P3（可选，与任务二合流）跨算子融合

- GIE（zonal scatter + radial basis + 归一化）→ 1 个 kernel；
- edge 几何 / 类型特征 / 环境矩阵 → 1 个 kernel；
- segment softmax + attention aggregate → 1 个 kernel；
- force assembly + virial → 1 个 kernel。

**收益预期与判断**：0.2 说明 elementwise/index 占 66%，所以 P3 的融合才是数量级收益的来源；
P1/P2 是"在现有结构里榨 1.2–1.8x"。如果 P3 的目标算子用 Triton 写不到目标性能，就交给
任务二的 DMK HIP 实现——这两个任务在这里天然衔接。

### 任务一验收指标

| 指标 | 门槛 |
| --- | --- |
| 单次 eval（3000 原子，`--no-profile` 中位数） | 相对 P0 基线 ≥ 1.5x（P1+P2 后），≥ 2.5x（含 P3） |
| Triton kernel 数 | 每次 eval 的 launch 数下降 ≥ 30%（P2+P3） |
| 单 kernel | 达到该 shape 的 memory roofline ≥ 60%（访存型）/ ≥ 50% rocBLAS（GEMM 型） |
| 数值 | energy/force/virial 对齐 baseline；每个 op 过 reference 单测；精度降级的路径必须额外过 fp64 sweep |
| 回归 | 不允许 NVIDIA 路径性能回退（配置表按设备名隔离） |

### 任务一风险

- **R1** Triton 在 gfx928 上的可用性未知（能否出 kernel、是否发 MFMA、LDS 溢出如何表现）。
  → P0 第 1 天就冒烟，失败则先解决工具链或用 DMK 接手热点算子。
- **R2** 设备名未收录导致的回落是静默的 → 必须有可见的 resolve 日志与断言。
- **R3** fp16x3 / TF32 改数值，力对能量误差敏感 → 双层门禁（op 级 fp64 sweep + 整模对照）。
- **R4** `pt` 与 `pt_expt` 两条路径都有 triton 分派，但只有 `pt_expt` 支持本项目的原生 patch。
  建议基线统一到 `pt_expt`（用 `dp convert-backend` 生成 `pt_expt` 的 `.pt`），否则任务二接入时
  还要再迁一次。

---

## 2. 任务二：面向 DCU 的 megakernel 运行框架

**当前最大问题**：GEMM 47.69 GFLOP/s，为 rocBLAS 的 0.35%（288x 差距）。先把这一条堵住，
框架才谈得上"运行"，否则 megakernel 只是把慢算子搬进一个更大的慢 kernel。

### 差距的根因清单（按贡献排序，M0 会逐条量化）

1. **算术强度太低**：16×16 输出 tile + k 步长 4，一次 MMA（1024 MAC）要配 128 次标量访存。
2. **同步粒度太粗**：每个 12-k 批次一次 `__syncthreads()`，k=2048 时单块 171 次屏障去保护 512 次 MMA，
   延迟完全没被隐藏。
3. **地址计算走 64-bit 除法/取模**：`tile_indices()` 每元素 `index / cols`、`index % cols`，AMD 上是
   软件 div_u64，量级几百 cycle；每 k-tile 每 lane 要算十几次。
4. **访存未向量化**：A/B/C 全是 4B 标量访存；A 的 16×4 小段只占 25% 的 cacheline。
5. **LDS 无 swizzle**：`a_tile[16*4]` 行距 16 B，`du_load_matrix_sync` 的 bank 冲突不可避免；
   `ShmemSwizzleLayout` 写了但取片路径只接受 row-major。
6. **MMA 用了 k4**：gfx928 支持 `m16n16k8`，指令数可以减半；独立算子已能用 k8，page 版还没接。
7. **无 L2 复用调度**：block 按行优先走，A/B 的 HBM 流量是理论值的 ~4 倍。
8. **epilogue 标量 + 边界判断**：每 lane 4 次 scalar store，带 div/mod 与 predicate。

**目标值**：2048³ 达 rocBLAS 的 ≥60%（≥8.2 TFLOP/s，即 ≤2.1 ms）；DPA4 真实形状 ≥55%。

### M0（2–3 天）工程化 + 度量标尺

1. 让脚手架可编译可运行：`megakernel.cpp` 用 `uint32_t`、修 `WARP_SIZE`、实现 4 角色空跑 +
   LDS 邮箱 + 计数，`launch_mk()` 真启动；新增 `tests/csrc/run_dmk_megakernel_test.py`
   （复用 GEMM runner 的 hipcc 直编 + arch 检测）。
2. `embedding.h` 已按 `BaseOp` 补成页面版 `EmbeddingOp`；`rmsnorm.h` / `attention.h`
   两个不能编译的旧桩（未纳入 op pack、也无人 include）已删除，需要时按 `BaseOp` 重写。
3. **三个 μbench（最重要）**，它们是后续所有优化的判决器：
   - 纯 MMA 循环（同 tile 形状、无访存）→ DUMMA 的实际上限 FLOP/s；
   - HBM→LDS 16 B 向量拷贝 → 有效带宽 + bank 冲突计数；
   - `ld_matrix` / `store_matrix` 单独计时（不同 ld / 不同 LDS 布局）→ 定位第 5 条。
4. 顺带把 0.1–0.8 的根因逐条量出来（例如把 `tile_indices` 换成编译期常量再跑一次），
   形成"谁贡献了多少"的清单，作为 M1 的设计输入。

> 状态：M0 的第 1 条已部分落地——按 HazyResearch Megakernels 的结构实现了
> `csrc/dmk/config.h`（静态 LDS + page 预算）、`lds.h`（mbarrier parity 替代）、
> `state.h`（state/pid/page semaphore）、`op.h`（虚基类 + `make_op` 工厂）、
> `controller.h`（取指 / page 置换 / 动态 semaphore）、`noop.h` /
> `ops/elementwise.h` / `ops/gemm.h` / `ops/embedding.h`、`dmk/megakernel.h` 与
> `tests/csrc/run_dmk_lds_test.py`；`copy.h` 只保留 `load_vec`/`store_vec`/
> `aligned` 三个向量原语，16 B 访存的循环写在 Elementwise/Gemm/Embedding 各自的
> `load()/store()` 里；按 DCU 同步 DUMMA 的实际情况去掉了
> launcher 波（controller + loader + storer + 4 consumer waves，448 threads），
> 设计说明见 `docs/lds-management.md`。
> 三个 μbench 与真实 op 的 page 规划仍待做；`rmsnorm.h` / `attention.h` 两个不能编译的
> 旧桩已删除（见第 2 条），需要时按 `BaseOp` 重写。

### M1（1–1.5 周）GEMM 打到 roofline

设计（按 64 KB LDS / wave64 / 无 TMA 反推）：

- **层次**：CTA = 4 waves（256 线程）；macro tile 128×128（起步可以 128×64）；每个 wave 承担
  32×64 或 64×32，即 2×4 / 4×2 个 16×16 累加器；k tile 16 或 32。
- **MMA**：`GemmOp<T, K>` 已支持 `m16n16k4/k8` 和 half/bf16 的 k16，下一步是让 page 版 `GemmOp` 复用它。
- **LDS 双缓冲 + swizzle**：取片路径现在只接受 row-major，需要让 `GemmWave` 支持
  `ShmemSwizzleLayout`；HBM→LDS 的搬运已经按行主序向量化（`GemmOp::copy_tile`）。
- **访存**：A/B 用 16 B 向量化（`float4`），布局以"相邻 lane 访相邻地址"为准则；删掉 inner loop
  里的 64-bit div/mod（tile 索引改成编译期常量或位移）。
- **流水**：DCU 没有 TMA，所以是"loader 波预取到寄存器 → 写 LDS → barrier → consumer 读"
  的软件双缓冲，深度 2（64 KB 限制）。
- **epilogue**：fragment → LDS → 16 B store；支持 C 累加；边界用整块 fast path + 尾块慢路径。
- **L2 复用**：block 按 8~16 列一组的 snake 顺序走，把 A/B 的 HBM 流量压到理论值附近。
- **验收**：`test_dmk_lds_megakernel` 里承载 `GemmOp` 的用例全 PASS；2048³ ≥ 8 TFLOP/s（当前标量实现达不到，见下）；
  DPA4 形状 ≥55%；把 GFLOP/s 写成断言，防止回退。
- **顺带**：DMK GEMM 的对照基准现在只有 `tests/csrc/run_dmk_gemm_test.py`（kernel-only vs rocBLAS）；原 `tests/dcu/test_dcu_gemm_compare.py` 依赖的 `kernels.gemm` 已删除，需要时在新入口上重建三方对比。

### M2（1 周）Op 抽象与调度器

- **虚函数边界**：当前 `BaseOp` 的虚调用只在每条指令每个 role 发生一次，op 内部数据循环
  仍然内联，因此先保留多态接口；如果 profile 显示 indirect call / vtable 访存有影响，
  再在保持 `BaseOp` 接口的前提下加 compile-time `OpId + 模板分发` 快路径。
- **op 描述表**：类型 / 参数指针 / 输入输出 buffer 索引 / 依赖位图 / shard 维度 / epilogue 变体。
- **buffer 与 arena**：单块 HBM arena + 静态 lifetime 规划（现在峰值分配 30.7 GB，每次 eval 都在
  申请释放；megakernel 必须零分配）。
- **角色与信号量**：CTA 内 4 角色 + LDS 邮箱（`ds_add` + spin，深度 2）；
  CTA 间先用 `hipLaunchCooperativeKernel` + grid.sync 验证（DTK 26.04 手册有），
  不可用则退化为"grid ≤ 常驻块数"的原子 sense-reversing barrier。
- **验收**：`gemm → rmsnorm → gemm` 三条 op 在一个 kernel 内完成，数值与逐 op 一致，launch 数 = 1。

### M3（1.5–2 周）把 DPA4 的一层做成 megakernel

- 第一块选 **GIE**（边几何 + 径向基 + scatter 归一化）或 attention-free 的 SO2 卷积 envelope reduction：
  这两块边界清晰、现有 HIP 实现已经是正确性基准。
- 控制器跑"指令流"：host 按模型结构生成，参考 `Megakernels` 的 instruction pipeline，但 DCU 上
  用 LDS + `ds_add` 实现信号量、深度 2（没有 mbarrier/TMA）。
- **保底路径**：先做"单 CTA 内、按 atom/edge tile 分区"的融合（scatter 用原子加），不要求跨 CTA
  融合。这一版就能拿到大部分收益，且不依赖 grid 同步能力。
- **验收**：目标层 launch 数从几十降到 1–2；该层设备时间相对 Triton 最优版本再降 ≥30%；
  整模数值对齐。

### M4（1–2 周）整模型 + 接入 PyTorch

- descriptor 的每层都走同一个 megakernel（先"每层一次 launch"，再往"整个 descriptor 一次 launch"推）。
- 构建与入口：`setup.py` 编 `kernels._dmk`；新增 `csrc/dmk_bindings.cpp` 暴露
  `dmk_forward(coords, atom_types, cell, params_ptr)`；`python/model/dpa4.py` 加
  新增一个 `fused=("megakernel",)` seam（`native_ops` 已随 pt_expt-only patcher 一起删除，
现有接缝是 `segment_sum` / `segment_softmax`）。
- 验证矩阵：3000 / 12000 / 36000 原子 ×（数值 / 性能）；用 `rocprof` 确认 launch 数与 kernel 时间。

### 任务二验收指标

| 指标 | 门槛 |
| --- | --- |
| GEMM 2048³ | ≥ 60% rocBLAS（≥8.2 TFLOP/s，≤2.1 ms） |
| GEMM @ DPA4 形状 | ≥ 55% rocBLAS |
| 每次 eval 的 launch 数 | 2614 → ≤ 200（M3）→ ≤ 20（M4 目标） |
| 整模设备时间 | 相对"Triton 最优版本"≥ 1.5x（先），≥ 2x（目标） |
| 数值 | force 的 max_abs / 相对误差按 1e-5 量级门禁（MMA 累加顺序与 FMA 不同，力比能量敏感） |
| 稳定性 | 不允许 > 常驻块数的 grid（否则 barrier 死锁），启动时断言 |

### 任务二风险

- **R5** DCU 没有 `setmaxnreg` → 角色特化无法给 loader/consumer 分配不同寄存器预算。
  对策：consumer-heavy 布局（4 consumer 波 + 1 loader 波），或干脆让 consumer 自己预取、去掉 loader 波。
- **R6** grid 同步能力待实测。不可用就锁定"每 CU 一个 CTA + 层内不跨 CTA"的分区设计（M3 保底路径）。
- **R7** LDS 只有 64 KB，macro tile 与流水深度都被卡住；任何 tile 参数都要先过 "LDS 预算" 断言。
- **R8** 数值：fp32 MMA 的累加顺序与 FMA 不同，力的误差会被放大，需要按 op 与整模两级门禁。

---

## 3. 排期与并行关系

```
周 1      P0（Triton 基线, 1-2d）        M0（脚手架 + μbench, 2-3d）
周 2-3    P1（DCU tile config）           M1（GEMM 到 roofline, 1-1.5w）
周 4-5    P2（结构性改造）                M2（Op 抽象 + 调度器）
周 6-8    P3（融合，可与 M3 合流）        M3（DPA4 单层 megakernel）
周 9-10   —                               M4（整模型 + 接入）
```

两条线共用一个前提：**必须能上 DCU 机器**（我本地是 macOS，无法跑 HIP/DCU 验证）。
建议 P0 与 M0 都在第一周落地，因为它们分别给出两条线后面所有决策所依赖的两个数：
"Triton 在这个 DCU 上到底能不能跑、跑到什么水平" 和 "360 ms 里各因素各占多少"。

## 4. 需要确认的三个决策点

1. **精度红线**：任务一是否允许 fp16x3 / 3×TF32 这类"有界数值扰动换速度"的路径？
   允许 → 可以开 `DP_TRITON_INFER=3`；不允许 → 只做 P1/P2，收益上限大约 1.5x。
2. **最终形态**：任务二的目标是"关键 op 的加速库（GEMM/RMSNorm/GIE 等）"，还是"整模型单 kernel"？
   这决定 M3/M4 投不投（后者风险显著更高，收益也更大）。
3. **基线后端**：是否统一到 `pt_expt`（需要 `dp convert-backend` 生成对应 `.pt`）？
   现在 profile 跑的是 legacy `pt`，而本项目原生 patch 只支持 `pt_expt`，两条线迟早要统一。
