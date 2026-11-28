# dense GEMM 与 rocBLAS 的差距：数据与下一步

## 实测（gfx928，2026-11-28）

| 形状 | rocBLAS | dense | 比值 |
| --- | --- | --- | --- |
| 512³ | 0.032 ms / 8.51 TFLOP/s | 0.071 ms / 3.77 | 0.443× |
| 1024×1024×4096 | 0.408 / 21.03 | 1.097 / 7.83 | 0.372× |
| 2048³ | 0.613 / **28.00** | 1.446 / 11.88 | 0.424× |
| 2048×8192×2048 | 2.458 / 27.96 | 6.002 / 11.45 | 0.409× |
| 4096³ | 4.383 / **31.36** | 10.908 / 12.60 | 0.402× |

两条读数决定了全部结论：

1. **rocBLAS 能到 28–31 TFLOP/s** → 这张卡的 fp32 算力上限不是之前记的 13.7（那是 FFMA 探针值），
   矩阵单元的上限至少在 31 以上。dense 的 11.9–12.6 恰好落在 FFMA 峰值附近，说明**我们用矩阵单元的
   效率只相当于一个 FFMA kernel**。
2. **dense 的吞吐在 512³→4096³ 之间几乎不变（3.8→12.6，仅 3.3×，而 FLOP 增长 512×）** →
   不是 launch 受限；要么是带宽，要么是发射效率。

## 每 CU 的发射速率

以 4096³ 为例（120 CU、~1.7 GHz）：

| | 每 CU | MAC/cycle |
| --- | --- | --- |
| rocBLAS | 261 GFLOP/s | **77** |
| dense | 105 GFLOP/s | **31** |

`m16n16k8` = 2048 MAC/指令 → rocBLAS 每 **~27 周期**发一条 mma，我们每 **~66 周期**发一条。
差 2.4×，缺口不在指令本身，在**它前面的 LDS 取片与依赖链**：每条 mma 要 2 次 fragment 装载
（A 2 floats/lane + B 2 floats/lane），而每个 wave 只有 4 条独立累加器链，LDS 延迟基本暴露在关键路径上。

## 下一步（按性价比）

| # | 手段 | 针对的缺口 | 预期 | 代价 |
| --- | --- | --- | --- | --- |
| **1** | **fragment 双缓冲**：把 k+1 的 A/B fragment 先装进备用寄存器，再发 k 的 mma | LDS 延迟（~25–50 周期/次）被 mma 盖住 | 1.2–1.5× | +10~12 VGPR/lane |
| **2** | **per-wave tile 加大**：128×64（每 wave 32 行×64 列 = 8 个 fragment），独立 mma 链 4 → 8 | ILP 不足 | 与 1 叠加 1.5–2× | 累加器 32→64 VGPR；LDS 20 KiB（含 C 分块 8 KiB）→ 仍 2 block/CU |
| 3 | 3 级全局预取 | 若仍受 HBM 延迟限制 | 1.1× | 多一套 stage（+6~8 KiB） |
| 4 | LDS swizzle | 残余 bank 冲突 | 未知 | 需 `du_load_matrix_sync` 支持 layout，或自己拼 fragment |
| 5 | split-K / stream-K | 小形状 CU 喂不满（512³ 只有 64 个 block / 120 CU） | 小形状 1.5–2× | 需跨 block 归约 |

**先做 1，再做 2**：两者都针对"每 CU 每周期只发出 31 个 MAC"，而且都不动算法。

## 一个待确认的数

要区分"带宽受限"与"发射受限"，测一次可达带宽即可：

```bash
PYTHONPATH=python python3 - <<'PY'
import torch
n = 1 << 27                                  # 512 MiB of fp32
x = torch.empty(n, dtype=torch.float32, device="cuda")
y = torch.empty_like(x)
for _ in range(3): y.copy_(x)
torch.cuda.synchronize()
t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
t0.record()
for _ in range(10): y.copy_(x)
t1.record(); t1.synchronize()
ms = t0.elapsed_time(t1) / 10
print(f"copy {ms:.3f} ms  -> {2 * n * 4 / (ms * 1e6):.0f} GB/s (read+write)")
PY
```

dense 在 12.6 TFLOP/s、AI = 16 FLOP/byte 时需要的带宽是 **0.79 TB/s**。

| 测到的可达带宽 | 结论 |
| --- | --- |
| ≥ 1.5 TB/s | 我们只用了一半 → **发射受限**，做 1 和 2 |
| ≈ 0.8 TB/s | **带宽受限** → 必须提高 AI，直接做 2（128×64 把 AI 提到 21）甚至 128×128（AI 32） |
