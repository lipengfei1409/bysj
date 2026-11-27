# 用现成工具 profile 持久化 kernel

不写自定义计时，用 `rocprofv3` / `torch.profiler` / `nsys` 就够。关键是先认清一件事：

> **持久化 GEMM 只有一次 kernel launch（`gemm_mk`），而 rocBLAS 是一串 kernel。**
> 所以核函数级的对比是"我们的 1 个 kernel"对"厂商的 N 个 kernel"，总设备时间才是可比量。

## 0. 先准备一个可 trace 的工作负载

trace 不要把整个测试套件包进来（探针本身会污染时间）。用一个最小脚本：

```bash
cat > /tmp/bench.py <<'PY'
import os, sys
from dmk import dispatch, emitters, mk
m = n = k = int(os.environ.get("SHAPE", "2048"))
cus = int(os.environ.get("CUS", "32"))
g = mk.GemmGlobals.from_shapes(m, n, k)
g.attach(emitters.GemmEmitter(m, n, k).queues(cus))
for _ in range(5):                       # 预热
    dispatch.launch_gemm(g.instructions, g.a, g.b, g.c)
import torch; torch.cuda.synchronize()
for _ in range(20):                      # 被测部分
    dispatch.launch_gemm(g.instructions, g.a, g.b, g.c)
torch.cuda.synchronize()
PY
PYTHONPATH=python python3 /tmp/bench.py            # 先确认能跑
```

## 1. `rocprofv3`：kernel 时长 + 资源占用

```bash
PYTHONPATH=python rocprofv3 --kernel-trace -f csv -o /tmp/gemm_k \
  -- env SHAPE=2048 CUS=32 python /tmp/bench.py
```

CSV 里每行一个 launch，重点列：

| 列 | 看什么 |
| --- | --- |
| `Duration` | 单次 `gemm_mk` 的时长；× 20 次就是总时间 |
| `Grid_Size` | 应等于 `CUS`（一个 block 一个 CU） |
| `Workgroup_Size` | 应为 512（8 waves × 64） |
| `VGPR_Count` / `LDS_Size` | LDS 约 49 KiB（6 页）→ 一个 CU 只能放 **1 个 block**；VGPR 决定是否还有余量 |

> legacy 版（`rocprof`）：`rocprof --stats python /tmp/bench.py`，输出 `results.stats.csv`。

## 2. PMC：决定"下一步改哪"的唯一依据

```bash
rocprofv3 --list-pmc | head -50          # 先确认本 DTK 支持哪些计数器名

PYTHONPATH=python rocprofv3 --pmc \
  SQ_INSTS_VALU,SQ_INSTS_LDS,SQ_LDS_BANK_CONFLICT,SQ_WAVES,\
SQ_INSTS_VMEM,TCP_TCC_REQ,GRBM_GUI_ACTIVE \
  -f csv -o /tmp/gemm_pmc -- env SHAPE=2048 CUS=32 python /tmp/bench.py
```

判读（表里的"动作"对应 `docs/dmk-host.md` 里讨论的 P1–P7）：

| 观测 | 结论 | 动作 |
| --- | --- | --- |
| `GRBM_GUI_ACTIVE` 接近 1、`SQ_INSTS_VALU` 高，但 TFLOP/s 只有 2–5 | 标量 FFMA 到顶（`fma_probe` 实测 18.1 TFLOP/s 是理论上限） | **P5：DUMMA 进 `GemmOp`** |
| `SQ_LDS_BANK_CONFLICT` 与 `SQ_INSTS_LDS` 同量级 | LDS stride=16 的列访问冲突 | **P7：padding 到 17** |
| `SQ_INSTS_LDS` 远大于 `SQ_INSTS_VALU` | 每条 FMA 两次 LDS（现状就是这样） | **P2：累加器进寄存器 + P3 寄存器分块** |
| `SQ_WAVES` 少于 `kConsumerWaves` | consumer wave 没在算 | 查指令覆盖与 `tile_column` 裁剪 |
| `TCP_TCC_REQ` 高、`TCC_HIT` 低 | 访存放大（A 被重读 N/16 次） | **P6：宏 tile / tile 顺序** |

## 3. `torch.profiler`：主机 vs 设备

单看 kernel 不够——`emitters.gemm_queues(4096³)` 要建 6.5 万条指令，主机侧可能比 kernel 还慢：

```bash
PYTHONPATH=python python3 - <<'PY'
import torch, os
from torch.profiler import profile, ProfilerActivity
from dmk import dispatch, emitters, mk
m = n = k = 4096
g = mk.GemmGlobals.from_shapes(m, n, k); 
t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
import time; start = time.perf_counter()
g.attach(emitters.GemmEmitter(m, n, k).queues(32))
print("host build:", (time.perf_counter() - start) * 1e3, "ms")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3): dispatch.launch_gemm(g.instructions, g.a, g.b, g.c)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
PY
```

两个数字对比：**host build** 与 **kernel 时长**。若两者同量级，说明调度构建是瓶颈，与 kernel 优化无关。

## 4. `nsys`（若已安装）

```bash
nsys profile --trace=cuda,nvtx,osrt -o /tmp/gemm_ns -- python /tmp/bench.py
nsys stats --report cuda_gpu_kern_sum /tmp/gemm_ns.nsys-rep
```

用途是把"一次 launch"放进时间线看：launch 前的空档（主机准备）与 kernel 本身的比例。

## 5. rocBLAS 对照

同形状跑厂商 baseline，得到比值：

```bash
DPA4_RUN_PERSISTENT_GEMM_LATENCY=1 DPA4_PERSISTENT_GEMM_JSON=/tmp/gemm.json \
PYTHONPATH=python python3 -m unittest tests.csrc.latency.test_persistent_gemm_latency -v
```

它内部用 `tests/dcu/rocblas_cpp.py`（直接调 rocBLAS 的 C++ 包装），输出 `persistent_ms` /
`rocblas_ms` / `ratio`。**profile 给了"差在哪"，这个给了"差多少"**，两个都要。

## 如果本卡是 gfx906

矩阵单元不存在（`du_mma.h` 只认 gfx926/928/936/938），`setup.py` 会自动退回标量 GEMM。这时 rocBLAS
的 SGEMM 在 gfx906 上也是 VALU/FFMA 实现，所以 `ratio` 比的是**两套标量调度的差距**，不是"标量 vs
tensor core"——判断"要不要上矩阵单元"要换到支持 DUMMA 的卡上测。

## 一条经验

先跑 §1 与 §2 各 **一次**（约两分钟），再决定改哪一项。不要在没有计数器的前提下直接写 P1/P2——那正是上一轮我差点做的：三个 consumer wave 空转是**读代码看出来的猜测**，而 `SQ_WAVES` 会直接告诉你是 4 还是 8。
