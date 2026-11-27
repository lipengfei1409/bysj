# 主机侧持久化调度：`python/dmk/`

目录结构照参考实现 [`Megakernels/megakernels/`](../../Megakernels/megakernels/) 组织：

```
python/dmk/
  model_types.py   值类型：Work（各算子的工作量）、CuTileConfig / CuPlan（每 CU 的 tile）
  utils.py         CU 数量、工作量切分（split_units / plan_cus / instruction_counts）
  instructions.py  指令对象：serialize() → 32 字，cost() → 调度代价
  scheduler.py     DAG + 分配策略：assign_dag_to_cus（堆）/ round_robin_assign
  emitters.py    模型 → 指令序列：gemm_instructions / gemm_queues
  dispatch.py      交给运行时：kernels._C.dmk_gemm
  mk.py            Globals + launch（emulate / launch / verify / time）
  python_vm.py     指令流的 Python 模拟器（无 DCU 也能验调度）
  demos/gemm.py    模型 demo（参考实现的 llama.py 位置）
  scripts/run_gemm.py  命令行入口
```

区别只在命名与代价来源：参考实现的 "SM" 在这里叫 **CU**（DCU 的一个 workgroup），`cost()` 是
每指令的 FLOP 估计而不是实测核时间。

## 一条命令三种模式

```bash
# 1) 本机：只构建并打印调度
PYTHONPATH=python python3 -m dmk.scripts.run_gemm 64 64 32 --cus 3 --dry-run
#   3 CUs, at most 6 instructions each
#     cu 0: gemm=6
#     cu 1: gemm=5
#     cu 2: gemm=5

# 2) 本机：用 Python VM 执行调度并和稠密乘对拍（不需要 DCU、不需要编译）
PYTHONPATH=python python3 -m dmk.scripts.run_gemm 64 64 32 --cus 3 --emulate
#   emulated: max |C - A@B| = 3.815e-06

# 3) DCU：真跑持久化 kernel
python -m dmk.scripts.run_gemm 256 256 256 --cus 4
#   persistent GEMM: max |C - A@B| = ..., ... ms/iter
```

## 新增算子：写一个 Emitter 子类

`emitters.py` 里每个算子是一个 `OperatorEmitter` 子类，分片（shard）逻辑在基类里，子类只写
"怎么把一段范围变成指令"：

```python
class OperatorEmitter:
    kind: str                                  # gemm / embedding / attention / elementwise
    def units(self) -> int                     # 原生单位（元素 / tile / 行 / query 行）
    def slots(self) -> int                     # 可寻址槽位，默认等于 units
    def work(self) -> Work                     # 交给 utils.plan_cus 的工作量
    def emit(self, tiles, start, cover)        # ← 子类唯一必须实现的方法
    def slots_per_instruction(self, tiles)     # 默认 1，按 tile 覆盖多个槽位时重写
    def instructions(self) -> list[Instruction]  # 整段给一个 CU 时的指令（基类派生）
    def queues(self, cus) -> list[list[Instruction]]  # 分片，基类派生
```

`queues()` 是所有算子**共用**的一条路径：工作量 → `utils.plan_cus`（按份额定每 CU 的 tile）→
`utils.instruction_counts`（ceil(份额/tile)）→ 每个 CU 调自己的 `emit` 覆盖自己的份额。所以新增算子
不会碰分片、补 NoOp、CU 边界这些容易写错的地方。

一个最小实现（Embedding）：

```python
class EmbeddingEmitter(OperatorEmitter):
    kind = EMBEDDING
    def __init__(self, rows, *, elem_per_row):
        super().__init__(rows); self.elem_per_row = elem_per_row
    def work(self) -> Work:
        return Work(embedding_rows=self.units(), embedding_tile=1)  # 一条指令一行
    def emit(self, tiles, start, cover) -> Embedding:
        del tiles, cover
        return Embedding(src_row=start, dst_row=start, count=self.elem_per_row)
```

模型 = 一组 Emitter：`ModelSchedule([...]).rows(cus)` 按模型顺序把每个算子的每 CU 行拼起来
（一个 CU 从前往后走自己那一行，所以顺序必须是"算子 1 的份额 → 算子 2 的份额"）。

**加一个算子要动的四处**：

| 位置 | 改什么 | 忘了会被谁抓 |
| --- | --- | --- |
| `csrc/ops/<op>.h` | `BaseOp` 子类（页、信号量、指令字） | 编译 |
| `csrc/ops/config.h` | `kOp<X>` 加进 `enum Opcode`（不改已用值） | `test_opcodes_are_declared_once_in_config` |
| `csrc/dmk/*_launch.cpp` | op pack 加类型、Globals 加张量 | 编译（字段名要对得上） |
| `python/dmk/emitters.py` | 一个 `OperatorEmitter` 子类 + 注册进 `EMITTERS` | `test_each_kind_has_an_emitter_and_a_registry_entry` |

`instructions.py` 里通常已经有对应的 dataclass；没有就加一个（**字段顺序 = 线格式**，会被双编码器
逐字比对的测试抓到）。

## 指令编码与线格式

`instructions.py` 里每个算子是 dataclass，`serialize()` 按**字段声明顺序**填字，所以字段顺序就是
线格式：

| 算子 | word 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| `Elementwise` | page | scale | bias | count |
| `Gemm` | m0 | n0 | k_tiles | |
| `Embedding` | src_row | dst_row | count | |
| `Attention` | query_tile | num_keys | dim | |

opcode 是 C++ 侧的表（`ops::config`），镜像在 `kernels.mk_stream`。有两个编码器（dataclass 与
`mk_stream`），一条"逐字比对"的测试把两边钉在一起——我第一版把 `Elementwise` 字段写成
`(tile, count, scale, bias)`，就是被它当场抓出来的。

## Python VM 为什么值得有

`python_vm.emulate_gemm` 只读指令字、在 CPU 上重放：按 `(m0, n0, k_tiles)` 累加 16×16 tile。
它能验的是**调度**而不是 kernel——某个 tile 没被任何 CU 覆盖、被覆盖两次、编码错位，都会在这里
露出来，且不需要卡。它验不了的是页、信号量、DUMMA 这些只存在于设备的语义，所以
`tests/csrc/accuracy/test_persistent_gemm.py` 仍然保留。

## 只有一处 launch 实现

`mk_internal` 是模板，**没被实例化就发不出去**，所以"每个 op pack 一个具体 `__global__`"不可避免。
除此之外的部分全部共享：

```python
Python  dmk.dispatch.launch_gemm(...)
  → C++  dmk_gemm_hip        （每 pack 一个：校验形状、把 torch 张量装进 MkGlobals）
  →      dmk::mk::launch_kernel   （唯一实现：预留动态 LDS + 每 instruction 行一个 block + 当前流）
  →      mk_all              （唯一实例化：mk_internal<MkGlobals, NoOp, Elementwise, Gemm, Embedding>）
  →      mk_internal         （运行时：controller 发布、三波执行）
```

`launch.h` 还服务第二个 pack（`megakernel.cpp` 的 all-ops demo，供
`tests/csrc/test_dmk_lds_megakernel.cpp` 验证页与流水），所以 LDS 预留、grid、stream 三件事
在整棵树里只写了一遍。

## 持久化 GEMM 的完整链路

```
demos/gemm.py            GemmDemo(m, n, k, cus)
  → emitters.gemm_queues      一个 16×16 tile 一条指令，按 CU 均分
  → instructions.InstructionStream   补 NoOp 成矩形 → (cus, length, 32) int32
  → mk.GemmGlobals.attach       指令张量 + A/B/C
  → dispatch.launch_gemm        kernels._C.dmk_gemm
  → csrc/dmk/megakernel.cpp     __global__ mk_all
  → mk_internal<GemmGlobals, NoOp, GemmOp>  → ops/gemm.h
```

M/N/K 不是 16 的倍数会在**三处**被拒（`emitters.gemm_instructions`、
`GemmGlobals.from_shapes`、C++ launcher），因为 `GemmOp` 没有边界路径。

## 这套设计适合 DCU K100 吗

分场景，不能一句话答。当前状态先摆清楚：**只有 16³ 的数值正确性（`max err = 0.0`），
没有任何性能数字**。所以下面是结构性判断，不是实测结论。

### K100 的三条硬约束

| 约束 | 对设计的含义 |
| --- | --- |
| LDS 64 KiB/CU，6 × 8 KiB 页 | 一个 block 就用掉大半 → **一个 CU 只能放一个 block** |
| 一个 block 意味着没有 block 级延迟隐藏 | 只能靠波内并行/流水自己藏延迟 |
| DUMMA 形状 `16×16×4`（fp32）/ `16×16×8` | 一个 wave 出一条 16×16 的 tile，k 方向一次只走 4 |

### 现在的实现在 K100 上的四个短板（都能指着代码说）

| 短板 | 位置 | 后果 |
| --- | --- | --- |
| **stage 与 compute 串行** | `run()` 里 `stage_slices → barrier → 4×mma → barrier` | 每个 k-tile 吃一次 HBM 延迟（数百 ns 量级），而 4 个 mma 只有几十 ns → 利用率可能 <10% |
| **无 L2 分块 / 无 operand 复用** | 一个 tile 把它的 A 切片读满 K | 总 A 流量 ≈ `M·K·N/16`，是理论最小值的 `N/16` 倍 |
| **一个 wave 时没有并行** | `kConsumerWaves = 1` | 单个 wave 只能顺序执行，页的 ping-pong 也只剩 3 页可用 |
| **指令粒度太细** | 1 instruction = 1 个 16×16 tile | 2048³ 就是 16384 条指令，每条都要走 controller 发布 + 页周转 |

换句话说：**在大稠密 GEMM 这条赛道上，它现在打不过一个带双缓冲和寄存器分块的普通 kernel**，
更不用说 rocBLAS。短板不在"persistent"这个词，而在我们还没做双缓冲和宏 tile。

### 它真正适合的场景

persistent megakernel 的价值来源是**省掉大量小 kernel 的发射与调度开销**，而不是把一个大
GEMM 算得更快。这个项目的目标负载恰好是前者：

- DPA4 一步推理里有几百到上千个 per-edge 小算子（conv、gie、segment_*），
  单个 kernel 只有几十微秒甚至更短；
- 这些算子的参数（边数、度数、通道）在推理期固定 → 指令流可以**预先编好**，正是 megakernel 的形态；
- 稠密 GEMM 在这种模型里是小尺寸（16×16 出 tile、K = 64–256），**本来就不该拿去和 GEMM 库比**。

所以判断是：

| 用途 | 结论 |
| --- | --- |
| 替代 rocBLAS 跑大稠密 GEMM | ❌ 现在不行；补齐双缓冲 + 宏 tile 后也只是"能用"，不会是强项 |
| **把模型里的小算子链融合成一个 kernel** | ✅ 这才是它的设计目标，K100 的 64 KiB LDS 和 wave64 都合适 |
| 单条 GEMM 的 tile 级并行 | ⚠️ 只有在 K 很小（64–256）时成立，否则被访存拖死 |

### 决定性问题与下一步

先让数字说话，再谈适配：

1. 跑 w1 的延迟扫描，与 rocBLAS 比（`tests/csrc/latency`）；
2. 若大形状 `ratio < 0.1` → 说明串行 stage 是主因，先做**双缓冲**（页够：A/B 各 2 页 ping-pong，
   当前只用了 3 页）；
3. 若双缓冲后仍 < 0.2 → 做**宏 tile**（一条 instruction 覆盖 32×32 或 64×64，摊薄页与信号量开销）；
4. 若都不理想，就把优化目标换成"融合小算子链"，用模型真实的算子序列来评估——那才是这套框架的主场。

## 测试

```bash
PYTHONPATH=python python3 -m unittest tests.python.test_dmk_host tests.python.test_dmk_schedule -v
```

覆盖：份额和等于总数、余数分配、tile 随份额收缩、每 CU 的指令数 = ceil(份额/tile)、
一个 tile 一条指令且覆盖恰好一次、形状非 16 倍数被拒、两个编码器逐字一致、字宽与已知 opcode、
float 位转换、代价排序、NoOp 补齐、链序保持、有环报错、round-robin 发牌、模拟器重放与稠密乘一致、
删掉一条指令后模拟器必须报错、未知 opcode 被拒、缺扩展时 `dispatch` 明确报"未构建"。
