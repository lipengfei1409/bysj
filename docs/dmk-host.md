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
  →      mk_harness          （唯一实例化：mk_internal<MkGlobals, NoOp, GemmOp>）
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
  → csrc/dmk/mk_harness.cpp    __global__ mk_harness
  → mk_internal<GemmGlobals, NoOp, GemmOp>  → ops/gemm.h
```

M/N/K 不是 16 的倍数会在**三处**被拒（`emitters.gemm_instructions`、
`GemmGlobals.from_shapes`、C++ launcher），因为 `GemmOp` 没有边界路径。

## 测试

```bash
PYTHONPATH=python python3 -m unittest tests.python.test_dmk_host tests.python.test_dmk_schedule -v
```

覆盖：份额和等于总数、余数分配、tile 随份额收缩、每 CU 的指令数 = ceil(份额/tile)、
一个 tile 一条指令且覆盖恰好一次、形状非 16 倍数被拒、两个编码器逐字一致、字宽与已知 opcode、
float 位转换、代价排序、NoOp 补齐、链序保持、有环报错、round-robin 发牌、模拟器重放与稠密乘一致、
删掉一条指令后模拟器必须报错、未知 opcode 被拒、缺扩展时 `dispatch` 明确报"未构建"。
