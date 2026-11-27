# 主机侧：Per-CU 计算图构建（`python/dmk/`）

持久化调度的主机侧分三步，各自一个文件：

```
python/dmk/
  instructions.py  指令对象（每个算子一个 dataclass）：serialize() → 32 字，cost() → 调度代价
  scheduler.py     把指令分给各 CU：DAG 堆分配（按代价、尊重依赖）或 round-robin
  config.py        每个 CU 的算子配置：各类 tile size，以及把总工作量切成每 CU 的份额
  shard.py         把计算图（DataFlow）按 CU 切开，用 instructions 编码成队列
  mk.py            模型 kernel：Globals + launch（对应参考实现的 mk.py）
  main.py          入口：build_flow → shard → launch
```

结构对齐参考实现 `Megakernels/megakernels/`（`instructions.py` / `scheduler.py` /
`mk.py`），区别只在于"SM"在这里叫 CU（一个 DCU workgroup），代价模型是每指令的 FLOP 估计而不是
实测核时间。

## 为什么按"可寻址槽位"计数

每个算子被寻址的单位不同，配置与切分都以它为准：

| 算子 | 一个槽位 | 一条指令覆盖 |
| --- | --- | --- |
| elementwise | 一页（2048 float = 8 KiB） | **正好一页**（指令里带页索引，所以这个 tile 不能按 CU 变） |
| gemm | 一个 16×16 输出 tile | 一个 tile（走完整个 K） |
| embedding | 一行表 | `config.Work.embedding_tile` 行 |
| attention | 一行 query | `config.Work.attention_tile` 行 |

这样"每 CU 的 tile 不一样"就有了实际含义：份额小于默认 tile 的 CU 用**更小的 tile**，只发一条
指令，而不是发一条大半空闲的指令。elementwise 例外——它的地址粒度就是页，所以只能靠**最后一条
指令的 count** 收尾。

## instructions：指令就是对象

每个算子一个 dataclass，`serialize()` 按**字段声明顺序**填字，所以字段顺序就是线格式：

```python
Elementwise(tile, scale, bias, count)   # word 1=page, 2=scale, 3=bias, 4=count
Gemm(m0, n0, k_tiles)                   # word 1=m0, 2=n0, 3=k_tiles
```

`cost()` 给调度器排序（GEMM 是按 FLOP 估的、逐元素按元素数估）。这条容易写错——我第一版把
`Elementwise` 的字段写成 `(tile, count, scale, bias)`，与 `kernels.mk_stream` 的线格式
（1=page, 2=scale, 3=bias, 4=count）不一致，一条"两个编码器逐字比对"的测试立刻失败。

## scheduler：把指令分给 CU

```python
nodes, end = scheduler.linear_chain(instructions)   # 一条扁平流的 DAG
queues     = scheduler.assign_dag_to_cus(nodes, cus)  # 依赖感知的堆分配
queues     = scheduler.round_robin_assign(instructions, cus)  # 或平均发牌
```

堆分配按"哪个 CU 先空出来"派活（`heapq` 按结束时间），就绪指令优先放重的，避免长 GEMM 排在
一串短指令后面。有环会直接报错而不是悄悄少排。

## config：份额与 tile

```python
work = config.Work(elementwise=181_838 * 640, gemm_tiles=181_838,
                   embedding_rows=3000, attention_queries=3000)
plans = config.plan_cus(work, cus=4)      # 每个 CU 一份 TilePlan
counts = config.instruction_counts(work, plans)
```

`config.split(total, parts)` 把总槽位切成份额（余数给最前面的 CU，和恒等于总数）。
`plan_cus` 再按份额定每 CU 的 tile：`min(默认 tile, 份额)`，份额为 0 则 tile 为 0、不发指令。

## shard：切开并编码

```python
flow   = dmk.main.build_flow()             # 一组 OpCall(kind, units, extra)
queues = dmk.shard.shard(flow, cus=4)      # Queues(stream, plans, counts, work)
arr    = queues.tensor()                   # (cus, length, 32) int32 —— 运行时吃的形状
```

每条指令带**绝对槽位起点**，所以一个 CU 不需要知道别的 CU 分到了什么。短的行由
`kernels.mk_stream` 用 NoOp 补齐成矩形，长度一致才能一次 launch。

## main：三步 + 一条诚实的失败

```bash
python -m dmk.main --cus 4 --dry-run            # 打印每 CU 的队列（本机可跑）
python -m dmk.main --cus 4 --dry-run --json     # 机器可读
python -m dmk.main --cus 4                      # 需要 dmk 扩展；当前会明确报"运行时未构建"
```

`launch()` 目前**故意失败**：`csrc/dmk/megakernel.cpp` 还没被任何扩展构建，它要 launch 的 op pack
也还需要模型的 `Globals`。与其假装跑过，不如返回 2 并说明缺什么——这样 CI 里 `--dry-run` 是真验证，
真跑是真空缺。

## 测试

```bash
PYTHONPATH=python python3 -m unittest tests.python.test_dmk_host -v
```

21 项，覆盖三层的契约：份额和等于总数、余数分配、槽位换算、tile 随份额收缩、空闲 CU 拿 0 tile 且
被 NoOp 补齐、每 CU 一行且长度一致、指令 opcode 都在 `mk_stream.KNOWN_OPCODES` 内、空计算图被拒绝
而不是悄悄产出空流、`--json` 可解析、`--cus 0` 是用法错误、缺少运行时返回 2 而不是假装成功。
