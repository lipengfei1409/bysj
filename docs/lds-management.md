# DCU Megakernel 的 LDS / page 管理

> 范围：`myproject/csrc/dmk` 的 DCU megakernel。
> 设计直接对照 HazyResearch [Megakernels](https://github.com/HazyResearch/Megakernels)
> 的 `default_config` / `state<config>` / controller / page allocator / NoOp 结构，
> 只把 NVIDIA 专有的 mbarrier、TMA、`setmaxnreg` 换成 DCU 可用的实现。
>
> 这里的 LDS 管理不是"每个算子自己切一块 `__shared__`"，而是：
> **固定大小的 LDS page + controller 每指令维护 logical→physical page 映射 +
> per-page phase semaphore + 每指令动态 semaphore + 指令流水。**

---

## 0. 对照 HazyResearch Megakernels

上游的关键件与 DCU 版本的对应关系：

| HazyResearch Megakernels | 本实现 | 位置 |
| --- | --- | --- |
| `default_config`（228 KiB / 16 KiB page / 13 page / 16 consumer warps） | `Config`（64 KiB / 8 KiB page / 6 page / 4 consumer waves） | `csrc/dmk/config.h` |
| `instruction_state_t<config>`（instruction / timings / pid_order / semaphores / scratch） | `InstructionState<Config>` | `csrc/dmk/config.h` |
| `state<config>`（`pid(lid)` / `wait_page_ready` / `finish_page` / `next_instruction`） | `State<Config>` | `csrc/dmk/state.h` |
| `page<config>[NUM_PAGES]` 放在 `extern __shared__` | `Page<Config>[kNumPages]`，base 128 B 对齐 | `csrc/dmk/state.h`、`csrc/dmk/megakernel.h` |
| controller：fetch → page order → semaphores → publish | `controller::prepare_instruction` / `main_loop` | `csrc/dmk/controller.h` |
| `MAKE_WORKER` + 五个角色 | `BaseOp` 虚接口 + `make_op` 工厂 + `Role`/`worker_loop`（DCU 版去掉 launcher，见 §8） | `csrc/dmk/op.h`、`csrc/dmk/state.h` |
| `kittens::semaphore`（mbarrier） | `lds::Semaphore`（LDS 上的 parity 计数） | `csrc/dmk/lds.h` |
| `__syncwarp()` / `kittens::warp::sync()` | `lds::wave_barrier()`（`__builtin_amdgcn_wave_barrier`） | `csrc/dmk/lds.h` |
| `NoOp<config>` | `NoOp<Config, Globals>` | `csrc/dmk/noop.h` |
| Llama 的 matvec/attention op | `ElementwiseOp` / `GemmOp` / `EmbeddingOp` 作为具体实现 | `csrc/dmk/demo_op.h`、`gemm.h`、`embedding.h` |
| `get_smid()` 选择指令行 | `blockIdx.x` 选择指令行 | `csrc/dmk/controller.h` |
| TMA load/store、`cp.async.bulk` timings | 普通 HBM 读写；timings 默认关闭 | `gemm.h`、`embedding.h`、`config.h` |
| `setmaxnreg` / cluster / Blackwell tensor allocator | 不适用，未实现 | — |

保持不变的语义：

- 每个 block 一条指令流，`g.instructions[blockIdx.x][instruction_index][0..31]`；
- 指令第 0 个字是 opcode，0 是 NoOp；
- 每条指令有独立的 `pid_order`：`pid_order[lid] = pid`；
- 下一条指令的 page order 由**上一条指令**的 `release_lid` 映射出来；
- 每个物理 page 有 `page_finished[NUM_PAGES][STAGE_BITS]`，按指令序号取 phase bit；
- 每指令动态 semaphore 由 controller 构造，roles 用完后 controller 在复用该流水槽时 invalidate；
- 四个角色：consumer waves（前 N 个 wave）+ loader / storer / controller。launcher 在 Blackwell 参考实现里用于异步 MMA/TMA issue，DCU 同步 DUMMA 不需要，已去掉（§8）。

---

## 1. DCU 上的配置与预算

DCU 每个 CU 只有 **64 KiB LDS**（DTK 26.04 HIP 编程指南 §2.2），wavefront 64 lane。
因此不能照搬 H100 的 16 KiB × 13 page，本实现取：

| 项 | 值 |
| --- | --- |
| `kConsumerWaves` | 4 |
| 服务波 | loader / storer / controller，共 3 |
| `kTotalWaves` | 7，即 448 threads/block |
| `kPageSize` | 8 KiB |
| `kNumPages` | 6 |
| `kStages` | 2（instruction pipeline） |
| `kInstructionWidth` | 32 个 int（128 B） |
| `kDynamicSemaphores` | 16 |
| `kScratchBytes` | 4 KiB/槽 |

静态 LDS：

```
InstructionState<Config> × 2        = 2 × 5120 B = 10240 B
instruction_arrived  × 2            = 32 B
instruction_finished × 2            = 32 B
page_finished[6][1]                 = 96 B
semaphores_ready                    = 16 B
------------------------------------------------------
kStaticBytes                        = 10416 B
```

动态 LDS：

```
6 pages × 8192 B = 49152 B
launch 申请      = 49152 + 128（对齐 slack）= 49280 B
static + launch  = 59696 B < 65536 B
```

`config.h` 里有 `static_assert` 把这三项卡死；`launch_mk` 还会在运行期
用 `hipDeviceProp_t.sharedMemPerBlock` 和 `hipFuncGetAttributes().sharedSizeBytes`
再查一遍，避免"改了 tile 参数但没改预算"。

**occupancy**：~58 KiB/CTA，一个 CU 只能放一个 CTA；这正是 megakernel 想要的
"每 CU 一个常驻 CTA"形态。若希望两个 CTA/CU，需要把 page 数或 page 大小砍到 ≤32 KiB。

---

## 2. 静态 LDS：指令流水槽

`InstructionState<Config>`（`csrc/dmk/config.h`）逐字段对应上游：

```cpp
template <typename Config>
struct alignas(128) InstructionState {
    int instruction[Config::kInstructionWidth];   // 32 words
    int timings[Config::kTimingWidth];            // 128 ints（默认不写）
    int pid_order[Config::kNumPages];             // lid -> pid
    int num_semaphores;                           // 本指令动态 semaphore 数
    lds::Semaphore semaphores[Config::kDynamicSemaphores];
    int scratch[Config::kScratchBytes / sizeof(int)];
};
```

`mk_internal` 静态声明两块槽 + 三类 semaphore：

```cpp
__shared__ alignas(128) InstructionState<Config> instruction_states[kStages];
__shared__ lds::Semaphore instruction_arrived[kStages];
__shared__ lds::Semaphore instruction_finished[kStages];
__shared__ lds::Semaphore page_finished[kNumPages][kStageBits];
__shared__ lds::Semaphore semaphores_ready;
extern __shared__ int dynamic_lds[];   // 对齐 128 B 后解释成 Page<Config>[kNumPages]
```

初始化规则与上游一致：

- `instruction_arrived[i].init(1)`：controller 每条指令 arrive 一次；
- `instruction_finished[i].init(kWorkerWaves)`：每个 worker wave 每条指令 arrive 一次；
  本配置是 4 consumer + loader + storer = **6**；
- `page_finished[p][b].init(kConsumerWaves << b)`，并立即 `arrive` 一次，
  **完成 phase 0**，让第 0 条指令可以直接用 page；
- `semaphores_ready.init(1); arrive(1)`（保留位，当前未使用）。

`mk_internal` 里还有一个 `static_assert`，要求
`sizeof(instruction_states)+...== kStaticBytes`，防止静态布局与常量脱节。

---

## 3. 动态 LDS：page 与 logical/physical 映射

### 3.1 page

`Page<Config>` 是固定 8 KiB 的裸缓冲区，128 B 对齐：

```cpp
template <typename Config>
struct alignas(128) Page {
    int data[Config::kPageSize / sizeof(int)];
    __device__ void* ptr(size_t byte_offset = 0);
};
```

dynamic LDS 的 base 由 kernel 自己对齐到 128 B，然后直接
`reinterpret_cast<Page<Config>*>`；page `p` 的地址就是 `pages[p]`。
没有 arena、没有 bump pointer、没有 free list——页数在编译期固定，
生命周期完全由 controller 的置换和 `page_finished` 决定。

### 3.2 `pid_order` 与 `release_lid`

每条指令槽有自己的 `pid_order`：

```
pid_order[lid] = pid      // 逻辑页 lid 当前落在物理页 pid
```

- 第 0 条指令：`pid_order[lane] = lane`（identity）。
- 第 i 条指令（i > 0）：对每个物理槽 `q`（lane < NUM_PAGES），
  取**上一条指令**的 opcode，调用它的 `release_lid(instruction, q)` 得到 `lid`，
  然后 `pid_order_new[q] = pid_order_old[lid]`。

也就是说，新指令的物理槽 q 继承上一条指令逻辑页 `lid(q)` 的物理页。
op 通过 `release_lid` 返回的置换，声明"哪些逻辑页继续存活、以什么顺序轮转"。
上游 llama 的 `release_lid` 用 `ret_order[13]` 表达 3-stage weight pipeline 的轮转；
本仓库的 `ElementwiseOp` 用 identity，`NoOp` 也用 identity。

op 代码只使用逻辑页：

```cpp
int pid = s.pid(lid);        // 当前指令的映射
s.wait_page_ready(pid);      // 这个物理页对当前指令可用
s.pages[pid].ptr();          // 访问
```

**约束**：一个 op 必须在每条用到它的指令里，对每个 page 恰好完成一次
`finish_page`；未使用的 page 由 loader 立即释放（`NoOp` / `ElementwiseOp`
都是这么做的）。这样 `page_finished[p]` 的 phase 每条指令翻转一次，
`wait_page_ready` 的 `(instruction_index & 1)` 才总是对得上。

### 3.3 `page_finished` 的 phase 计数

沿用上游 `init_semaphore(N); arrive(N);` 的写法，page semaphore 在 kernel
启动时完成 phase 0（phase 变成 1），所以第 0 条指令 `wait(0)` 直接返回。
之后每条指令对每个 page 完成一次 phase：

| instruction | 期望 phase bit | 该 bit 在什么时候完成 |
| --- | --- | --- |
| 0 | 0 | kernel 启动时的 init+arrive |
| 1 | 1 | 第 0 条指令 finish_page |
| 2 | 0 | 第 1 条指令 finish_page |
| 3 | 1 | 第 2 条指令 finish_page |
| ... | `i & 1` | 第 `i-1` 条指令 finish_page |

`State::wait_page_ready(pid)` 对所有 `STAGE_BITS` 做
`wait((instruction_index >> bit) & 1)`；当前 `kStageBits = 1`。
`finish_page(pid, count)` 对每个 bit `arrive(count)`。

---

## 4. `lds::Semaphore`：DCU 上的 mbarrier parity

DCU 没有 mbarrier，`lds::Semaphore` 用 LDS 上的三个 32-bit 字实现同样的
**parity 语义**：

```cpp
struct alignas(16) Semaphore {
    uint32_t ticket;    // 累计到达数，只增
    uint32_t phase;     // 当前未完成 phase 的 parity（0/1）
    uint32_t expected;  // 每个 phase 需要的到达数
};
```

- `init(N)`：`ticket=0, phase=0, expected=N`。
- `arrive(k)`：`atomicAdd(ticket, k)`；当 `ticket` 恰好跨过 `N` 的整数倍时，
  由最后一个到达者把 `phase` 更新为 `(ticket/N)&1`。
- `wait(p)`：自旋到 `phase != p`，然后 `__threadfence_block()`。
  含义与 `mbarrier.try_wait.parity` 一致：等待 parity 为 `p` 的那个 phase 完成。
- `init(N); arrive(N);` 完成 phase 0，`phase` 变成 1，等价于 mbarrier 的
  "初始化后先翻转一次"。

例子（`expected=4`）：

```
ticket:  0 1 2 3 4 5 6 7 8 ...
phase:   0 0 0 0 1 1 1 1 0 ...
wait(0) 在 ticket=4 之后返回；wait(1) 在 ticket=8 之后返回。
```

与上游的差别与注意点：

- `expected` 必须非 0；`invalidate()` 会把三个字段清零，使残留 arrive 不再改 phase。
- 计数器是 32-bit；按每条指令每页一次的频率，2³² 条指令才会回绕，实际不可达。
- `arrive` 的最后一个到达者写 `phase`，其余到达者只做 release fence；
  这是有意避免对 pending counter 做 read-modify-write。
- `Semaphore` 的 16 B 对齐让它在 `InstructionState` 里布局稳定。
- 单 lane 发信号前调用 `lds::wave_barrier()`：gfx9 波前是 lockstep，但发散 copy
  循环退出后需要一个显式收敛点，再让 lane 0 `arrive`/`finish_page`；这对应上游的
  `__syncwarp()` / `kittens::warp::sync()`。

---

## 5. 角色与指令流水

### 5.1 波划分

`mk_internal` 按 `wave = threadIdx.x / 64` 分派：

| wave | 角色 | 数量 |
| --- | --- | --- |
| 0 .. `kConsumerWaves-1` | consumer | 4 |
| `kConsumerWaves` | loader | 1 |
| `+1` | storer | 1 |
| `+2` | controller | 1 |

worker 循环是 `worker_loop<Role, Config, Globals, Ops...>`。每条指令在本地
buffer 里构造一次当前 op，然后通过 `BaseOp*` 虚调用进入 op：

```cpp
alignas(16) unsigned char storage[Config::kOpStorageBytes];
for (instruction_index = 0, ring = 0; index < g.num_instructions;
     next_instruction()) {
    await_instruction();                    // wait instruction_arrived[ring]
    BaseOp* op = make_op<Config, Globals, Ops...>(
        instruction()[0], storage, instruction());
    if constexpr (Role == Loader) op->loader(s, g);
    else if constexpr (Role == Storer) op->storer(s, g);
    else op->consumer(s, g);
}
```

这里虚调用只发生在**每条指令、每个 role 一次**，不在元素循环里；op 内部的
数据循环仍然由编译器正常内联/优化。

`next_instruction()`：lane 0 `arrive(instruction_finished[ring])`，
然后 `index++`、`ring = (ring+1) % kStages`。
controller 在复用流水槽时 `wait(instruction_finished[ring], parity)`，
再 invalidate 上一条指令的动态 semaphore。

### 5.2 controller

`controller::prepare_instruction`（`csrc/dmk/controller.h`）四步：

1. **退休旧槽**：若 `instruction_index >= kStages`，等待
   `instruction_finished[ring]`，invalidate 该槽上次使用的
   `num_semaphores` 个动态 semaphore。
2. **取指**：lane `< kInstructionWidth` 从
   `g.instructions[blockIdx.x][instruction_index][lane]` 取 32 个字。
3. **page 置换**：第 0 条指令 identity；否则用上一条指令的 opcode 调
   `release_lid`，再按 `pid_order_new[q] = pid_order_old[lid]` 写出新映射。
4. **构造 op 与动态 semaphore**：lane 0 用 `make_op` 在本地 storage 里构造
   当前 op，调 `op->init_semaphores(s, g)`，把返回数量写进 `num_semaphores`；
   `NoOp`（opcode 0）也走同一条虚调用，返回 0。
5. **发布**：`wave_barrier + __threadfence_block()` 后 lane 0
   `arrive(instruction_arrived[ring])`。

controller 循环结束后还有一个 drain：对最后 `kStages` 条真实指令各等待一次
`instruction_finished` 并 invalidate 动态 semaphore，保证 kernel 退出前
worker 已经真正结束。

### 5.3 op 接口

每个 op 都继承 `BaseOp<Config, Globals>`，实现五个 hook：

```cpp
template <typename Config, typename Globals>
class MyOp final : public BaseOp<Config, Globals> {
public:
    static constexpr int opcode = N;

    __device__ explicit MyOp(const int* instruction)
        : BaseOp<Config, Globals>(instruction) {}

    // controller：新指令物理槽 query 应继承的旧逻辑页 id
    __device__ int release_lid(int query) const override;

    // controller：初始化本指令的动态 semaphore，返回数量
    __device__ int init_semaphores(State<Config>& s,
                                   const Globals& g) const override;

    // worker waves：各自角色的工作
    __device__ void loader(State<Config>& s, const Globals& g) const override;
    __device__ void consumer(State<Config>& s, const Globals& g) const override;
    __device__ void storer(State<Config>& s, const Globals& g) const override;
};
```

controller 和 worker 都通过 `make_op<Config, Globals, Ops...>(opcode, storage,
instruction)` 构造当前 op，并直接调用虚函数；controller 不再出现 opcode
switch。`Ops...` 是 kernel 实例化时的 op pack，新增 op 只需在
`megakernel.cpp` 的 pack 里加一个类型。

HIP/DCU 约束：

- **多态对象必须在 device 上构造**（HIP 文档：host 构造的对象不能拿到 device
  调虚函数）。本实现每条指令在每个 role 的本地 `alignas(16)` buffer 里
  placement-new 一次，调用完即丢弃。
- op 必须**可平凡析构**（factory 有 `static_assert`），因为 VM 复用 buffer
  不调用析构函数。
- op 对象大小 ≤ `Config::kOpStorageBytes = 64`，对齐 ≤ 16。
- op 目前是 header-only/template；如果拆到独立 TU，device 函数指针/vtable
  跨 TU 需要 HIP 的 `-fgpu-rdc`。
- 虚调用是每条指令每 role 一次；不要在 op 内部逐元素调用虚函数。

op 内部用 `s.pid(lid)`、`s.wait_page_ready`、`s.finish_page`、`s.semaphores()[i]`、
`s.scratch()` 访问 VM 提供的设施；不要直接碰 `pid_order` 或 page 数组下标。

写新 op 的检查清单：

1. 指令字布局定好，并把 host 侧序列化对齐；
2. 明确用哪些逻辑页、每条指令对每个 page **恰好 finish 一次**；
3. 未使用的 page 由 loader 立即 `wait_page_ready + finish_page`；
4. 若 op 有流水，`release_lid` 要按 pipeline 轮转，不能简单 identity；
5. 动态 semaphore 数量 ≤ `kDynamicSemaphores`，且每个都在 `init_semaphores` 里初始化；
6. 任何跨角色的数据依赖都走 `s.semaphores()`，不要依赖 wave 调度顺序；
7. op 必须可平凡析构，并且只通过 `BaseOp*` 使用。

### 5.4 参考 op：`ElementwiseOp`

`ElementwiseOp`（`csrc/dmk/demo_op.h`）用 1 个 page 做 `y = scale*x + bias`：

- 指令字：`[1]=tile`、`[2]=scale bits`、`[3]=bias bits`、`[4]=有效元素数`；
- lid 0：loader 从 HBM 搬入，4 个 consumer wave 分工缩放，storer 写回；
- lid 1..5：loader 立即释放；
- 动态 semaphore：`data_ready`（loader→consumer，expected 1）、
  `data_done`（consumer×4→storer，expected 4）；
- page 0 由 storer 在拷完后 `finish_page`（expected 4）。

`NoOp`（`csrc/dmk/noop.h`）负责指令流填充：loader 对 6 个 page 全部
`wait_page_ready + finish_page`，其他角色空操作。

### 5.5 具体 op

- **`GemmOp`**（`csrc/dmk/gemm.h`，opcode 2）：一条指令算一个 16×16×16 tile，
  page 0/1/2 分别是 A/B/C；loader 搬 A/B，consumer wave 0 做标量参考
  GEMM，storer 写 C。文件里同时保留低层 `GemmWave` / `run_gemm_wave`，
  它是原来的 DUMMA microbench，不经过 page VM。
- **`EmbeddingOp`**（`csrc/dmk/embedding.h`，opcode 3）：一条指令把 table 的
  一行 gather 到 output 的一行；page 0 做中转，consumer 当前为空，后续可加
  positional/scale。
- **`ElementwiseOp`**（`csrc/dmk/demo_op.h`，opcode 1）：`y = scale*x + bias`，
  用来验证数据通路和 4 个 consumer wave 的分工。
- **`NoOp`**（opcode 0）：page 释放与指令流填充。

`GemmOp`/`EmbeddingOp` 需要 `MkGlobals` 里的 `matrix_a/matrix_b/matrix_c` 和
`embedding_table/embedding_output`；demo 的 Elementwise/NoOp 不使用这些字段。

---

## 6. 代码结构

| 文件 | 内容 |
| --- | --- |
| `csrc/dmk/config.h` | `Config`、`InstructionState`、静态/动态/launch 字节常量与 `static_assert` |
| `csrc/dmk/lds.h` | `lds::Semaphore`（parity）、对齐、`phase_for_ticket` |
| `csrc/dmk/op.h` | `BaseOp<Config, Globals>` 虚接口、`make_op` 工厂、op 存储约束 |
| `csrc/dmk/state.h` | `Page`、`State`、`Role`、`worker_loop`、`remap_page_order` |
| `csrc/dmk/controller.h` | `prepare_instruction`、`main_loop`（op 通过 `BaseOp*` 直接调用） |
| `csrc/dmk/noop.h` | `NoOp<Config, Globals>` |
| `csrc/dmk/demo_op.h` | `ElementwiseOp<Config, Globals>` |
| `csrc/dmk/gemm.h` | 低层 `GemmWave`/`run_gemm_wave` + 页面版 `GemmOp` |
| `csrc/dmk/embedding.h` | `EmbeddingOp<Config, Globals>` |
| `csrc/dmk/megakernel.h` | `MkGlobals`、`mk_internal`、`mk` kernel 模板、`launch_mk` 声明 |
| `csrc/megakernel.cpp` | demo kernel 实例、动态 LDS/设备检查、`launch_mk` |
| `tests/csrc/test_dmk_lds.cpp` | host 逻辑测试 |
| `tests/csrc/test_dmk_lds_megakernel.cpp` | DCU 设备测试 |
| `tests/csrc/run_dmk_lds_test.py` | hipcc 直编 + 运行 |

---

## 7. 测试

### 7.1 host 逻辑测试（无需 DCU）

```bash
python tests/csrc/run_dmk_lds_test.py --host-only
```

覆盖：

- `Config` 预算：`kStaticBytes == ...`、`kStaticBytes + kLaunchBytes <= 64 KiB`；
- `sizeof(Semaphore)==16`、`sizeof(Page)==kPageSize`、`InstructionState` 128 B 对齐；
- `phase_for_ticket` 的 parity 序列（expected=1 和 4）；
- `remap_page_order` 的 identity / rotation 置换；
- `ring_advance/ring_retreat`；
- `MkGlobals::valid`：指令 shape、宽度、`num_instructions`、unit-stride 输入输出。

另外，本地（mock HIP/ATen 头）用真实 `State`/controller/op 代码路径跑过一个
6 指令的序列（NoOp → 2×Elementwise → NoOp → Elementwise → NoOp），
验证了 page phase、动态 semaphore、流水槽复用和最终数值；这个模拟没有提交，
只作为设计验证。

### 7.2 DCU 设备测试

```bash
source /opt/dtk/env.sh
python tests/csrc/run_dmk_lds_test.py
```

用例：

| 用例 | elements | blocks | 验证点 |
| --- | --- | --- | --- |
| `single` | 1 tile | 1 | 最小规模、NoOp 起始 |
| `multi` | 8 tiles | 2 | 多 block、page/流水槽复用 |
| `tail` | 3 tiles + 777 | 2 | 尾块有效元素数 |
| `idle` | 2 tiles | 4 | 没分到 tile 的 block 只跑 NoOp，不会卡死 |
| `noop` | 1 tile | 1 | 全 NoOp，输出保持哨兵值 |
| `relaunch` | 8 tiles | 2 | 第二次 launch 重新初始化 page phase |
| `gemm` | 16×16×16 | 1 | `GemmOp` 的 page/标量参考 GEMM 路径与数值 |
| `embedding` | 4×8 table | 1 | `EmbeddingOp` 的行 gather |

期望输出形如：

```
device: ... (gfx928)
wave=64 LDS/block=65536 static=10416 dynamic=49152 launch=49280 pages=6 x 8192 B
...
page-based megakernel checks passed
```

失败排查：

1. 先看 `launch_mk` 的 stderr（LDS 超限 / warpSize / launch 失败）；
2. 数值错 → 先看 op 的 page 使用与 `finish_page` 次数（每指令每页一次），
   再看 `release_lid` 是否与 host 指令流匹配；
3. 卡死 → 检查 `page_finished` 的 init+arrive 是否完成 phase 0、
   `instruction_finished` 的 expected 是否等于 worker waves、
   NoOp/loader 是否释放了所有未使用 page；
4. `--host-only` 排除预算/布局问题。

---

## 8. 限制与后续

0. **没有 launcher 波**：Blackwell 参考实现用 launcher 做异步 MMA / TMA 预取 /
   `tensor_finished` 握手；DCU 的 DUMMA 是 wave-synchronous，fragment 在发起波
   自己的寄存器里，另一个波无法代发，也没有 TMA。因此本实现只保留
   controller / loader / storer / consumer，block 从 512 线程降到 448，
   `instruction_finished` 的 expected 从 7 降到 6。以后若引入异步 copy 或
   异步 MMA，再把这个波加回来即可。
1. **page 数固定**：6 × 8 KiB 是 64 KiB 下的折中；真实 DPA4 op 需要按
   tile 大小和流水深度重新算 `kPageSize/kNumPages`，并同步更新 `static_assert`。
2. **没有 free list**：page 的生命周期完全由 `release_lid` 置换 + phase 表达；
   这是上游的有意设计，但也意味着 op 必须精确声明自己的页面轮转。
3. **32-bit semaphore**：`ticket` 回绕在 2³² 次到达之后；当前规模不可达，
   但超长持久化 kernel 需要换成 64-bit 或显式 epoch。
4. **generic atomic**：`atomicAdd` 作用在从 `extern __shared__` 推导出的
   generic 指针上，可能生成 flat atomic 而不是 `ds_add`；当前每指令只有几次，
   若 profile 显示瓶颈，可把 semaphore 移到静态 `__shared__` 或改用
   `__builtin_amdgcn_ds_add_*`。
5. **没有 TMA / cp.async.bulk**：loader/storer 是普通 HBM 读写；
   访存流水、16 B 向量化和 L2 复用是后续优化项。
6. **timings 默认关闭**，也没有 Blackwell 的 tensor allocator / cluster。
7. **持久化 kernel / grid 同步**：目前是普通 grid launch（每 block 一条指令流），
   不是常驻 kernel；跨 CTA 依赖仍走 HBM，未纳入本层。
8. **数值门禁**：`ElementwiseOp` / `GemmOp` / `EmbeddingOp` 是数据通路与同步的
   参考实现；接真实 GEMM/GIE 后仍需按 `docs/next-steps-plan.md` 做 fp64
   exactness 与整模对照。
9. **虚函数开销**：每条指令每个 role 一次虚调用，成本可忽略；但 op 内部
   不能逐元素虚调用。若未来要完全消除 indirect call，可以把 `BaseOp` 保留为
   接口，再加一套 compile-time 的 `OpId + 模板分发` 快路径。

---

## 9. 参考

- HazyResearch Megakernels：`Megakernels/include/megakernel.cuh`、
  `include/util.cuh`、`include/controller/controller.cuh`、
  `include/controller/page_allocator.cuh`、`include/noop.cuh`。
- `docs/next-steps-plan.md`：M0–M4 的路线与 LDS 风险（R7）。
- `dtk_doc/DTK 26.04 HIP C++编程指南.pdf`：64 KB/CU、32 bank × 4 B、wave64。
- `dtk_doc/DTK 26.04 HIP最佳实践手册.pdf`：LDS 分块、双缓冲、bank 冲突。
- ROCm HIP：[Launch API](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.1/reference/hip_runtime_api/modules/launch_api.html)
  （`extern __shared__`）、[Execution Control](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.2/doxygen/html/group___execution.html)
  （`hipFuncSetAttribute`）。
- CUDA mbarrier parity 语义：`mbarrier.try_wait.parity` / `test_wait.parity` 文档。
