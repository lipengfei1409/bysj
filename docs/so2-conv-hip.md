# 把 fused SO(2) 卷积移植到 HIP（`csrc/so2_conv_hip/`）

目标：让 DeepMD 自己的融合算子 `deepmd::dpa4_so2_conv` 在 DCU 上可用。
**不修改 DeepMD 源码**——翻译后的算子注册进**同一个 `deepmd` 算子命名空间**，
于是上游的 Python 侧（`op_available()`、`_setup_context`、`_backward`、fake impl、
`_profitable`）原样生效。

- 上游实现：`deepmd-kit/source/op/pt/dpa4/so2_conv{,_kernel}.cuh` + 24 个 `(CF,L)` 特化
- 本目录：机械翻译 + 4 处人工修补，v1 只实例化 `c64_l3` 的 fwd/bwd
- 为什么值得做：算术投影 **55 ms** vs 稠密前向 **163.8 ms**（见 [`execution-model.md`](execution-model.md) §5）

## 1. 目录

| 文件 | 内容 |
| --- | --- |
| `tools/hipify_so2_conv.py` | **生成器**：copy → `torch.utils.hipify.hipify` → 4 处人工 patch → lint。改代码改这里 |
| `hip_compat.cuh` | 唯一手写的 shim：32-lane 单元、divergent barrier |
| `so2_conv.hip` | 主机侧：校验、CSR、op schema、`dpa4_fp32_ridge`（重写为 HIP 版） |
| `so2_conv_hip.cuh` | 布局代数 + warp 归约 |
| `so2_conv_kernel_hip.cuh` | 前向/反向 kernel 本体（1112 行） |
| `so2_conv_launch_hip.h` | 启动策略与 `(L,CF)` 分派 |
| `so2_conv_instantiate_hip.cuh` | 模板实例化收尾 |
| `so2_conv_fwd_c64_l3.hip` / `so2_conv_bwd_c64_l3.hip` | 一个特化的两个方向（各 10 行） |
| `build.py` | 自己驱动 `hipcc` 编译，再 `torch.ops.load_library` 注册（见 §4.1） |

`hipify` 的产物是自动生成的，**不要手改**；改 `tools/hipify_so2_conv.py` 后重跑：

```bash
python3 csrc/so2_conv_hip/tools/hipify_so2_conv.py   # 任何有 torch 的机器
```

## 2. 翻译映射

| CUDA | HIP | 说明 |
| --- | --- | --- |
| `cuda_runtime.h` / `ATen/cuda/*` | `hip/hip_runtime.h` / `ATen/hip/*` | hipify 自动 |
| `cudaStream_t`、`cudaXxx`、`cudaDevAttr*` | `hipStream_t`、`hipXxx`、`hipDeviceAttribute*` | hipify 自动 |
| `k<<<g,b,s,st>>>(args)` | `hipLaunchKernelGGL((k), g, b, s, st, args)` | hipify 自动；它会把 `dim3` 套两层，已归一化 |
| `__shfl_xor_sync(mask, v, d)` | `__shfl_xor(v, d, kWarp)` | **人工**，`kWarp = 32`，见 §3.1 |
| `__syncwarp()` | `dpa4_wave_barrier()` → `__builtin_amdgcn_wave_barrier()` | **人工**，见 §3.2 |
| `__pipeline_memcpy_async` / `_commit` / `_wait_prior` | `__builtin_memcpy` / 注释 / 注释 | **人工**，见 §3.3 |
| `dpa4_fp32_ridge()`（读 CUDA compute capability、128 lane） | `hipGetDeviceProperties` + `prop.warpSize` | **人工**，见 §3.4 |

## 3. 四个必须人工处理的点

### 3.1 32-lane 单元是算法的一部分，不是习惯

上游 kernel 的每一列输出都映射成 `j * 32 + lane`，`kWarp = 32` 切分通道维
（`CFB = CF / 32`）。ROCm 的 wavefront 是 64 lane，所以**保留 32-lane 单元、一个 wave 里跑两个**
——`__shfl_xor(v, d, kWarp)` 的 `width = 32` 让交换只在每个 32-lane 段内进行，语义与 CUDA 一致。
代价是波前占用率减半。**把宽度改成 64 是改算法，不是移植**，不在 v1 范围。

### 3.2 `__syncwarp()` 在发散代码里

四处 `__syncwarp()` 都在 `if (active) { … }` 内部（发布 warp 私有 stage、读回广播值）。
换成 `__syncthreads()` 会死锁；AMD 的 wave barrier 是保守超集（顺带同步同 wave 的另一半，
无害且更严格）。

### 3.3 异步 pipeline 变成同步

`__pipeline_memcpy_async` 每次拷 4 个 float（16 B），双缓冲让权重面板的搬运与计算重叠。
v1 直接换成 `__builtin_memcpy`，两个 wait 变成注释——因为前后各有一个 `__syncthreads()`
已经为共享缓冲排序，正确性不受影响，只丢了重叠。代码里留了 `v1: synchronous stage`
标记，v2 换 `hip::memcpy_async` 时按它改。

### 3.5 ATen 的 masquerading 头必须换成 c10（编译才会过）

hipify 会把 `ATen/cuda/CUDAContext.h` 变成 `ATen/hip/HIPContext.h`，而这个头会拉进
`ATen/hip/Exceptions.h`：

```
/usr/local/.../torch/include/ATen/hip/Exceptions.h:10:10: fatal error:
      'cusolver_common.h' file not found
```

`cusolver_common.h` 是 CUDA 的，ROCm/DTK 里没有。当前流在 c10 层有对应物、也不拉求解器头：
`c10/hip/HIPStream.h` 的 `c10::hip::getCurrentHIPStream()`。

**设备守卫干脆不用 torch 的**：它的头位置在每个 torch 构建里都不一样——这台机器上没有
`c10/hip/impl/HIPGuardImplMasqueradingAsCUDA.h`（第一次改就撞上），另一台可能是
`ATen/hip/impl/...`，第三台可能两处都没有。`hipGetDevice`/`hipSetDevice` 永远在，
所以生成器注入一个 20 行的 `dpa4::DeviceGuard` RAII 包装，行为等价（torch dispatch 通常
已经选好设备，这层只是保险）。

### 3.6 分派必须只引用"已编译的 shard"

第一次链接成功后 `load_library` 报：

```
undefined symbol: dpa4::conv_forward_launch<1, 32>(...)
```

原因是上游的 `DPA4_CONV_FOR_EACH_SHAPE` 展开全部 **12 个 `(CF, L)`** 组合，
而 v1 只编译 `(3, 64)`，于是分派函数引用了 11 个不存在的符号——链接不报错（它们是
`extern template` 声明），运行时 dlopen 才炸。生成器因此注入
`DPA4_CONV_SHIPPED_SHAPES(macro)`（由 `SHARDS` 列表生成），分派与
`conv_shape_instantiated()` 都改用它。主机测试会核对：分派里出现的每个 `(L, CF)`
都必须在 `build.SOURCES` 里同时有 fwd 与 bwd 的 shard。

### 3.4 `dpa4_fp32_ridge()` 必须重算

它决定 `_profitable` 的设备缩放，原实现读 CUDA 的 compute capability 和“128 FMA lane”：
在 AMD 上是错的。HIP 版用 `hipDeviceProp_t`：`2 × warpSize × CU × 时钟`（CDNA 的 fp32 FMA
宽度就是波前宽度）。本卡实测 ridge ≈ **28.7**（`dpa4_fp32_ridge()` 会打印），标定件是 73.2。

## 4.1 为什么不用 `torch.utils.cpp_extension.load`

第一次在 DCU 上构建就踩到了：

```
File ".../cpp_extension.py", line 2799, in object_file_path
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

ROCm 上 `load()` 会对每个源文件跑一遍 hipify，然后用
`hipify_result[s].hipified_path` 组对象列表；**对已经是 `.hip` 的源文件这个值是 `None`**，
于是 `sources` 里混进 `None`，ninja 初始化时 `basename(None)` 炸掉。

而且这个 hipify 本来就是多余的：这里的源文件已经是我们人工 patch 过的 HIP 代码，
再跑一遍只会自找麻烦。所以 `build.py` 改为**直接调 hipcc**：

```python
hipcc -c src.hip -o obj.o -O3 -std=c++17 -fPIC -fno-gpu-rdc --offload-arch=gfx928 \
      -D_GLIBCXX_USE_CXX11_ABI=<与 torch 一致> -D__HIP_PLATFORM_AMD__ -I<torch include> -I<port>
hipcc -shared -o dpa4_so2_conv_hip.so *.o -L<torch lib> -ltorch -ltorch_hip -lc10
torch.ops.load_library(so)
```

- `--offload-arch` 从 `torch.cuda.get_device_properties(0).gcnArchName` 取，去掉 `:sramecc+` 之类后缀；
- ABI 标志从 `torch._C._GLIBCXX_USE_CXX11_ABI` 取，不一致会加载失败；
- 产物落在 `~/.cache/dpa4_so2_conv_hip/`（可用 `DPA4_SO2_CONV_HIP_BUILD` 覆盖），
  源码或头文件更新后自动重编，`--clean` 强制全编；
- `--dry-run` 只打印编译命令，**在没卡的机器上也能看**（先过 HIP 检查会提前退出）。

## 4. 构建与验证（DCU）

```bash
cd ~/private_data/lipengfei/bysj
source /opt/dtk/env.sh
git pull

python3 csrc/so2_conv_hip/build.py --check
```

期望输出：

```
registered: True
  deepmd::dpa4_so2_conv: ok
  deepmd::dpa4_so2_conv_backward: ok
  deepmd::dpa4_fp32_ridge: ok
default schema:
  dpa4_so2_conv(Tensor x, Tensor src, ...) -> (Tensor out, Tensor alpha, Tensor pre_gate, Tensor z_all)
dpa4_fp32_ridge(): 28.7
```

`registered: True` 之后，DeepMD 自己的 `op_available()`（`pt_expt/kernels/cuda/dpa4/so2_conv.py`）
会返回 True，`make_cuda_so2_conv` 就会去构造入口——但**还需要过 `_is_supported` 和 `_profitable`**。

## 4.5 单算子测试：直接跑上游自己的测试

上游为这个算子带了一整套参考测试（`source/tests/pt/model/test_descriptor_sezm_cuda.py`
的 `TestSeZMConvCuda` 与 `TestSeZMConvCudaGate`）：**每个 zoo 形状的前向 vs 稠密转录**、
**反向 vs autograd**、归约可复现、`make_fx` 追踪、零边/零 envelope、多头拒绝。
它们以 `op_available()` 为门，所以在收集测试之前把本移植注册进去即可全部生效——
这正是 `tests/dcu/so2_conv_hip_plugin.py` 做的事（pytest 插件，收集前 `build()`）。

```bash
DEEPMD_SRC=~/private_data/lipengfei/deepmd-kit \
  python3 tests/dcu/run_upstream_conv_test.py
```

它先 `build.py`（预热扩展缓存、顺带打印 schema），再用 `-k ConvCuda` 只跑卷积那几组。
第一次在 DCU 上跑，这一条命令同时回答"编得过吗"和"数值对吗"。

## 5. 已知取舍（v1）

| 取舍 | 影响 | 何时修 |
| --- | --- | --- |
| 只实例化 `(lmax=3, CF=64)` | 只服务本 checkpoint | 需要时在生成器的 `FILES` 里加 shard |
| 32-lane 单元 → 半波前占用 | 有效吞吐上限约 50% | 需要真正提速时改成 64-lane 算法 |
| 同步 staging | 权重面板搬运不与计算重叠 | 实测确认是瓶颈后 |
| 未接 `_profitable` 旁路 | 上游仍会拒绝本形状（225 280 vs 25 718） | 下一步：`deepmd_patch` 里加显式开关 |

## 6. 与 DeepMD 的接线（下一步）

算子注册只是第一半；要让模型真的走到它，还需要：

1. 在构造 `DeepPot` **之前**设 `DP_CUDA_INFER=2`（gates 在模型构造时读取）；
2. 绕过 `_profitable`（它拿 Triton 组合当基准，而 Triton 在本机编不出来），
   并把这个判断写进 `FusedStats`，不要静默；
3. 用 `verify_fused.py` 的 e2e 判据对比：`fused` 计数、launch 数、设备时间。

第 1、2 步是 `DPA4Model` 的事，改动很小，但**必须先有本文 §4 的 `registered: True`**。

## 7. 验证清单

| 检查 | 在哪 | 状态 |
| --- | --- | --- |
| 翻译无 CUDA-only token | `tests/python/test_so2_conv_hip_port.py`（主机） | ✅ |
| `kWarp` 仍是 32、shuffle 宽度 32 | 同上 | ✅ |
| 发散 barrier 用 wave 内建 | 同上 | ✅ |
| schema 与上游逐字一致 | 同上（C 字面量拼接后比对） | ✅ |
| ridge 用 HIP 属性重算 | 同上 | ✅ |
| **能编译** | DCU：`python3 csrc/so2_conv_hip/build.py` | ⬜ 待跑 |
| **数值与稠密一致** | DCU：一次 conv 的 A/B | ⬜ 待跑 |
| **端到端收益** | DCU：`verify_fused.py --stage all` | ⬜ 待跑 |
