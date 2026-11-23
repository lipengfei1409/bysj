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
| `build.py` | 用 `torch.utils.cpp_extension.load` 编译并注册 |

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

### 3.4 `dpa4_fp32_ridge()` 必须重算

它决定 `_profitable` 的设备缩放，原实现读 CUDA 的 compute capability 和“128 FMA lane”：
在 AMD 上是错的。HIP 版用 `hipDeviceProp_t`：`2 × warpSize × CU × 时钟`（CDNA 的 fp32 FMA
宽度就是波前宽度）。本卡实测 ridge ≈ **28.7**（`dpa4_fp32_ridge()` 会打印），标定件是 73.2。

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
