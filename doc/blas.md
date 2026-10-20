# rocBLAS 库封装

本轮封装本地 DTK 26.04《rocBLAS 库使用手册》的 handle/stream、SGEMM/DGEMM 和 strided-batched GEMM。BLAS 是 **host 调用、DCU 执行** 的库级基线，不能在 device consumer/megakernel 内调用；device 指令路径仍使用 `ops/group/dumma.cuh`。当前没有实现 hipBLASLt、BLAS1/2、FP16/BF16、指针数组 batched 或 fused activation epilogue。

## 文件与接口

- `include/blas/layout.hpp`：可独立host编译的 `row_major_plan`，检查32位尺寸及leading dimension。
- `include/blas/rocblas.hpp`：header-only host封装；`handle`、`gemm<float/double>`、`gemm_strided_batched<float/double>`。
- `csrc/gemm/blas.cu`：Torch current-device/current-stream接线、空维度与scale epilogue、按线程/设备/stream缓存handle。
- `python/dcu_dpa4/blas.py`：`gemm`、`linear`，显式 `backend='reference'/'rocblas'`；native forward调用rocBLAS，backward使用Torch矩阵乘法。
- `tests/csrc/test_blas_accuracy.py`：真实DCU精度/梯度/stream/多设备测试；`benchmark_blas.py`：精度检查后的event微基准。

## Python 合同

```python
from dcu_dpa4 import blas

# A[M,K], B[K,N], C[M,N]，返回新的Y，不原地改写C
Y = blas.gemm(A, B, C, alpha=1.0, beta=0.5, backend='rocblas')
# A物理布局[K,M]时，trans_a=True；B物理布局[N,K]时，trans_b=True
Y = blas.gemm(A_transposed, B, trans_a=True, backend='rocblas')
# 同样接口支持3D，[B,M,K]×[B,K,N]→[B,M,N]，不广播batch
Y = blas.gemm(A_batch, B_batch, backend='rocblas')
# linear weight是[K,N]，不是nn.Linear的[N,K]
Y = blas.linear(X, W, bias, backend='rocblas')
```

所有数值tensor必须为相同device/dtype的float32或float64，C不允许广播，bias为[N]；alpha/beta必须是有限标量且不溢出输入dtype。系数按输入dtype表示，包含FP32下溢到0的语义。接口拒绝autocast，避免隐藏精度改变。native只支持ROCm/DTK设备，不隐式搬到CPU；非连续输入由Python wrapper连续化，性能对比应包括或明确排除这些copy。

`beta=0`不读C（包括NaN C），`alpha=0`不读A/B进行乘法；K=0输出beta*C或0；空M/N/batch返回对应空tensor。没有C时只允许beta=0。C的原对象不改写；非零beta的native普通GEMM会先把C拷到新输出，然后让rocBLAS原地累加该输出。

验证一阶A/B/C梯度；native backward和bias epilogue均不是自定义rocBLAS kernel，也不代表已经完成全DCU融合VJP。没有自动加入DeePMD adapter/backend配置，没有改变已有`ops.linear(..., backend='hip')`路径。

## row-major 到 column-major

手册第7节注明rocBLAS使用column-major。不能将Torch的row-major指针直接配原来的M,N,K/lda调用。

```text
Yrow = op(Arow) × op(Brow)
Yrowᵀ = op(Brow)ᵀ × op(Arow)ᵀ
```

因此封装交换A/B指针、M/N、leading dimension、transpose flags；batched调用也交换A/B stride。无需真的转置设备内存。host测试用非方形、带padding的矩阵逐元素验证全部4种转置组合，避免方阵掩盖错误。

## C++ host 使用

编译添加 `-I include -DDPA4_ENABLE_ROCBLAS=1`，链接 `-lrocblas`；包含 `blas/rocblas.hpp` 不会把BLAS依赖强加给所有device指令头文件。

```cpp
#include "blas/rocblas.hpp"

// 在正确hipSetDevice上下文中创建一次；不要每个MD timestep重复创建。
dcu::blas::handle owner;
// A[M,K], B[K,N], C[M,N]；ld和batch stride的单位均为元素。
dcu::blas::gemm(owner, stream, M, N, K,
                A, K, false, B, N, false, C, N, 1.f, 0.f);
dcu::blas::gemm_strided_batched(owner, stream, M, N, K,
                A, K, M*K, false, B, N, K*N, false,
                C, N, M*N, batch_count, 1.f, 0.f);
```

与Python不同，C++低层API的C是**原地输出/累加对象**。A/B不能与C重叠；调用方负责device内存分配容量、对齐、生命周期和stream依赖。低层要求K>0，K=0的scale epilogue由caller执行，Torch绑定已处理。低层允许输入batch stride=0复用矩阵，但不允许输出batch重叠；Python层暂不开放batch广播。

handle记录创建它的device，不可复制；首次调用绑定指定stream，后续不能改到另一个stream，避免并发任务共享handle内部workspace。调用时设置host pointer mode（alpha/beta是host标量）、atomics-not-allowed。不保证跨架构/库版本bitwise一致。缓存按host线程、device和stream隔离，不把同一可变handle给多个host线程同时使用；大量动态创建stream会增加缓存资源，当前在线程退出时统一释放。销毁时尝试回到创建device；没有在每次GEMM后显式同步。库初始化和handle销毁仍可能有库内部开销，不能把它描述为完全无同步保证。

未声明支持HIP graph capture；如需捕获，应先预热handle/库并另作实机验证。调用者负责跨stream输入依赖，wrapper不自动等待任意其他stream。

## 构建与实机验证

在工程根目录、已有DTK/ROCm PyTorch环境中：

```bash
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
DPA4_BUILD_HIP=1 DPA4_BUILD_ROCBLAS=1 MAX_JOBS=4 \
  python setup.py build_ext --inplace
python -c 'from dcu_dpa4.blas import _rocblas_extension; print(_rocblas_extension().blas_available())'
DPA4_RUN_ROCBLAS_TESTS=1 \
  python -m unittest tests.csrc.test_blas_accuracy -v
python -m tests.csrc.benchmark_blas --m 192 --k 16 --n 16
python -m tests.csrc.benchmark_blas --batch 64 --m 16 --k 16 --n 16
```

如果`rocblas/rocblas.h`或`librocblas.so`未被找到，检查真实DTK安装路径和当前环境，配置编译include/library搜索路径及运行时`LD_LIBRARY_PATH`；不要安装CUDA或盲目覆盖系统依赖。`DPA4_BUILD_ROCBLAS=1`必须同时设置`DPA4_BUILD_HIP=1`。不用BLAS时保持原构建命令即可，不需要链接rocBLAS。老扩展/未开启BLAS构建会报明确重建提示，不会fallback。

`DPA4_RUN_ROCBLAS_TESTS=1`请求实机验证后，扩展/rocBLAS/设备缺失会失败；只有两卡隔离测试在单卡时允许skip。未设置该变量时4项实机测试skip，不表示数值正确。

微基准要求真实DCU，先比较Torch最高FP32 matmul精度，再进行10次warmup、5组event计时取中位数。测量包括forward wrapper的分配、dispatch及实际矩阵计算，不是纯内核时间；Torch matmul自身也可能调用rocBLAS，不应预设封装必定比Torch更快。不能将这个时间直接解释成完整DPA4或LAMMPS加速率。

## 本地可验证与未验证边界

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  python3 -m unittest tests.python.test_blas tests.csrc.test_blas_layout -v
```

本机无hipcc/DCU：仅验证Python reference合同、host layout、未启用rocBLAS时header自包含、pybind声明；实际BLAS header/API兼容性、HIP源码编译、链接、设备精度及性能仍需上面的实机命令。实现依据为本地DTK手册5.3.1、5.6.1与7.2。
