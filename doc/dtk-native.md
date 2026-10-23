# DTK/HIP native 算子开发约定

本轮参考相邻 `../dtk_doc` 的本地资料，不假定 DCU 与 NVIDIA warp/tensor-core 指令相同。

## 参考与本轮落点

| 本地资料（PDF 页序，1-based） | 使用内容 | 工程落点 |
|---|---|---|
| DTK 26.04 HIP C++编程指南.pdf，13、25 | grid/block 索引、hipLaunchKernelGGL、设备数学函数 | `csrc/parallel/*.cu` 的线程索引与 launch |
| DTK 26.04 HIP Runtime API开发手册.pdf，40、41 | hipGetLastError、hipGetErrorString | `csrc/include/checks.h::launch_check` |
| DTK 26.04 HIP最佳实践手册.pdf，20、32 | LDS/寄存器双缓冲的资源权衡、event 计时 | 第一版不使用大 LDS；benchmark 用 GPU event |
| 随堂软件包 2/demo/02_demo1_stream/hip_Stream_test.cpp | stream 与异步数据搬运 | kernel 使用 PyTorch 当前 stream，不创建隐藏 stream |

## 首批内核

| 接口 | 分工 | backward | 当前边界 |
|---|---|---|---|
| edge_geometry | 一线程/边，计算位移/距离/方向 | 分析式梯度，原子和晶胞 atomicAdd | FP32；零距离方向定义为0、方向梯度为0 |
| radial | 一线程/边，循环径向频率；稳定 sinc + C3 | dr、dw，dw atomicAdd | Bessel；RBF 和 edge envelope 分别指定指数 |
| rmsnorm | 一线程/行，循环通道 | dx、dscale | 普通标量 norm，非等变 norm；待 block reduction |
| linear | 一线程/输出元素，循环 K | Torch GEMM backward | 正确性基线，尚未接 hipBLASLt/MFMA |
| segment_sum | 展平元素并 atomicAdd | Python/Torch gather | FP32，非确定性归约 |
| force_virial | 一线程/边，force/virial atomicAdd | 暂不提供二阶导 | inference assembly，多帧/mask；FP32 |

binding → Python `native.py` 自定义 autograd → `ops.py` backend dispatch。`dcu_dpa4.deepmd` 可把首批 primitive 接到原 DeePMD eager 模型；独立 `modeling` 另有 OMol25 的部分 reference stage 和 task-DAG planner，但尚未组成完整 E/F/V 模型或 HIP MegaKernel。整体模型/HIP 数值与性能仍需实际环境验证，见 `dpa4-megakernel-architecture.md` 和 `verification.md`。

## ABI、同步与安全

- 浮点 FP32、索引 INT64、mask BOOL；张量共享 DCU device。C++ 检查 shape/dtype/contiguous/device，Python wrapper 负责连续化。
- `CUDAGuard`/`getCurrentCUDAStream` 经 ROCm extension HIP 化后分别对应 HIP guard/current stream；不得自行在算子里同步全卡。
- host `hipGetLastError` 检查 launch，不证明异步执行无误；数值测试里的同步/断言会暴露执行错误。
- 所有 src/dst、segment index、frame 必须落在合法范围。当前 kernel 没有逐元素 bounds check；调用者必须保证合法，即使 mask=false 也不能传 -1。
- 默认不用 fast-math；先锁定数学、误差预算和梯度。浮点 atomic 次序可改变末位，验证使用明确容差，不能要求 bitwise 一致。
- p=0 envelope 常数1，r=0 稳定 sinc，E=0 空 launch，不访问越界。当前一阶 geometry/radial/norm autograd 明确不支持二阶导。

## 测试与计时

`tests/csrc/test_accuracy.py` 用同一 DCU 上的 Torch reference 对比 native forward/backward，覆盖多帧、mask、r=0/rcut、空边和非默认 stream。请先 assert `native_available()`，避免把全部 skip 当作通过。

`python -m tests.csrc.benchmark` 先比较结果再计时。输入预驻留 DCU，预热后用 event 统计多次 trial 的中位数；当前包含 Python wrapper、launch、分配与清零开销，不包括 H2D 和 backward。它是 primitive 基线，不是完整模型吞吐率。记录 GPU型号、torch/hip、N/E/C/R；native naive GEMM 可能慢于 Torch，这是预期的优化起点。

## 优化和 device 调度阶段

1. 普通 norm 改 block/wave reduction；归约构造 dst CSR 以减小 atomic 争用；GEMM 先接 DTK 支持的 BLAS 路径，再考虑 MFMA tile。
2. 建立 load/compute/store 数据生命周期、tile shape、LDS 字节和寄存器预算，先做独立融合 kernel，复现梯度。
3. 借鉴 ThunderKittens 的 loader/consumer/storer/controller 分工，实际 wave size、屏障与内存序必须依据 DCU 设备能力，不移植 CUDA 专属 TMA/WGMMA。
4. `csrc/megakernel/runtime.cu` 已加入 opt-in edge-feature task queue/worker：使用有界 tile ticket、task completion/dependency counters 与 generation/status 字段；stage 依赖在所有 producer tiles 完成后释放。dispatcher 目前只支持融合 `edge_features` 或 `edge_geometry -> radial_basis` 两阶段子图，不是完整图执行器。
5. persistent kernel 的跨 CTA 等待必须避免超出实际驻留 CTA 数导致死锁；当前 worker 限制为最多一个 block/SM，队列空且无 active worker 时通过 status 报 stalled，而不是永久自旋。HIP 编译、队列/依赖实机测试与 HIP Graph/普通 launch 对照仍未执行。角色化调度不能替代缺失的算子和 VJP。
