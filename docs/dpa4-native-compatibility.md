# DPA4 GIE / SO(2) native 算子兼容性检查

对照本地 `deepmd-kit/deepmd/dpmodel/descriptor/dpa4_nn/embedding.py` 的
`GeometricInitialEmbedding.call`、`so2.py` 的 `SO2Convolution.forward_envelope`，
以及 `pt_expt/descriptor/dpa4_nn` 的 PyTorch runtime bindings。

## 当前覆盖范围

| 当前实现 | 对应 DeepMD 阶段 | 限制 |
| --- | --- | --- |
| `csrc/gie.cpp` | zonal × radial gather → destination scatter → degree normalization；标量行保留为零 | contiguous FP32，int64 索引；adapter 对 spin、source gate、padded mask 等调用保留原实现 |
| `csrc/so2_reduce.cpp` | 已算好 edge message 的 envelope 加权 → destination scatter → degree normalization | contiguous FP32，attention-free；不包含 edge_message、Wigner 旋转、SO(2) mixing、attention 或 pre/post focus mixing |

这两个文件是局部算子，并非完整 GIE / SO2Convolution 替代品。adapter 依赖
DeepMD eager `pt_expt` 模型；旧 `pt` 和导出 `.pt2/.pte` 不支持加载后 patch。
当前 profiling CLI 仍只测试原始 DeepMD，不会启用这些补丁。

## 本次修正

- GIE 改为先汇总 edge message，再对 destination 结果归一化，与 DeepMD 顺序一致。
  FP32 下两者不是同一运算顺序：`[1e8, -1e8+8, 1]` 的先求和后乘 `0.3`
  得到约 `2.7`，逐项乘后求和约为 `4.3`。reference 同步修正。
- GIE 的归一化系数梯度也改为使用未归一化的 destination sum，再与 upstream 做点积；
  backward 重算该 sum，保持现有 forward ABI，增加了一次 scatter 和临时显存，性能需在设备上重新测量。
- 补齐两个 C++ 文件使用 `numeric_limits` 的直接 include，GIE 补齐空节点与空径向槽检查。
- SO(2) backward 检查完整三维 shape，拒绝元素总数相同但 coefficient/channel 不同的张量，
  避免错误内存布局被接受。
- SO(2) adapter 在权重相乘前统一 flatten envelope 和 source gate，避免 `(E,1)*(E,)`
  广播成 `(E,E)`；非 FP32 compute precision 保留原 DeepMD 路径。
- 修正原 GIE parity 测试的旧 `pt/sezm_nn` 路径、构造参数和索引转换；新增当前
  DPA4 SO(2) aggregation 方法的前向、一阶梯度、source gate、空边与形状错误对照。

## 验证

主机测试：

```bash
PYTHONPATH=python python3 -m unittest discover -s tests/python -v
```

目标 DTK/DCU 上重编译后运行真实 DeepMD 对照：

```bash
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python:$PWD/../deepmd-kit${PYTHONPATH:+:$PYTHONPATH}"
MAX_JOBS=4 python3 setup.py build_ext --inplace
DP_RUN_GIE_DEEPMD_PARITY=1 DP_RUN_SO2_DEEPMD_PARITY=1 \
  python3 -m unittest discover -s tests/csrc/accuracy -p '*deepmd_parity.py' -v
```

SO(2) 对照调用真实 `forward_envelope`，使用预计算 edge message，验证归约边界，
不等价于完整 SO2Convolution 对照。原子加仍有累加顺序不确定性，需按容差比较；
不承诺 bit-exact。自定义 backward 只针对一阶梯度，训练及 force-loss 的高阶梯度
不在当前接入范围内。C++ 入口仍要求调用方提供合法、范围内的 dst/radial_slot 索引。

本机没有 HIP/DCU 与已安装的 DeepMD，设备测试只能收集并跳过；CPU 测试不能证明
HIP 编译、性能或完整模型精度已通过。
