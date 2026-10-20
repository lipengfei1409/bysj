# HIP 算子库

`bindings.cpp` 暴露 pybind 接口；`include/` 定义合同和检查；`parallel/` 是 edge/row/scatter 算子，`gemm/` 是 linear 基线。通过根目录 setup.py 构建 `dcu_dpa4._C`，Python 封装在 `python/dcu_dpa4/native.py`，对外统一接口在 `ops.py`。

已有六类 primitive 源码；geometry、radial、普通 RMSNorm 有 native 一阶 backward。linear/segment 的 backward 暂用 Torch。force/virial 暂不支持二阶导。所有源码必须在实际 ROCm/DTK PyTorch 环境编译验证；本地 Mac 无 HIP 编译结论。

还未实现 Wigner、SO2、等变 norm、null-mass attention、grid、fitting 与完整 VJP；不要将普通 RMSNorm 当等变 RMSNorm。

precision/性能测试位于 `tests/csrc/`，不是放在算子源目录。构建/运行命令见根 README；数学 shape 见 `doc/dpa4-operators.md`，DTK 与 stream 约定见 `doc/dtk-native.md`。
