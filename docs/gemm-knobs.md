# 没有 profiler 时怎么定位 dense GEMM

这台 DCU 上 **`rocprofv3` 不存在**（`command not found`），所以用**对照实验**代替计数器：
每次只改一个决策，重编（只有一个 TU），量同一个形状。三个开关都在 `csrc/gemm.cpp`，
由 `setup.py` 从环境变量转成 `-D`。

## 三个开关

| 环境变量 | 默认 | 改变什么 | 验证哪个假设 |
| --- | --- | --- | --- |
| `DPA4_DENSE_PAD` | 4 | LDS 行距 padding（0 = 关闭） | **bank conflict** |
| `DPA4_DENSE_KSTAGE` | 1 | 每次屏障覆盖几个 k8 步 | 屏障开销 / A 行宽 |
| `DPA4_DENSE_CCHUNK` | 0 | 1 = 尾声用 4 KiB 分块缓冲（省 12 KiB LDS） | **occupancy** |

## 实验矩阵

```bash
cd ~/private_data/lipengfei/bysj && source /opt/dtk/env.sh

run() {  # $1 = 标签
  DPA4_DENSE_PAD=$PAD DPA4_DENSE_KSTAGE=$KSTAGE DPA4_DENSE_CCHUNK=$CCHUNK \
    MAX_JOBS=8 python3 setup.py build_ext --inplace >/dev/null 2>&1 || { echo "$1 build failed"; return; }
  local out
  out=$(DPA4_DEBUG_GEMM=1 PYTHONPATH=python python3 -m dmk.scripts.bench_gemm \
        dense --shape 2048x2048x2048 --iterations 20 2>&1 | tail -2 | tr '\n' ' ')
  echo "$1: $out"
}

PAD=0 KSTAGE=1 CCHUNK=0 run "baseline(pad0)"
PAD=4 KSTAGE=1 CCHUNK=0 run "pad4"
PAD=4 KSTAGE=2 CCHUNK=0 run "pad4+kstage2"
PAD=4 KSTAGE=1 CCHUNK=1 run "pad4+cchunk"
PAD=4 KSTAGE=2 CCHUNK=1 run "pad4+kstage2+cchunk"
```

每行会打印 `regs / static_lds / kMT kNT kKT` 与耗时、TFLOP/s。

## 判读

| 观察 | 结论 |
| --- | --- |
| `pad4` 明显快于 `pad0` | bank conflict 是主因之一，padding 有效 |
| `pad4` 与 `pad0` 相同 | 冲突不是主因 → 看 occupancy（下一行） |
| `cchunk` 明显更快 | **occupancy** 是主因：LDS 从 ~21 KiB 降到 ~9 KiB，block/CU 增加 |
| `cchunk` 更慢 | occupancy 不是瓶颈（多出的 4 次屏障有代价），回到 k-stage |
| `kstage2` 更快 | 屏障 / A 行宽（32 B → 64 B）是主因 |
| 全部接近 rocBLAS 的 0.1 | 还有结构性问题（mma 宽度、C 写回路径），需要 profiler 才能继续 |
