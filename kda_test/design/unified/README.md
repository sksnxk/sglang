# KDA 6 算子统一验证目录

KDA（Kimi Linear Delta Attention）chunked attention 拆解为 6 个 kernel，本目录
对每个 kernel 的 **torch_npu 元算子版**（基准）与 **triton kernel 版**（被测）
做统一的正确性 + 性能对比。

- K1 `gate_chunk_cumsum` — Gate 激活 + chunk 内 cumsum
- K2 `token_parallel` — 对角线 Aqk/Akk
- K3 `inter_solve` — 非对角线 Aqk/Akk + 对角块前向替换求逆
- K4 `recompute_w_u` — 解耦 w/u/kg 重计算
- K5 `delta_rule_h` — Delta Rule H 状态递推
- K6 `gla_output` — GLA Output（跨块 + 块内）

## 目录结构

```
unified/
├── README.md             本文件
├── gen_cases.py          生成 105 个 case → cases_meta.json（仅元信息，不落张量）
├── cases_meta.json       （生成物）case_id → {B,T,H,K,V,group,desc}
├── bench.py              主脚本: 模式 A 正确性 + 模式 B msprof 分段采集
├── correctness.csv       （生成物）每 case 每 kernel 的 max_diff / status
├── profile_meta.json     （生成物，模式 B）每段 (case,kernel) 元数据
├── per_case_profile.py   解析 msprof op_summary: marker 切分 → results.csv
├── results.csv           （生成物）case×kernel 的 torch_us / triton_us / speedup
├── analyze_results.py    解析 results.csv: 加速比矩阵 + 每 kernel 汇总 + 相对时间
├── run_cpu.sh            容器启动器（source CANN + LD_LIBRARY_PATH + AUTOLOAD=0）
└── prof*/                （生成物）msprof 输出目录
```

6 个算子源码在各自目录 `../<op>/src/*.py`，bench.py 把各 `src/` 加进 sys.path 后
import。详见 `../UNIFIED_BENCH_PLAN.md`。

## 数据流（每 case，见 UNIFIED_BENCH_PLAN.md §4.1）

```
g           = K1_torch(x, A_log, dt_bias)                      # [B,T,H,K]
Aqk_d, Akk  = K2_torch(q, k, g, beta, scale)                   # [B,T,H,BT],[B,T,H,BC]
Aqk_nd, Akk_inv = K3_torch(q, k, g, beta, Akkd=Akk, scale)    # [B,T,H,BT],[B,T,H,BT]
w, u, kg    = K4_torch(k, v, beta, A=Akk_inv, gk=g)            # 各 [B,T,H,K]
h, v_new    = K5_torch(kg, w, u, gk=g, initial_state=init, idx)
Aqk_merge   = Aqk_d + Aqk_nd
o           = K6_torch(q, v_new, g, Aqk=Aqk_merge, h=h, scale)
```

对 K1..K6 **各自**用完全相同的输入（torch 链算出的中间量）分别跑 `K_torch` 与
`K_triton`，得到 max_diff 与 msprof 内核时间。

## 快速开始

### 1. 生成用例表（宿主机可跑，不依赖 torch/NPU）

```bash
python3 gen_cases.py             # → cases_meta.json (105 个 case)
python3 gen_cases.py --limit 5   # 只生成前 5 个（冒烟）
```

105 个 case 的分布（T 覆盖 1k–128k）:

| 组 | B | H | T | 数量 | 说明 |
|----|---|---|---|------|------|
| A. T 主扫描（2 的幂） | 1,2,4 | 2,4,8 | 1024..131072（8 档） | 72 | 全支持（K=V=64）|
| B. T 非 2 幂 / 边界 | 1 | 8 | 1023,1025,…,100000（20 档） | 20 | 尾 chunk 不满、非对齐 |
| C. 多 batch 放大 | 8 | 4,8 | 4096..32768 | 8 | 大 batch/head |
| D. 不支持演示 | 1 | 2,8 | K=V=128 {1024,16384}×{2,8} + K=V=32 {4096,H=4} | 5 | K5 不支持 → 标注 |

### 2. 模式 A — 正确性（默认，无计时）

```bash
# 容器内执行（必须；宿主 triton 无 ascend driver）
bash run_cpu.sh                           # 全 105 个 case
bash run_cpu.sh --limit 3                  # 只跑前 3 个 case（冒烟）
bash run_cpu.sh --group A                  # 只跑 A 组
bash run_cpu.sh --start 46 --limit 4       # 从第 46 个开始跑 4 个
```

输出 `correctness.csv`:

```
case_id,B,T,H,K,V,kernel,max_diff,status
A_B1_H2_T1024,1,1024,2,64,64,K1,7.629395e-06,OK
A_B1_H2_T1024,1,1024,2,64,64,K2,0.000000e+00,OK
...
A_B1_H2_T32768,1,32768,2,64,64,K2,K2 grid(flattened)=65536 超过 NPU coreDim 上限 65535,不支持
...
```

`status` ∈ {`OK`, `不支持`, `FAIL`}；`max_diff` 为 fp32 绝对误差最大值
（`不支持` 时填原因，`FAIL` 时填 `RUN_ERR: ...`）。

### 3. 模式 B — msprof 性能采集

```bash
# msprof 包裹 bench.py --msprof，对每个支持的 kernel 在 marker 分界内跑
# K_torch N 次 + K_triton N 次（默认 warmup=2, repeats=5）
bash run_cpu.sh --msprof ./prof                       # 全 105 个 case
bash run_cpu.sh --msprof ./prof --repeats 3 --warmup 1
bash run_cpu.sh --msprof ./prof --limit 5             # 冒烟: 只跑前 5 个
```

输出 `prof/PROF_*/mindstudio_profiler_output/op_summary_*.csv`（msprof 原始）+
本目录 `profile_meta.json`（段元数据）。

### 4. 解析 msprof → results.csv

```bash
python3 per_case_profile.py --latest-dir ./prof
```

按 `_kda_bench_marker` 行把 op_summary 切分为 (case, kernel) 段，段内:
- 以该 kernel 的 triton op 名开头的行 = triton 时间；
- 其余（`aclnn*` 等）= torch_npu 拼接时间。

输出 `results.csv`（UNIFIED_BENCH_PLAN.md §4.4 schema）:

```
case_id,B,T,H,K,V,kernel,torch_us,triton_us,speedup,max_diff,status
A_B1_H2_T1024,1,1024,2,64,64,K1,653.160,67.780,9.636,7.629395e-06,OK
...
```

`torch_us` = 该 kernel torch_npu 拼接总时长，`triton_us` = triton N 次调用总和，
`speedup = torch_us / triton_us`；`status` 来自 `correctness.csv`。

### 5. 分析结果

```bash
python3 analyze_results.py                           # 默认读 results.csv
python3 analyze_results.py --pivot-case A_B1_H8_T16384
```

输出:
1. **加速比矩阵** — 行=case（按 T/H/B 排序），列=K1..K6，格=`speedup`（`-`=不支持）；
2. **每 kernel 汇总** — 支持/不支持 case 数、平均/中位/最小/最大 speedup；
3. **6 kernel 相对时间** — 代表 case（默认取 T∈{1024,4096,16384,65536} 各一个）
   下 6 kernel 的 `torch_us` / `triton_us` 占比表。

## 已知约束（`_kernel_supports()`）

bench.py 在调用 triton kernel 前检查 case shape，不支持的直接标注 `不支持` 并跳过
（不污染 msprof trace）:

| kernel | 约束 | 原因 |
|--------|------|------|
| 全部 | 展平 grid 数 ≤ 65535 | triton-ascend 把 3D/2D grid 展平为 1D，超过 NPU coreDim 上限会报 `ERR00100 "value 65536 for parameter coreDim is invalid"` 且可能污染后续 NPU 状态 |
| K5 | K=V=64 | flat 1D store 假定行步长 K=64（上游约束） |
| K2/K3/K4 | K≠32 | `BK=next_power_of_2(32)=32` 时 `tl.dot` 数值不稳定 |
| K1/K6 | K=32 时仅 T≤256 | K=32 大 T 下精度不足（max_diff > 1e-2） |

## msprof 分段：marker kernel

每个 (case, kernel) 段前后各发一个 `_kda_bench_marker`（1 元素 store，微秒级）做分界，
`per_case_profile.py` 按行序找 marker 把 trace 切成精确段:

- 段内 Op Name 以该 kernel 的 triton op 名（`_gate_cumsum_kernel` /
  `_token_parallel_kernel` / `_inter_solve_kernel` / `_recompute_w_u_kernel` /
  `_delta_rule_h_kernel` / `chunk_gla_fwd_kernel_o`）开头的行 = triton 时间；
- 其余 = torch_npu 拼接时间；
- marker 行本身不计入；
- 首个 marker 之前的行（setup 派生 + 进程启动）归入虚拟 `setup` 段，不计入；
- `--repeats N` 只影响段内行数，不影响分段正确性（marker 精确切分）。

## 注意事项

- **必须容器内执行**: `bash run_cpu.sh` 需在容器 `triton-ascend-env-zhm` 内运行
  （宿主 triton 无 ascend driver）。从宿主机:
  `docker exec -w <dir> triton-ascend-env-zhm bash run_cpu.sh ...`
- **性能口径只有 msprof**: 不做 wall-clock 计时；唯一性能来源是
  `op_summary.csv` 的 `Task Duration(us)`（设备侧 kernel 时间）。
- **K5 in-place**: delta_rule_h 的 triton/torch 版都会 in-place 更新
  `initial_state`；bench.py 为 torch/triton 各保留独立的 init 张量，每次调用前
  用 `init_backup` 复位，避免跨实现污染。
- **NPU arange 陷阱**: `torch.arange(B, dtype=torch.int32, device="npu")` 在
  多种 B 下产生垃圾值（如 `4547155312683678328`）；bench.py 在 CPU 创建后
  `.to(device)` 规避。
- **大 T 内存**: T=131072/B=8/H=8/K=64 的单张 q/k/v/g 约 4GB（fp32）；
  bench.py 及时释放中间量，910B2 内存足够。
