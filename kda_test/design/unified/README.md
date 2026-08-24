# KDA 6 算子统一 bench — 脚本使用文档

> 本文档是 `unified/` 目录下统一测试框架的**使用说明**，由原统一测试方案
> `design/UNIFIED_BENCH_PLAN.md` 与 `unified/README.md` 的用法内容合并而来。
> **测试结果分析见独立的 `ANALYSIS.md`**（本文档不含实测数据/结论）。

KDA（Kimi Linear Delta Attention）chunked attention 拆解为 6 个 kernel，本框架
对每个 kernel 的 **torch_npu 元算子版**（基准）与 **triton kernel 版**（被测）
做统一的正确性 + 性能对比，产出可 pivot 的 CSV。

- K1 `gate_chunk_cumsum` — Gate 激活 + chunk 内 cumsum
- K2 `token_parallel` — 对角线 Aqk/Akk
- K3 `inter_solve` — 非对角线 Aqk/Akk + 对角块前向替换求逆
- K4 `recompute_w_u` — 解耦 w/u/kg 重计算
- K5 `delta_rule_h` — Delta Rule H 状态递推
- K6 `gla_output` — GLA Output（跨块 + 块内）

- **目标 case**: `B=1, T=16384, H=96, K=V=128`（case_id `D_KV128_H96_T16384`）
- **硬件**: Ascend 910B2（NPU 20 核），CANN 9.0.0 / triton-ascend 3.2.1 / torch_npu 2.7.1
- **运行容器**: `triton-ascend-env-zhm`

---

## 1. 需求回顾

统一框架满足以下原始需求（来自统一测试方案）：

1. 构造 **100+ 测试用例**（序列长度 T 覆盖 1k–128k）；
2. 每个 case **按顺序**调用 6 个算子的 **torch_npu 版本**和 **triton 版本**；
3. 支持 **msprof 包裹**测内核性能；
4. 输出 **CSV**，可分析：
   - (a) 每个 kernel 在**每个 case** 下 triton 相对 torch_npu 的**加速比**；
   - (b) 同一输入 case 下 **6 个 kernel 的相对时间**；
5. 某 kernel 不支持某 case → 直接标注 **"不支持"**。

## 2. 目录结构与产物

```
unified/
├── README.md               # 本文件（脚本使用文档）
├── ANALYSIS.md             # 测试结果分析（独立文档）
├── gen_cases.py            # 重新生成 cases_meta.json（106 个 case，宿主机 CPU 可跑）
├── cases_meta.json         # （生成物）case_id → {B,T,H,K,V,group,desc}
├── bench.py                # 主脚本: 模式 A 正确性 + 模式 B msprof 分段采集
├── correctness.csv         # （生成物，模式 A）每 case 每 kernel 的 max_diff / status
├── profile_meta.json       # （生成物，模式 B）每段 (case,kernel) 元数据
├── per_case_profile.py     # 解析 msprof op_summary: marker 切分 → results.csv
├── results.csv             # （生成物，模式 B）case×kernel 的 torch_us / triton_us / speedup
├── analyze_results.py      # 解析 results.csv: 加速比矩阵 + 每 kernel 汇总
├── run_cpu.sh              # 容器启动器（source CANN + LD_LIBRARY_PATH + AUTOLOAD=0）
├── run_all_msprof.sh       # 全 case 分批 msprof 采集
└── prof*/                  # （生成物）msprof 输出目录
```

6 个算子的**唯一实现**在各自目录 `src/<op>_kernel.py`：

```
design/
├── gate_chunk_cumsum/  token_parallel/  inter_solve/
├── recompute_w_u/      delta_rule_h/    gla_output/
│   └── src/<op>_kernel.py      # ★ 该算子最终最优 triton kernel（含 torch 参考实现）
│       ├── *_torch / *_triton   # torch_npu 参考（精度基准） / triton 被测
│       ├── run.py               # 单算子精度&性能对比脚本
│       ├── run_cpu.sh           # 单算子容器内运行入口（含 msprof 模式）
│       └── …（DESIGN.md / README.md / OPTIMIZATION_LOG.md 等）
└── unified/                     # 本框架
```

bench.py 把 6 个 op 的 `src/` 加进 `sys.path` 后 import。

## 3. 用例构造（gen_cases.py）

统一固定 `BT=64, BC=16, scale=RCP_LN2`；`K=V` 取公共值 64 为主（6 个算子全支持）。
case 表为 **106 个**（A 72 + B 20 + C 8 + D 6，T 覆盖 1023–131072）：

| 组 | B | H | T | K=V | 数量 | 说明 |
|----|---|---|---|-----|------|------|
| A. T 主扫描（2 的幂） | 1,2,4 | 2,4,8 | 1024,2048,…,131072（8 档） | 64 | 72 | 全支持 |
| B. T 非 2 幂 / 边界 | 1 | 8 | 1023,1025,2047,…,131071,1536,6144,24576,98304,100000 | 64 | 20 | 尾 chunk 不满、非对齐 |
| C. 多 batch 放大 | 8 | 4,8 | 4096,8192,16384,32768 | 64 | 8 | 大 batch/head |
| D. K=V 扩展 + 目标 case | 1 | 2,8,4,96 | K=V=128 {1024,16384}×H∈{2,8} + K=V=32 {4096,H=4} + K=V=128 H=96 T=16384 | 128/32/128 | 6 | K=V=32 大 T 为不支持演示；末项为目标 case |

> 注：`cases_meta.json` 保存 dict 顺序（A→B→C→D），目标 case
> `D_KV128_H96_T16384` 在**索引 105**（末位），`--start 105 --limit 1` 可单独采它。

`gen_cases.py` **只写 case 表（元信息），不写张量数据**。张量由 `bench.py` 按
固定种子即时生成（`bench.py::_gen_case_inputs`），避免 T=131072/B=8/H=8 的多 GB
级 CSV 文件。

## 4. 数据流（每个 case，先 torch 链算共享中间输入，再逐 kernel A/B）

```
g       = K1_torch(x, A_log, dt_bias)                         # [B,T,H,K]
Aqk_d, Akk  = K2_torch(q, k, g, beta, scale)                  # [B,T,H,BT],[B,T,H,BC]
Aqk_nd, Akk_inv = K3_torch(q, k, g, beta, Akkd=Akk, scale)    # [B,T,H,BT],[B,T,H,BT]
w, u, kg = K4_torch(k, v, beta, A=Akk_inv, gk=g)              # 各 [B,T,H,K]
h, v_new = K5_torch(kg, w, u, gk=g, initial_state=init, idx)
Aqk_merge = Aqk_d + Aqk_nd
o       = K6_torch(q, v_new, g, Aqk=Aqk_merge, h=h, scale)
```

对 **K1..K6 各自**用**完全相同的输入**（即上面 torch 链算出的中间量）分别跑
`K_torch` 与 `K_triton`，得到该 kernel 的 max_diff 与 msprof 内核时间。A/B 输入
一致、可公平比较；torch 链只承担"正确输入的派生"。

## 5. 运行前提

所有验证须在**容器内**执行（宿主 triton 无 ascend driver，直接跑会报
`0 active drivers`）。`run_cpu.sh` 已内置 CANN `set_env.sh` source 与
`LD_LIBRARY_PATH` / `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 设置：

```bash
# 从宿主机调用（目录用绝对路径）
docker exec -w /docker/zhm/0505_skill_test/sonnet/sglang/kda_test/design/unified \
    triton-ascend-env-zhm bash run_cpu.sh ...
```

用例表 `cases_meta.json` 不存在时 `run_cpu.sh` 自动调 `gen_cases.py` 生成
（宿主 CPU torch 即可，无需 NPU）。

## 6. 两种运行模式

性能口径**只有一种**：msprof 采集的**设备侧 kernel 执行时间**（`op_summary.csv`
的 `Task Duration(us)`），与各算子 README 的 msprof 级测试一致。**不做
wall-clock 计时**。

### 6.1 模式 A — 正确性（默认，无计时）

```bash
bash run_cpu.sh                    # 全量 106 case 正确性（模式 A）
bash run_cpu.sh --limit 3          # 只跑前 3 个 case（冒烟）
bash run_cpu.sh --group A          # 只跑 A 组
bash run_cpu.sh --group D          # 只跑 D 组（含目标 case）
bash run_cpu.sh --start 105 --limit 1   # 只跑目标 case
```

逐 case 派生输入 → 跑各 kernel 的 torch/triton → 比对 max_diff → 写
`correctness.csv`（每 case 每 kernel 一行）：

```csv
case_id,B,T,H,K,V,kernel,max_diff,status
D_KV128_H96_T16384,...,K1,1.144409e-05,OK
```

`status` ∈ {`OK`, `不支持`, `FAIL`}；`max_diff` 为 fp32 绝对误差最大值
（`不支持` 时填原因，`FAIL` 时填 `RUN_ERR: ...`）。

**判定**：`status=OK` 且 `max_diff < 1e-2`（默认阈值，`--max-diff` 可改）。
`不支持` 表示该 case 超出该 kernel 支持范围，属预期、不算失败。

### 6.2 模式 B — msprof 性能采集（计时唯一来源）

```bash
bash run_cpu.sh --msprof ./prof_target --start 105 --limit 1 --repeats 5 --warmup 2
python3 per_case_profile.py --latest-dir ./prof_target --mean
python3 analyze_results.py
```

- 省略 `--repeats/--warmup` 时 `run_cpu.sh` 默认 `--repeats 5 --warmup 2`；
- msprof 包裹 `bench.py --msprof`：逐 case 派生输入（这些 setup 的 aclnn 算子会
  出现在 trace 开头），对每个 kernel（支持时）在 **marker kernel 分界**内先跑
  `K_torch` N 次、再跑 `K_triton` N 次（不做 wall-clock 计时）；不支持 kernel
  直接跳过（trace 中不出现，不污染统计）；写出 `profile_meta.json`；
- `per_case_profile.py` 读 `prof_target/` 下 `op_summary_*.csv` +
  `profile_meta.json`，写 **results.csv**（§7 schema）；`--mean` 取每次 repeat 平均。

全量 106 case 可选 `bash run_cpu.sh --msprof ./prof_full --repeats 5 --warmup 2`，
或分批断点续跑 `bash run_all_msprof.sh`。

### 6.3 msprof 分段：marker kernel 精确切分

用单个恒等 triton 小 kernel `_kda_bench_marker`（1 元素 store，微秒级）做分界：
每个 (case, kernel) 的 timed region 前后各发一个 marker。

`per_case_profile.py` 按 op_summary **行序**找 `_kda_bench_marker` 行把 trace 切成
精确的 (case, kernel) 段：

- 段内以该 kernel 的 triton op 名（`_gate_cumsum_kernel` / `_token_parallel_kernel` /
  `_inter_solve_kernel` / `_recompute_w_u_kernel` / `_delta_rule_h_kernel` /
  `chunk_gla_fwd_kernel_o`）**开头**的行 = triton 时间；其余 = torch_npu 拼接时间；
- marker 行本身不计入；
- 首个 marker 之前的行（setup 派生 + 进程启动）归入虚拟 `setup` 组，不计入任何 kernel；
- `--repeats N` 只影响段内行数，不影响分段正确性（marker 精确切分，无需按次数猜界）；
- 对"部分 kernel 不支持"的 case（不支持 kernel 无 marker，自然不成段）完全健壮。

## 7. 结果 CSV（results.csv，msprof 解析产物）

`schema`: `case_id,B,T,H,K,V,kernel,torch_us,triton_us,speedup,max_diff,status`

- `torch_us` = 该 kernel 的 torch_npu 拼接总时长（op_summary Task Duration 求和），
  `triton_us` = 该 kernel 的 triton N 次调用总和，`speedup = torch_us / triton_us`；
- `status` ∈ {`OK`, `不支持`}；不支持时 torch/triton 列填 `-`、max_diff 填原因；
- 由此 CSV 直接得到需求 (a)：按 kernel pivot 出 case×kernel 加速比矩阵；
  (b)：同一 case 下比较 6 行 `torch_us`（或 `triton_us`）得 6 kernel 相对时间。

## 8. 分析输出（analyze_results.py）

```bash
python3 analyze_results.py                           # 默认读 results.csv
python3 analyze_results.py --pivot-case A_B1_H8_T16384
```

1. **加速比矩阵** — 行=case（按 T/H/B 排序），列=K1..K6，格=`speedup`（`-`=不支持）；
2. **每 kernel 汇总** — 支持/不支持 case 数、平均/中位/最小/最大 speedup、随 T 变化趋势；
3. **6 kernel 相对时间** — 代表 case（默认取 T∈{1024,4096,16384,65536} 各一个）下
   6 个 kernel 的 `torch_us` / `triton_us` 排序与占比表。

## 9. 已知约束（`bench.py::_kernel_supports()`）

bench.py 在调用 triton kernel 前检查 case shape，不支持的直接标注 `不支持` 并跳过
（不污染 msprof trace）：

| kernel | 约束 | 原因 |
|--------|------|------|
| 全部 | 展平 grid 数 ≤ 65535 | triton-ascend 把 3D/2D grid 展平为 1D，超过 NPU coreDim 上限会报 `ERR00100 "value 65536 for parameter coreDim is invalid"` 且可能污染后续 NPU 状态。head-merge（HM=16，H%16==0 时启用）已把 K2/K3/K4 在目标 case（H=96）的 grid 从 24576 降到 1536；H%16!=0 时 HM=1 无此收益（如 A_B4_H8_T131072 展平 65536，仍属不支持） |
| K5 | K=V 且 K≤256 | delta_rule_h 已支持 K=128（BV=V；K=64 走 flat 1D store，K≠64 走 2D store），上游要求 K==V |
| K2/K3/K4 | K≠32 | `BK=next_power_of_2(32)=32` 时 `tl.dot` 数值不稳定 |
| K1/K6 | K=32 时仅 T≤256 | K=32 大 T 下精度不足（max_diff > 1e-2） |

## 10. 已知环境注意事项

本机 Ascend 设备在**持续多 case 负载**下会出现瞬时驱动故障，表现为：

```
[ERROR] ... ERR00100 PTA call acl api failed.
terminate called after throwing an instance of 'std::runtime_error'
  what(): ... current working operator name is aclnnMul.
```

- **现象**：`npu-smi info` 显示全部芯片 `Health=OK`、简单单算子（如一次 matmul）
  可正常跑，但连续跑数十个 case（模式 A 全量 / 部分 B 组大 T case）时随机在某
  个 `aclnnMul` 处崩溃（`runtime 507015`，进程 exit 134）。属**环境/驱动瞬时故障**，
  与 kernel 代码无关（崩溃点随机，复跑同一 case 可通过）。
- **规避**：
  1. 用 `--group D` 或 `--start 105 --limit 1` 只跑小批量/目标 case，避免长时间负载。
  2. 崩溃后**重启容器**恢复设备态：`docker restart triton-ascend-env-zhm`，再重跑
     失败批次的 case（`bench.py --start <s> --limit 10`）。
  3. 全量 106 case 建议**分批**跑（每次 `--start <s> --limit 10`），逐批把
     `correctness.csv` 另存，失败批次重试。
- **性能影响**：msprof 目标 case（单 case 采集）不受影响，官方数值可稳定复现。

## 11. 其他注意事项

- **必须容器内执行**: `bash run_cpu.sh` 需在容器 `triton-ascend-env-zhm` 内运行
  （宿主 triton 无 ascend driver）。
- **性能口径只有 msprof**: 不做 wall-clock 计时；唯一性能来源是 `op_summary.csv`
  的 `Task Duration(us)`（设备侧 kernel 时间）。
- **K5 in-place**: delta_rule_h 的 triton/torch 版都会 in-place 更新
  `initial_state`；bench.py 为 torch/triton 各保留独立的 init 张量，每次调用前
  用 `init_backup` 复位，避免跨实现污染。
- **NPU arange 陷阱**: `torch.arange(B, dtype=torch.int32, device="npu")` 在
  多种 B 下产生垃圾值（如 `4547155312683678328`）；bench.py 在 CPU 创建后
  `.to(device)` 规避。
- **大 T 内存**: T=131072/B=8/H=8/K=64 的单张 q/k/v/g 约 4GB（fp32）；
  bench.py 及时释放中间量，910B2 内存足够。

## 12. 设计要点与实现 checklist（合入自统一测试方案）

框架按以下步骤实现（历史记录，已完成）：

1. 复制 6 个算子目录到 `kda_test/design/`（排除 `__pycache__/.git/prof*`），核对每个
   `src/*.py` 除 torch/triton 外无本地 import（有则把该 op 目录加进 sys.path）。
2. `gen_cases.py`：生成 106 个 case → `cases_meta.json`（宿主 CPU 跑），校验行数与 T 覆盖。
3. `bench.py`：模式 A 正确性 + 模式 B msprof（marker 分段采集、`profile_meta.json`）；
   数据流见 §4；"不支持"捕获；K5 每次调用前 `initial_state` 重新 `clone`。
4. `run_cpu.sh`：容器启动器（source CANN + LD_LIBRARY_PATH + AUTOLOAD=0 + `--msprof`）。
5. 容器内正确性冒烟：`docker exec -w ... triton-ascend-env-zhm bash run_cpu.sh --limit 3`。
6. 全量正确性（模式 A）→ 校验 `correctness.csv`（100 个 K=V=64 全 OK、D 组按约束标注）。
7. `analyze_results.py`：加速比矩阵 + 相对时间表 + 不支持统计。
8. msprof 全量性能采集（模式 B）→ `per_case_profile.py` 产出 results.csv，核对
   加速比量级与各算子 README 一致，并核对 marker 分段无错位。
9. 分析验证文档（本目录 `README.md` + `ANALYSIS.md`）。
10. 自检：复制完整性、100+ case、T∈[1k,128k]、性能全部来自 msprof、CSV 可 pivot
    出 (a)(b)、不支持标注生效、msprof 流程闭环。

## 13. 风险与注意

- **运行时间**：性能走 msprof 全量 106 case（模式 B），T=128k 的 case（尤其 K5 串行
  NT=2048、K6）单次调用几十 ms，× 6 kernel × 2 impl × N(repeats，建议 3~5) × 106 case，
  预计十几分钟到半小时；正确性模式（模式 A）只跑一遍、快很多，可先行验证。
  用 `--limit/--group/--start` 可做子集采集。
- **内存**：T=131072、B=8、H=8、K=V=64 时单张 q/k/v/g 约 4GB（fp32），910B2 内存足够，
  但避免同时保留过多中间量（bench.py 及时释放/复用）。
- **msprof 分段**：marker kernel 精确定界（§6.3），对"部分 kernel 不支持"的 case 也健壮；
  `profile_meta.json` 兜底交叉校验。
- **环境**：所有 NPU 运行必须在容器内（宿主 triton 无 ascend driver）；持续负载下
  设备可能瞬时故障（§10），用分批/重启规避。
