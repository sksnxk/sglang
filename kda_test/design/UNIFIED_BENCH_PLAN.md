# 6 算子统一测试方案与 Checklist（待确认）

日期：2026-08-15 ｜ 目标目录：`kda_test/design/` ｜ 容器：`triton-ascend-env-zhm`（Ascend910B2）

## 0. 需求回顾

把 KDA 的 6 个独立验证算子（gate_chunk_cumsum、token_parallel、inter_solve、
recompute_w_u、delta_rule_h、gla_output）复制到 `kda_test/design/` 下，并写一个
**统一测试脚本**，满足：

1. 构造 **100+ 测试用例**（序列长度 T 覆盖 1k–128k）；
2. 每个 case **按顺序**调用 6 个算子的 **torch_npu 版本**和 **triton 版本**；
3. 支持 **msprof 包裹**测内核性能；
4. 输出 **CSV**，可分析：
   - (a) 每个 kernel 在**每个 case** 下 triton 相对 torch_npu 的**加速比**；
   - (b) 同一输入 case 下 **6 个 kernel 的相对时间**；
5. 某 kernel 不支持某 case → 直接标注 **"不支持"**。

## 1. 已验证的环境（完成）

| 项 | 结果 |
|----|------|
| 宿主机 python3 | torch 2.10.0+cpu；`gen_csv.py` 可正常生成用例 |
| 容器 `triton-ascend-env-zhm` | torch 2.7.1+cpu、torch_npu 2.7.1.post4、triton 3.2.0、NPU=Ascend910B2 |
| 单算子冒烟 | `docker exec -w <op> triton-ascend-env-zhm bash run_cpu.sh` → gate_chunk_cumsum **15/15 PASS** |
| msprof | 容器内有 msprof；`run_cpu.sh --msprof ./prof --no-ref` 成功，`analyze_profile.py` 报 **21.31x**、`per_case_profile.py` 报 **15.60x**（与 README 量级一致） |
| ⚠️ 关键结论 | `run_cpu.sh` 必须在**容器内**执行（`docker exec`）。直接在宿主机跑会因宿主 triton 无 ascend driver 报 `0 active drivers` 错误 |

## 2. 目录规划（复制后）

```
kda_test/design/
├── gate_chunk_cumsum/   ← 整目录复制
├── token_parallel/      ← 整目录复制
├── inter_solve/         ← 整目录复制
├── recompute_w_u/       ← 整目录复制
├── delta_rule_h/        ← 整目录复制
├── gla_output/          ← 整目录复制
├── UNIFIED_BENCH_PLAN.md   ← 本文件
└── unified/                    ← 统一测试（新增）
    ├── README.md               ← 用法说明
    ├── gen_cases.py            ← 生成 ~105 个用例 + cases_meta.json
    ├── cases.csv               ←（生成物）base64 编码的输入张量
    ├── cases_meta.json         ←（生成物）id→{B,T,H,K,V} 元信息
    ├── bench.py                ← 正确性校验（默认）＋ msprof 模式（marker 分段采集）
    ├── correctness.csv         ←（生成物）每 case 每 kernel 精度/支持性
    ├── run_cpu.sh              ← 容器启动器（复用 per-op 约定 + --msprof 支持）
    ├── analyze_results.py      ← 解析 results.csv：加速比矩阵 / 6 kernel 相对时间 / 不支持统计
    ├── analyze_profile.py      ← msprof 全量聚合（torch 拼接 vs triton 总时长）
    ├── per_case_profile.py     ← msprof 逐 case/逐 kernel 分段（marker 分界），产出 results.csv
    └── results.csv             ←（生成物）最终：加速比 + 相对时间 + 不支持标注
```

> 复制时排除 `__pycache__`、`.git`、临时 `prof*` 目录。

## 3. 用例构造（gen_cases.py）

统一固定 `BT=64, BC=16, scale=RCP_LN2`，`K=V` 取公共值 64 为主（6 个算子全支持）。

| 组 | B | H | T | 数量 | 说明 |
|----|---|---|---|------|------|
| A. T 主扫描（2 的幂） | 1,2,4 | 2,4,8 | 1024,2048,…,131072（8 档） | 8×3×3=**72** | 全支持 |
| B. T 非 2 幂 / 边界 | 1 | 8 | 1023,1025,2047,…,131071,1536,6144,24576,98304,100000（20 档） | **20** | 覆盖尾 chunk 不满、非对齐 |
| C. 多 batch 放大 | 8 | 4,8 | 4096,8192,16384,32768 | 4×2=**8** | 大 batch/head |
| D. 不支持演示 | 1 | 2,8 | K=V=128: {1024,16384}（4）+ K=V=32: {4096,H=4}（1） | **5** | K5(delta_rule_h) 不支持→标注 |

合计 **105 个 case**，T 覆盖 1k–128k，其中 100 个 K=V=64 全支持、5 个用于演示"不支持"。

每个 case 生成共享基础张量（base64 存 `cases.csv`）：
`x[A_log 门控输入], A_log[H], dt_bias[H*K], q,k,v [B,T,H,K], beta[B,T,H]`。

**K5 约束**：delta_rule_h 的 triton kernel 只支持 `K=V=64`（上游 flat-1D store 假定行步长
K=64，见其 README）。因此 D 组 K=128/32 的 case 中 K5 直接标注 `不支持`；
torch_npu 版仍照跑（K5_torch 是串行循环，任意 K 可用）。

## 4. 执行策略：性能统一用 msprof（内核侧时间），不做 wall-clock 计时

> 性能口径只有一种：**msprof 采集的设备侧 kernel 执行时间**（op_summary.csv 的
> Task Duration(us)），与每个算子 README 的 msprof 级测试一致。`bench.py`
> **不输出 wall-clock 计时**（默认只做正确性校验；`--timing` 仅作调试可选项）。

### 4.1 数据流（每个 case，先 torch_npu 链算共享中间输入，再逐 kernel A/B）

```
g       = K1_torch(x, A_log, dt_bias)                         # [B,T,H,K]
Aqk_d, Akk  = K2_torch(q, k, g, beta, scale)                  # [B,T,H,BT],[B,T,H,BC]
Aqk_nd, Akk_inv = K3_torch(q, k, g, beta, Akkd=Akk, scale)    # [B,T,H,BT],[B,T,H,BT]
w, u, kg = K4_torch(k, v, beta, A=Akk_inv, gk=g)              # 各 [B,T,H,K]
h, v_new = K5_torch(kg, w, u, gk=g, initial_state=init.clone(), idx)
Aqk_merge = Aqk_d + Aqk_nd
o       = K6_torch(q, v_new, g, Aqk=Aqk_merge, h=h, scale)
```

对 **K1..K6 各自**用**完全相同的输入**（即上面 torch 链算出的中间量）分别跑
`K_torch` 与 `K_triton`，得到该 kernel 的 max_diff（精度）。A/B 输入一致、可公平比较；
torch 链只承担"正确输入的派生"。

### 4.2 两种运行模式

- **模式 A — 正确性（默认，无计时）**：`python3 bench.py [cases.csv]`。
  逐 case 派生输入 → 跑各 kernel 的 torch/triton → 比对 max_diff → 写 `correctness.csv`
  （每 case 每 kernel 一行：max_diff、status=OK/不支持/FAIL、不支持原因）。全流程可在
  msprof 之外单独跑，速度快、内存小，专门验证"哪些 kernel 支持哪些 case"。
- **模式 B — msprof 采集（计时唯一来源）**：`run_cpu.sh --msprof ./prof [--repeats N]`。
  msprof 包裹 `bench.py --msprof --repeats N`。此时 `bench.py`：
  - 逐 case 派生输入（这些 setup 的 aclnn 算子会出现在 trace 开头）；
  - 对每个 kernel（支持时）在 **marker kernel 分界** 内先跑 `K_torch` N 次、再跑
    `K_triton` N 次（**不做 wall-clock 计时**）；
  - 不支持 kernel 直接跳过（trace 中不出现 → 天然不污染统计）；
  - 写出 `profile_meta.json`（每 case 每 kernel：triton kernel 名、N、是否跳过）。

### 4.3 msprof 分段：marker kernel 精确切分

- 用单个恒等 triton 小 kernel `_kda_bench_marker`（1 元素 store，微秒级）做分界：
  每个 (case, kernel) 的 timed region 前后各发一个 marker；
- `per_case_profile.py` 按 op_summary **行序**找 `_kda_bench_marker` 行把 trace 切成
  精确的 (case, kernel) 段：段内以该 kernel 的 triton 名
  （`_gate_cumsum_kernel`/`_token_parallel_kernel`/`_inter_solve_kernel`/
  `_recompute_w_u_kernel`/`_delta_rule_h_kernel`/`chunk_gla_fwd_kernel_o`）开头的行 =
  triton 时间，其余 = 该 kernel 的 torch_npu 拼接时间；marker 行本身不计入；
- 首个 marker 之前的行（setup 派生 + 进程启动）归入虚拟 `setup` 组，不计入任何 kernel；
- `--repeats N` 只影响段内行数，不影响分段正确性（marker 精确，无需按次数猜界）；
- 对比 per-op 目录的"按 triton 出现次数猜界"方案，marker 方案对 **不支持 kernel 混合**
  的 case 也完全健壮。

### 4.4 结果 CSV（results.csv，msprof 解析产物）

`per_case_profile.py` 合并 `op_summary` 分段结果 + `correctness.csv`，输出最终 **results.csv**：

```
case_id, B, T, H, K, V, kernel, torch_us, triton_us, speedup, max_diff, status
```
- `torch_us` = 该 kernel 的 torch_npu 拼接总时长（op_summary Task Duration 求和），
  `triton_us` = 该 kernel 的 triton N 次调用总和，`speedup = torch_us/triton_us`；
- `status` ∈ {`OK`, `不支持`}；不支持时 torch/triton 列填 `-`、max_diff 填原因；
- 由此 CSV 直接得到分析 (a)：按 kernel pivot 出 case×kernel 加速比矩阵；
  (b)：同一 case 下比较 6 行 torch_us（或 triton_us）得 6 kernel 相对时间；
- `analyze_profile.py`：全量聚合（torch 拼接 vs triton 总时长，跨 case）。

### 4.5 分析输出（analyze_results.py）

读 results.csv 输出：
1. **加速比矩阵**：行=case，列=K1..K6，格=triton/torch speedup（`-`=不支持）；
2. **每 kernel 汇总**：支持 case 数、不支持 case 数、平均/中位加速比、随 T 变化趋势；
3. **6 kernel 相对时间**：指定 case（默认取各 T 档一个代表）下 6 个 kernel 的
   torch_us 与 triton_us 排序/占比表。

## 5. 实现 Checklist（待你确认后执行）

1. **复制** 6 个算子目录到 `kda_test/design/`（排除 `__pycache__/.git/prof*`），并核对
   每个 `src/*.py` 除 torch/triton 外无本地 import（若有则把该 op 目录加进 sys.path）。
2. **`unified/gen_cases.py`**：按 §3 生成 105 个 case → `cases.csv` + `cases_meta.json`；
   宿主机跑（CPU torch），并校验行数与 T 覆盖。
3. **`unified/bench.py`**：模式 A 正确性（§4.2）+ 模式 B msprof（marker 分段采集、
   `profile_meta.json`）；§4.1 数据流；"不支持"捕获；K5 每次调用前 `initial_state`
   重新 `clone`。
4. **`unified/run_cpu.sh`**：容器启动器（source CANN + LD_LIBRARY_PATH + AUTOLOAD=0 + `--msprof`）。
5. **容器内正确性冒烟**：`docker exec -w .../unified triton-ascend-env-zhm bash run_cpu.sh --limit 3`
   确认 3 个小 case 全过、`correctness.csv` 格式正确、D 组 K5 正确标注 `不支持`。
6. **全量正确性**：跑全部 105 个 case（模式 A）→ 校验 `correctness.csv`：100 个
   K=V=64 case 全部 OK、5 个 D 组 case 的 K5 为 `不支持` 且其余 kernel OK。
7. **`unified/analyze_results.py`**：出加速比矩阵 + 相对时间表 + 不支持统计。
8. **msprof 全量性能采集**：`bash run_cpu.sh --msprof ./prof --no-ref`（全 105 个 case）
   → `per_case_profile.py --latest-dir ./prof` 产出 **results.csv**（含每个 (case,kernel)
   的 torch_us/triton_us/speedup）；校验加速比量级与 per-op 目录实测一致
   （K6 约 20x、K2 约 24x、K1 约 1.1–1.5x 量级），并核对 marker 分段无错位。
9. **写 `unified/README.md`**：用法、CSV 字段、msprof 分段（marker）说明、注意事项。
10. **自检清单**：复制完整性（6 op 目录结构 vs 原目录）、100+ case、T∈[1k,128k]、
    性能全部来自 msprof（无 wall-clock）、CSV 可 pivot 出 (a)(b)、不支持标注生效、
    msprof 流程闭环。

## 6. 风险与注意

- **运行时间**：性能走 msprof 全量 105 case（模式 B），T=128k 的 case（尤其 K5 串行
  NT=2048、K6）单次调用几十 ms，× 6 kernel × 2 impl × N(repeats，建议 3~5) × 105 case，
  预计十几分钟到半小时；正确性模式（模式 A）只跑一遍、快很多，可先行验证。
  用 `--limit/--tag` 可做子集采集。
- **内存**：T=131072、B=8、H=8、K=V=64 时单张 q/k/v/g 约 4GB（fp32），910B2 内存足够，
  但避免同时保留过多中间量（bench.py 及时释放/复用）。
- **msprof 分段**：marker kernel 精确定界（§4.3），对"部分 kernel 不支持"的 case 也健壮；
  `profile_meta.json` 兜底交叉校验。
- **K5 约束**：triton 只支持 K=V=64，D 组用例会演示 `不支持` 标注（非 bug）。
- **环境**：所有 NPU 运行必须在容器内（宿主 triton 无 ascend driver）。
