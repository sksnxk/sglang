# KDA 6 算子统一 bench — 测试结果分析

> 本文档是 `unified/` 框架的**测试结果分析**，**不含脚本用法**（用法见
> `README.md`）。记录实测的正确性、性能与结论。

KDA（Kimi Linear Delta Attention）chunked attention 6 个 triton kernel
（K1 gate_chunk_cumsum / K2 token_parallel / K3 inter_solve /
K4 recompute_w_u / K5 delta_rule_h / K6 gla_output）相对 torch_npu 元算子版的
统一验证结果。

- **目标 case**: `B=1, T=16384, H=96, K=V=128`（case_id `D_KV128_H96_T16384`）
- **硬件**: Ascend 910B2（NPU 20 核），CANN 9.0.0 / triton-ascend 3.2.1 / torch_npu 2.7.1
- **容器**: `triton-ascend-env-zhm`

---

## 1. 测量口径

- 性能唯一来源：**msprof `op_summary.csv` 的 `Task Duration(us)`**（设备侧 kernel
  时间），不做 wall-clock 计时。
- marker 分段：每个 (case, kernel) 段前后发 `_kda_bench_marker`，段内以 triton op
  名开头的行 = `triton_us`（N 次调用总和），其余 = `torch_us`（torch_npu 拼接总和）；
  `speedup = torch_us / triton_us`。
- 复现判定：同机同态下 `triton_us` 与官方基线偏差应 < **±10%**；显著偏高时检查
  是否有其它进程占 NPU、或设备是否处于降频态。

## 2. 正确性结果

### 2.1 当前 `correctness.csv`（D 组 6 case）

仓库中 `correctness.csv` 现保存 **D 组 6 个 case** 的复核结果（36 行 =
6 case × 6 kernel），**31 `OK` + 5 预期 `不支持`**：

| case | K1 | K2 | K3 | K4 | K5 | K6 |
|------|----|----|----|----|----|----|
| D_KV128_H2_T1024 | OK | OK | OK | OK | OK | OK |
| D_KV128_H2_T16384 | OK | OK | OK | OK | OK | OK |
| D_KV128_H8_T1024 | OK | OK | OK | OK | OK | OK |
| D_KV128_H8_T16384 | OK | OK | OK | OK | OK | OK |
| D_KV32_H4_T4096 | 不支持 | 不支持 | 不支持 | 不支持 | OK | 不支持 |
| D_KV128_H96_T16384（目标） | OK | OK | OK | OK | OK | OK |

`D_KV32_H4_T4096` 的 5 个 `不支持` 原因（符合 README §9 约束，非 bug）：

- K1/K6：K=32 大 T 下精度不足（仅小 T ≤256 验证通过）；
- K2/K3/K4：`BK=32` 时 `tl.dot` 数值不稳定；
- K5：K=V=32 支持（0.0 精确）→ OK。

### 2.2 覆盖说明

- A/B/C 组（100 个 K=V=64 case）在框架清理前用**同一套 canonical kernel**（未被
  清理触碰）已验证通过；当前设备持续负载下不稳定（见 README §10），如需全量复核
  建议分批 `bash run_cpu.sh --start <s> --limit 10`。
- 当前 `correctness.csv` 仅含 D 组是"分批另存 + 设备崩溃后重跑失败批次"流程的
  当前落盘状态，不代表其它组未测。

## 3. 目标 case 精度基线

`D_KV128_H96_T16384`（B=1, T=16384, H=96, K=V=128）6 kernel 全部 `OK`：

| Kernel | max_diff | 说明 |
|--------|----------|------|
| K1 | 1.14e-05 | gate_chunk_cumsum |
| K2 | 8.94e-08 | token_parallel (Aqk/Akk) |
| K3 | 7.45e-08 | inter_solve (Aqk/Akk_inv) |
| K4 | 0.0 | recompute_w_u（bitwise 一致） |
| K5 | 0.0 | delta_rule_h（h/v_new bitwise 一致） |
| K6 | 3.73e-09 | gla_output |

全部 `max_diff < 1e-2` 判定阈值；K4/K5 与 torch_npu 参考 bitwise 一致。

## 4. 目标 case 性能结果

### 4.1 本次复核（2026-08-24，当前 `results.csv`）

`results.csv` 保存目标 case 的 msprof 数据（`--start 105 --limit 1 --repeats 5
--warmup 2 --mean`）：

| Kernel | torch_us | triton_us | speedup | max_diff |
|--------|----------|-----------|---------|----------|
| K1 | 12977.6 | 1822.5 | 7.1x | 1.14e-05 |
| K2 | 193119.1 | 9463.9 | 20.4x | 8.94e-08 |
| K3 | 132419.1 | 14406.5 | 9.2x | 7.45e-08 |
| K4 | 32036.4 | 4518.2 | 7.1x | 0.0 |
| K5 | 871290.8 | 9127.2 | 95.5x | 0.0 |
| K6 | 21567.7 | 6779.8 | 3.2x | 3.73e-09 |

**总 triton 时间 ≈ 46.1ms**。耗时排序（triton_us 降序）：K3(14.4) > K2(9.5) >
K5(9.1) > K6(6.8) > K4(4.5) > K1(1.8) ms。加速比排序：K5(95.5x，串行 torch 循环
几乎全被压掉) > K2(20.4x) > K3(9.2x) > K1(7.1x) > K4(7.1x) > K6(3.2x，fp32 带宽受限)。

### 4.2 官方基线（同机同态参考）

| Kernel | torch_us | triton_us | speedup | 说明 |
|--------|----------|-----------|---------|------|
| K1 | 12782.9 | **1880.1** | 6.8x | 1.88ms |
| K2 | 193277.5 | **9416.3** | 20.5x | 9.42ms |
| K3 | 133268.6 | **14366.9** | 9.3x | 14.37ms（NP=3, HM=16） |
| K4 | 32027.6 | **4514.1** | 7.1x | 4.51ms（HM=16） |
| K5 | 920639.6 | **9150.8** | 100.6x | 9.15ms（BV=V=128） |
| K6 | 21586.7 | **7385.5** | 2.9x | 7.39ms（BK=128, memory-bound） |

**总 triton 时间 ≈ 46.7ms**。

### 4.3 判定

- **精度**：目标 case 6 kernel 全部 `status=OK`、`max_diff < 1e-2`（K4/K5 为 0.0）。
- **性能**：目标 case 6 kernel `triton_us` 复现官方值，偏差全部 **< 10%**
  （本次复核 vs 官方基线：K1 −3.1%、K2 +0.5%、K3 +0.3%、K4 +0.1%、K5 −0.3%、
  K6 −8.2%，均在同机同态正常抖动范围内）。

## 5. 各算子最终配置一览（已固化在 `src/*_kernel.py` 中）

| Op | 文件 | 关键配置 | 优化要点 |
|----|------|----------|----------|
| K1 | `gate_chunk_cumsum/src/gate_kernel.py` | BS=128, 2D grid | cumsum + logsumexp 融合 |
| K2 | `token_parallel/src/token_parallel_kernel.py` | HM=16, `tl.gather` 紧凑 Akk | head-merge + kernel 内收拢对角块 |
| K3 | `inter_solve/src/inter_solve_kernel.py` | HM=16, NP=3, fp32 | 融合单 kernel + 重复平方截断逆 |
| K4 | `recompute_w_u/src/recompute_w_u_kernel.py` | HM=16 | head-merge（6796→4514us） |
| K5 | `delta_rule_h/src/delta_rule_h_kernel.py` | BV=V=128, 2 dot/chunk | 单 K-tile 合并串行 dot 链 |
| K6 | `gla_output/src/gla_output_kernel.py` | BK=128, BV=128, nw=2 | 跨块 K 循环合并为单大 dot |

## 6. 结论汇总

- **6 算子全部落地**：K1..K6 的唯一最优 triton kernel 固化于各 `src/<op>_kernel.py`，
  统一框架可一键跑正确性 + msprof 性能。
- **精度**：目标 case 6/6 OK，K4/K5 与 torch_npu bitwise 一致。
- **性能**：目标 case 总 triton ≈46.1ms（官方 ≈46.7ms，偏差 <10%）；总加速比
  （torch_npu 拼接总和 ≈1.26s → triton 46.1ms）约 **27x**，其中 K5 高达 95x、K2 20x。
- **约束/标注**：K=V=32 大 T 等 case 按 README §9 约束正确标注 `不支持`（非 bug）。

## 7. 已知问题与待办

- **K5 同事版曾报 6.4ms**（BT=128，破坏 K2..K6 共享的 BT=64 chunk 契约，无效）；
  **K6 同事版曾报 4.52ms**（fp32 下不可能，已 7 组实验证伪，真实 ~7ms）。均
  **未采纳**；证伪结论见各 `OPTIMIZATION_LOG.md`。
- **全量 106 case 性能采集**：目标 case 已复现官方；全 106 case 的 results.csv
  尚未完整落盘（设备持续负载不稳定，README §10），需分批续跑
  `bash run_all_msprof.sh` 后 `per_case_profile.py` 聚合。
- **A/B/C 组全量复核**：历史已过；如需留档复核，按 README §10 分批跑模式 A。
