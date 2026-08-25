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

## 5. 与 H100 横向对比：910B2 理论上限（2026-08-25 评估）

同事在 **H100 上跑同 6 个算子**（同 Triton kernel、fp32、同一目标 case）总时间
**6.9ms**；本机 910B2 为 46.1ms → 当前仅 **0.15x**（约慢 6.7x）。

### 5.1 算法本质是内存带宽受限

H100 的内存带宽下限 = 20.3GB / 3.35TB/s ≈ **6.0ms**，6.9ms ≈ **87.6% 峰值带宽**。
这套算子在 H100 上已贴近带宽下限，故 910B2 的上限由**带宽比**而非算力比决定。

目标 case 各 kernel 的 fp32 内存总流量（合计 ≈20.3GB）：

| kernel | 读（MB） | 写（MB） | 合计（MB） |
|--------|---------|---------|-----------|
| K1 | 805(x) | 805(g) | 1,611 |
| K2 | 2,422(q,k,g,β) | 503(Aqk_d+Akk) | 2,926 |
| K3 | 2,422(q,k,g,β) | 805(Aqk_nd+Akk_inv) | 3,228 |
| K4 | 1,611(k,v) | 2,416(w,u,kg) | 4,027 |
| K5 | 2,416(kg,w,u) | 1,611(h,v_new) | 4,027 |
| K6 | 3,624(q,v_new,g,Aqk,h) | 805(o) | 4,429 |

### 5.2 硬件规格对比

| 项 | H100 SXM5 | 910B2（本机） |
|----|-----------|---------------|
| 内存带宽 | 3.35 TB/s（HBM3） | 口径不一：多来源 ~400GB/s（HBM2e）、部分 1.6 TB/s（HBM3e） |
| 内存 | 80 GB | 64 GB（本机 npu-smi 确认） |
| FP16 | 989 TFLOPS | 376 TFLOPS |
| FP32 | 67 TFLOPS | 官方未给出 |

本机实测可**排除 ~400GB/s 档**：K1（最纯访存型）1.61GB / 1.82ms ≈ **884 GB/s**，
不可能出现在 400GB/s 的设备上。此 910B2 有效带宽至少 ~0.9 TB/s，峰值落在
1.2–1.6 TB/s 区间。

### 5.3 三个口径的上限

| 口径 | 910B2 总时间下限 | 相对 H100(6.9ms) 倍率 | 相对当前(46.1ms) 优化空间 |
|------|------------------|----------------------|--------------------------|
| 当前 Triton 实现 | 46.1ms | **0.15x**（慢 6.7x） | — |
| 本机实测 fp32 有效带宽 ~0.87TB/s（K1 884GB/s 为最佳） | ≈23ms | **≈0.3x** | ≈2x |
| 峰值 1.2 TB/s（HBM2e 口径） | ≈17ms | ≈0.4x | ≈2.7x |
| 峰值 1.6 TB/s（HBM3e 口径，理论极限） | ≈13ms | **≈0.5x** | ≈3.6x |

当前 46.1ms 对应平均带宽利用率仅 **27%**（H100 为 88%）；Triton-Ascend 的代码生成
成熟度是当前差距的主要来源，其次是 910B2 峰值带宽口径本身低于 H100。

### 5.4 结论与前提

- **结论**：910B2 对 H100 的理论上限 ≈ **0.5x**（峰值带宽口径）、现实可达 ≈ **0.3x**、
  当前仅 **0.15x**。优化空间主要来自 kernel 带宽利用率与带宽峰值口径，**不是算力代差**。
- **前提/限制**：
  1. H100 的 6.9ms 已 ≈ 其带宽下限，910B2 物理上无法追平（带宽比 ~1.6/3.35 ≈ 0.48x）。
  2. K3 目前是**计算/发射受限**（14.4ms vs 带宽下限 ~4ms），要压到带宽上限需进一步
     削计算（NP/dot 数）或改用 fp16/tf32 中间精度 —— 后者会改变现有 fp32 精度口径。
  3. K5 串行依赖限制并行度，未必压得到带宽下限。
  4. 若同事的 H100 数据用 fp16/tf32 张量核，口径需另算；但 6.9ms 与 fp32 带宽下限
     一致，按同口径理解合理。

> 规格来源：[H100 SXM5 带宽/FP32 3.35TB/s·67TFLOPS](https://vercel.hyper.ai/en/gpu-leaderboard/nvidia-h100-sxm5-80-gb)、
> [NVIDIA DGX / H100 (Wikipedia)](https://en.m.wikipedia.org/wiki/Nvidia_DGX-1)、
> [昇腾910B vs A100/H100 对比](https://hwcomputing.csdn.net/6a4b7009662f9a54cb8a4933.html)、
> [910B1/B2/B3/B4 选型（带宽口径差异）](https://ucache.cn/enterprise/new/318.html)。
> 910B2 峰值带宽口径不一（~400GB/s HBM2e / 1.6TB/s HBM3e）；本机 64GB、实测 K1
> 884GB/s，按 1.2–1.6 TB/s 档理解。

## 6. 各算子最终配置一览（已固化在 `src/*_kernel.py` 中）

| Op | 文件 | 关键配置 | 优化要点 |
|----|------|----------|----------|
| K1 | `gate_chunk_cumsum/src/gate_kernel.py` | BS=128, 2D grid | cumsum + logsumexp 融合 |
| K2 | `token_parallel/src/token_parallel_kernel.py` | HM=16, `tl.gather` 紧凑 Akk | head-merge + kernel 内收拢对角块 |
| K3 | `inter_solve/src/inter_solve_kernel.py` | HM=16, NP=3, fp32 | 融合单 kernel + 重复平方截断逆 |
| K4 | `recompute_w_u/src/recompute_w_u_kernel.py` | HM=16 | head-merge（6796→4514us） |
| K5 | `delta_rule_h/src/delta_rule_h_kernel.py` | BV=V=128, 2 dot/chunk | 单 K-tile 合并串行 dot 链 |
| K6 | `gla_output/src/gla_output_kernel.py` | BK=128, BV=128, nw=2 | 跨块 K 循环合并为单大 dot |

## 7. 结论汇总

- **6 算子全部落地**：K1..K6 的唯一最优 triton kernel 固化于各 `src/<op>_kernel.py`，
  统一框架可一键跑正确性 + msprof 性能。
- **精度**：目标 case 6/6 OK，K4/K5 与 torch_npu bitwise 一致。
- **性能**：目标 case 总 triton ≈46.1ms（官方 ≈46.7ms，偏差 <10%）；总加速比
  （torch_npu 拼接总和 ≈1.26s → triton 46.1ms）约 **27x**，其中 K5 高达 95x、K2 20x。
- **约束/标注**：K=V=32 大 T 等 case 按 README §9 约束正确标注 `不支持`（非 bug）。
- **横向对比**：910B2 相对 H100 当前 0.15x、现实可达 ~0.3x、理论上限 ~0.5x
  （详见 §5，内存带宽受限）。

## 8. 已知问题与待办

- **K5 同事版曾报 6.4ms**（BT=128，破坏 K2..K6 共享的 BT=64 chunk 契约，无效）；
  **K6 同事版曾报 4.52ms**（fp32 下不可能，已 7 组实验证伪，真实 ~7ms）。均
  **未采纳**；证伪结论见各 `OPTIMIZATION_LOG.md`。
- **全量 106 case 性能采集**：目标 case 已复现官方；全 106 case 的 results.csv
  尚未完整落盘（设备持续负载不稳定，README §10），需分批续跑
  `bash run_all_msprof.sh` 后 `per_case_profile.py` 聚合。
- **A/B/C 组全量复核**：历史已过；如需留档复核，按 README §10 分批跑模式 A。
