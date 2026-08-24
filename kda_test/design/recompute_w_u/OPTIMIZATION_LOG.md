# OPTIMIZATION_LOG — recompute_w_u (w = A@(k*beta*exp2(gk)), u = A@(v*beta), kg = k*exp2(gk_last-gk))

## 任务信息

| 字段 | 值 |
|------|-----|
| 算子名称 | recompute_w_u |
| 任务模式 | optimize-existing-kernel |
| 原始 kernel | `src/recompute_w_u_kernel.py` |
| 目标硬件 | Ascend 910B2 |
| 环境 | triton-ascend-env-zhm, triton 3.2.0, torch 2.7.1 |
| 完成时间 | 2026-08-18T16:00:00 |

---

## 优化路线总览

| 路线 | 策略 | 文件 | blk=4 dur (us) | blk=12 dur (us) | blk=41 dur (us) | Speedup (blk=12) | 精度 |
|------|------|------|---------------|----------------|----------------|-----------------|------|
| **Route A (WINNER)** | BK=K, BV=V 单 tile + T_FULL 分派 + num_warps=4 | `src/recompute_w_u_kernel_optA.py` | **9032** | **12030** | 14861 | **1.68x** | PASS |
| Route B | BK=64, BV=128 大 tile + T_FULL/K_FULL/V_FULL 分派 + num_warps=2 | `src/recompute_w_u_kernel_optB.py` | 9045 | 14228 | 15680 | 1.42x | PASS |
| Route C | BK=64, BV=64 + fp32 mask + num_warps=2 | `src/recompute_w_u_kernel_optC.py` | 9542 | 15931 | **14426** | 1.27x | PASS |

---

## 迭代记录

### 基线 (原始 kernel)

- Grid: (NT, B*H), BT=64, BK=32, BV=32, num_warps=1
- 9+ block_ptr loads/stores 全部带 boundary_check
- i32 mask 比较 (`o_k < K`), 运行时 `tl.cdiv`
- 精度: 15/15 PASS

**profiling 关键指标 (per-call, tiny_default)**:
- task_duration: ~12.22 us
- aiv_scalar_ratio: 0.489 (P0 标量退避)
- aic_scalar_ratio: 0.400
- aic_icache_miss_rate: 0.179
- aiv_vec_ratio: 0.045
- cube_utilization: 15.5% (P4 Cube 低占用)

**msprof 聚合 (按 block_num, 5 repeats + 2 warmup)**:
| blk | avg_dur (us) | aic_scalar | aiv_scalar | aiv_vec | cube_util |
|-----|-------------|-----------|-----------|---------|-----------|
| 4 | 11555 | 0.359 | 0.405 | 0.144 | ~15.5% |
| 12 | 20154 | 0.353 | 0.504 | 0.202 | ~15.5% |
| 41 | 19990 | 0.506 | 0.506 | 0.070 | ~15.5% |

---

### Route A — Round 1 (BK=K, BV=V 单 tile + num_warps=4)

**改动**:
1. BK=K, BV=V: 单 tile 直通，K/V 维无 tile 循环，无 mask
2. T_FULL constexpr 分派: T%BT==0 时完全无 boundary_check
3. gk_last 行: `tl.arange(0, K)` 无 mask（K 是 constexpr），一次加载
4. num_warps: 1→4

**结果**:
| blk | avg_dur (us) | Speedup | aic_scalar | aiv_scalar | aiv_vec |
|-----|-------------|---------|-----------|-----------|---------|
| 4 | 9032 | 1.28x | 0.094 | 0.489 | 0.174 |
| 12 | 12030 | 1.68x | 0.139 | 0.476 | 0.170 |
| 41 | 14861 | 1.35x | 0.264 | 0.919 | 0.082 |

- aic_scalar 从 0.359 降至 0.094 (-74%)，标量瓶颈大幅缓解
- blk=41 时 aiv_scalar=0.919，说明大 grid 时 AIV 标量管线仍有压力
- 精度: 15/15 PASS, max_diff=0.000e+00

### Route A — Round 2 (num_warps=8)

**改动**: num_warps 4→8

**结果**:
| blk | avg_dur (us) | vs R1 |
|-----|-------------|-------|
| 4 | 9084 | +0.6% |
| 12 | 12864 | +6.9% |
| 41 | 15546 | +4.6% |

num_warps=8 反而更慢（warp 调度开销 > 并行收益），回退至 num_warps=4。

---

### Route B — Round 1 (BK=64, BV=128 大 tile + num_warps=2)

**改动**:
1. BK=64 (fp32 行宽 256B), BV=128 (fp32 行宽 512B 对齐)
2. T_FULL/K_FULL/V_FULL constexpr 分派
3. num_warps: 1→2

**结果**:
| blk | avg_dur (us) | Speedup | aic_scalar | aiv_scalar | aiv_vec |
|-----|-------------|---------|-----------|-----------|---------|
| 4 | 9045 | 1.28x | 0.115 | 0.511 | 0.210 |
| 12 | 14228 | 1.42x | 0.282 | 0.478 | 0.174 |
| 41 | 15680 | 1.27x | 0.235 | 0.927 | 0.112 |

- aiv_vec 提升明显 (0.144→0.210, +46%)，512B 对齐有效
- blk=41 时 aiv_scalar=0.927，大 grid 标量瓶颈
- 精度: 15/15 PASS, max_diff=0.000e+00

### Route B — Round 2 (num_warps=4)

**改动**: num_warps 2→4

**结果**:
| blk | avg_dur (us) | vs R1 |
|-----|-------------|-------|
| 4 | 9292 | +2.7% |
| 12 | 17401 | +22.3% |
| 41 | 16908 | +7.8% |

num_warps=4 全面劣化，回退至 num_warps=2。

---

### Route C — Round 1 (BK=64, BV=64 + fp32 mask + num_warps=2)

**改动**:
1. BK=64, BV=64: tile 行宽对齐
2. fp32 mask: `o_k.to(tl.float32) < K` 替代 i32 比较
3. T_FULL/K_FULL/V_FULL constexpr 分派
4. gk_last 行复用: 每 K-tile 加载一次

**结果**:
| blk | avg_dur (us) | Speedup | aic_scalar | aiv_scalar | aiv_vec |
|-----|-------------|---------|-----------|-----------|---------|
| 4 | 9542 | 1.21x | 0.125 | 0.481 | 0.188 |
| 12 | 15931 | 1.27x | 0.485 | 0.447 | 0.165 |
| 41 | 14426 | 1.39x | 0.271 | 0.915 | 0.105 |

- blk=41 时表现最佳 (14426us)，但 blk=12 时 aic_scalar=0.485 偏高
- fp32 mask 在 K 不完整时仍有标量开销
- 精度: 15/15 PASS, max_diff=0.000e+00

**fix 记录**: 
1. `K.to(tl.float32)` → `K` (constexpr 不能 .to())
2. 移除 `b_gn = None` 预声明 (cannot reassign constexpr in loop)

---

## 最终推荐

**Route A** 为全局最优路线:

| 场景 | 代表 case | 加速比 |
|------|----------|--------|
| 小 case (blk=4) | tiny_default, T=128 | 1.28x |
| 中 case (blk=12) | k128t127, big_B2_H3_K128 | **1.68x** |
| 大 case (blk=41) | tiny_T2562, T=2562 | 1.35x |

核心优化效果:
- aic_scalar_ratio: 0.359→0.094 (-74%)，标量退避问题基本解决
- aiv_vec_ratio: 0.144→0.174 (+21%)，向量利用率提升
- 精度: 15/15 PASS, max_diff=0.000e+00 (完全一致)

**知识单元覆盖**:
- KU-1 boundary_check 标量降级: T_FULL constexpr 分派
- KU-2 i32/i64 比较标量降级: K/V 维无 mask, gk_last 用 constexpr K
- KU-3 gk_last 行复用: 一次加载整行
- KU-6 大 dot tile: BK=K, BV=V 单 tile 直通
- KU-11 tf32: 保持与上游一致

---

## Round 5: Head-Merge（2026-08-24）—— 目标 case 6796 → 4514us (1.51x)，已采纳

`_recompute_w_u_kernel` 加 `HM: tl.constexpr` head-merge：
grid=(cdiv(T,BT), cdiv(B*H,HM))，每 CTA 循环 HM 个 head（`for hh in range(HM)`），
`i_h = i_h0 + hh`。HM 由 driver 取 `_DEFAULT_HM=16`（H%16==0 时），否则 HM=1。

| 指标 | 旧 | 新 (HM=16) |
|------|-----|------------|
| grid | (256, 96) = 24576 | (256, 6) = 1536 |
| msprof | 6796us | **4514us** |
| max_diff | 0.0 | 0.0（bitwise 一致） |

**结论**: head-merge 摊薄 per-CTA 标量/launch 开销，1.51x 提速，精度零损失。已采纳。
官方 msprof 集成 K4 = **4.51ms**。