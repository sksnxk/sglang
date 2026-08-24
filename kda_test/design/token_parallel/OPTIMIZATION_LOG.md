# OPTIMIZATION_LOG — token_parallel (Diagonal Aqk/Akk)

## 任务信息

| 字段 | 值 |
|------|-----|
| 算子名称 | token_parallel |
| 任务模式 | optimize-existing-kernel |
| 原始 kernel | `src/token_parallel_kernel.py` |
| 目标硬件 | Ascend 910B2 |
| 环境 | triton-ascend-env-zhm, triton 3.2.0 |
| 完成时间 | 2026-08-18T01:50:00 |

---

## 优化路线总览

| 路线 | 策略 | 文件 | Per-token (us) | Speedup | scalar_ratio | 精度 |
|------|------|------|---------------|---------|-------------|------|
| Route B | block_ptr + scalar fix + care_padding | `src/token_parallel_kernel_opt_B.py` | **3.74** | **5.38x** | 0.301 (FAIL) | PASS |
| Route A | tl.dot 向量化 sub-chunk 计算 | `src/token_parallel_kernel_opt_A.py` | 22.04 | 0.91x | **0.059 (PASS)** | PASS |

---

## 迭代记录

### 基线 (原始 kernel)

- Grid: (B*T, H) → 大 case 1,572,864 > 65,535 (超限)
- scalar_ratio: 0.38 (FAIL, >> 0.10)
- Per-token: ~20.1 us
- 精度: 15/15 PASS

### Route B — Round 1 (block_ptr + scalar fix)

**改动**:
1. Grid 拓扑: grid=(B*cdiv(T,BT), H), 每 CTA 处理 BT=64 tokens
2. int64→fp32 比较: `o_k.to(tl.float32) < K`, `j_fp < i_t_fp`
3. block_ptr: 使用 `tl.make_block_ptr` + `tl.load` 加载 k[j]/g[j]
4. care_padding=False 所有 tl.load

**结果**:
- scalar_ratio: 0.38 → 0.30 (-21%)
- vec_ratio: 0.26 → 0.38 (+46%)
- Per-token: 20.1 → 3.74 us (5.38x)
- 精度: 15/15 PASS
- Grid 超限: 已解决

### Route A — Round 1 (tl.dot 向量化)

**改动**:
1. Grid 拓扑: 同 Route B
2. 数学变换: exp2(g[i]-g[j]) → exp2(g[i]) * exp2(-g[j])
3. tl.dot 批量矩阵乘: 预加载 BC 行, 一次性计算 BC×BC gated-dot
4. 因果掩码: tl.where 向量化
5. 2D 写回 + valid_rows 钳制

**结果**:
- scalar_ratio: 0.38 → **0.059** (**-84%, 首次低于 0.10 阈值!**)
- cube_util: 0% → 7.87% (首次使用 Cube 单元)
- mte3_ratio: 0.42 → 0.55 (新瓶颈: 写回占主导)
- Per-token: 20.1 → 22.04 us (0.91x)
- 精度: 15/15 PASS
- Grid 超限: 已解决

---

## 核心发现

1. **Scalar Fallback 根因**: Python for j 循环是标量退避的根本原因。Route A 通过 tl.dot 消除循环后, scalar_ratio 从 0.38 降至 0.059, 首次达标。
2. **数学变换正确性**: exp2(g[i]-g[j]) → exp2(g[i]) * exp2(-g[j]) 等价变换精度损失 max_diff=7.6e-6, 完全可接受。
3. **tl.dot 在 Ascend 上未充分利用 Cube**: cube_util=7.87%, 表明 [16,128]×[128,16] 矩阵太小, 编译器未调度到 Cube 单元。
4. **Route B 效率更高但 scalar 未达标**: 5.38x per-token 加速, 但保留 for 循环导致 scalar_ratio=0.301 未达标。

---

## 最终产出

| 文件 | 路径 |
|------|------|
| Route B kernel | `src/token_parallel_kernel_opt_B.py` |
| Route A kernel | `src/token_parallel_kernel_opt_A.py` |
| 分析文件 | `Ascend-SEKD/Ascend-Claude-Skill/analysis/token_parallel_analysis.md` |
| 基线 profiling | `prof_baseline/` |
| Route B profiling | `prof_opt_B/` |
| Route A profiling | `prof_opt_A/` |

---

## 后续优化建议

1. **P1: 优化 Route A 写回模式** — aiv_mte3_ratio=0.55, 尝试 1D store 逐行写回
2. **P1: 增加 num_warps=2/4** — 改善 Cube 利用率
3. **P2: 消除外层 `for s in range(NC)` 循环** — 展开为 4 个独立子程序

---

## Route C+ — Round 3 (head-merge, target case 10.5ms → 8.6ms)

**动机（msprof, target B1 H96 T16384 K128）**:
- 旧 `_token_parallel_kernel`（grid=(B*NT, H), 1 CTA/(chunk,head)）为 **标量寻址受限**:
  `aic_scalar_ratio=0.463` / `aiv_scalar_ratio=0.415`，`aic_mac_ratio` 仅 0.076
  （tl.dot 的 Cube 只占 7.6%，dot 本身不是瓶颈）。
- 24576 个小 CTA 每 CTA 重复做掩码/地址脚手架，标量开销摊不开。

**改动（`src/token_parallel_kernel.py` 新增 `_token_parallel_kernel_hm2`）**:
1. **Head-merge**: grid=(cdiv(T,BT), B*H//HM)，HM=16（H%16==0 时），每 CTA 串行循环
   16 个 head，CTA 数 24576→1536，摊薄每 CTA 掩码/寻址/launch 开销。
   非 2 幂 K 或 H%16!=0 时回退旧 `_token_parallel_kernel`。
2. **标量裁剪**（对齐 K3）:
   - 去掉 K 维 mask（K 为 2 幂，`tl.arange(0,K)` 直取，load 只带行 mask `m_rows[:,None]`）;
   - scale 折叠进 pre-dot q 乘（消除 post-dot `*scale`）;
   - `exp2(gc)`/`exp2(-gc)` 各算一次复用（每 head 省一次 exp2）;
   - keep/strict 对角块掩码循环外一次预计算。
3. **缓冲 empty 化**: kernel 做无掩码全量写回（掩码处写 0.0），输出无需预清零，
   `torch.zeros`→`torch.empty`，消除每次调用 800MB memset kernel（计时区段内）。

**结果**:
- target kernel 时间: 10.5ms → **8.66ms**（HM=16, nw=1）; HM 24/32/48 无进一步收益
  （8.62-8.72ms 平台），num_warps 1-4 无影响 → 瓶颈为每 head 的 2D 访存地址生成。
- 精度: target Aqk=1.12e-08 / Akk=8.94e-08; 集成后 bench group D 全部 OK
  （含 target case K2 max_diff=8.94e-08）。
- 集成后 wall-clock（含 gather + Python 开销）10.95ms; msprof 口径见 results.csv。

### Route C+ — Round 3.5 (compact Akk via tl.gather, kernel 8.66→9.43ms, 集成后 ~9.4ms)

**动机**: msprof 分段显示 hm2 的集成段 = `_token_parallel_kernel_hm2` 8.69ms +
**torch.gather 链 ~2.0ms/调用**（GatherElementsV2 1.67ms + Cast/BroadcastTo/Arange/
FloorDiv/Add 等 0.33ms）。gather 读 402MB 满宽 scratch，几乎和 kernel 一样贵。

**改动**:
1. 验证 triton-ascend 3.2.1 支持 `tl.gather`（max_diff=0）。
2. 新增 `_token_parallel_kernel_hm3`: 每 head 算满宽 Akk_full [64,64] 后，用
   `tl.gather(Akk_full, col_idx, axis=1)`（col_idx[r,j]=(r//16)*16+j, 行依赖列偏移）
   在 kernel 内把 4 个对角 16×16 块收拢为 [64,16]，写紧凑 [B,T,H,BC]。
   消除 scratch 缓冲 + driver torch.gather kernel。
3. driver 快速路径（K 为 2 幂）改走 hm3，Akk 直接返回 [B,T,H,BC]。

**结果**:
- kernel 时间 8.66→**9.43ms**（tl.gather 在 Ascend 上 ~0.77ms, 非零成本），
  但整体集成时间 **10.7→~9.4ms**（省掉 2.0ms torch.gather 链）。
- wall-clock（Python + launch）9.69ms; 精度 target Aqk=1.12e-08 / Akk=8.94e-08,
  group D 全 OK。

**遗留**:
- tl.gather 的 ~0.77ms 是 Triton 未把"行内连续 16 列"识别为廉价切片所致;
  用 4 个独立 [16,128]@[128,16] 对角块 dot 可免 gather，但需按 sub-chunk 切片
  ke/kbe（Triton 无法切 UB tile 的行），或按 sub-chunk 单独 load（load 指令 4x,
  更差），未做。
- BT=128（全部 6 算子统一）可把 load/store 指令数减半，风险高，未做。
4. **P2: block_ptr 预取** — 替代手动偏移加载