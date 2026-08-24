# OPTIMIZATION_LOG: inter_solve (K3)

## 优化思路

**问题分析**: K3 inter_solve 在目标 CASE (B=1, T=16384, H=96, K=128) 下不存在 grid 超限问题。当前 grid = (NT, B*H) = (256, 96)，展平后 24576 ≤ 65535，NPU 支持无问题。

**瓶颈诊断**:
- Kernel 已实现单 kernel 三阶段融合 (Phase 1: 非对角块计算, Phase 2: 对角块前向替换, Phase 3: 链式矩阵乘合并逆)
- speedup ~2.0x 已经优于 torch_npu 元算子版本
- 主要计算量在 Phase 1 的 K 维循环（tl.dot bf16 累加），Phase 3 的链式 matmul 占比相对较小

**优化方向**:
1. **去除 `input_precision="ieee"`** — Phase 3 的 `tl.dot` 调用全部带有 `input_precision="ieee"`，在 Ascend NPU 上强制使用 IEEE 精确语义会限制编译器优化。去除后让编译器自动选择最佳精度策略。
2. **Phase 2 寄存器预加载** — 从 KKB 的 `solve_tril.md` 经验中提取: 预加载整个 16×16 对角块到寄存器，前向替换时从寄存器提取行，避免 56 次 HBM 重读。但 triton 3.2.0 不支持 `tl.extract_slice` 和 `b_D0[i, :]` 动态索引，此优化暂不可行。

## 修改内容

**文件**: `src/inter_solve_kernel.py` — `_inter_solve_kernel` 函数

| 修改 | 说明 |
|------|------|
| 去除所有 `input_precision="ieee"` | Phase 3 的 6 个 `tl.dot` 调用链，共 15 处，全部去掉 |

## 优化前后对比

### 目标 CASE: B=1, T=16384, H=96, K=128

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| grid | (256, 96) | (256, 96) |
| 展平 grid | 24576 (≤65535 ✅) | 24576 (≤65535 ✅) |
| 精度 max_diff (Aqk) | 5.634e-04 | 5.634e-04 (不变) |
| 精度 max_diff (Akk) | 1.943e-03 | 1.943e-03 (不变) |
| triton 耗时 | 66.55ms | 66.53ms |
| torch_npu 耗时 | 132.97ms | 137.61ms |
| speedup | 2.00x | 2.07x |

### 是否解决不支持问题
✅ **不需要** — grid 展平 24576 ≤ 65535，原本就支持。

### 全量回归验证
15/15 case 全部 PASS，所有 case 精度 max_diff < 1e-2。

## 额外测试: K=32 稳定性

| 指标 | 值 |
|------|-----|
| torch_npu | 100.11ms |
| triton | 58.02ms |
| speedup | 1.73x |
| max_diff Aqk | 1.754e-03 |
| max_diff Akk | 5.505e-03 |
| 结论 | PASS (tl.dot 稳定，无异常) |

## 备注

- K3 inter_solve 已具有较好的性能，无阻塞性问题，speedup ~2.0x 已优于 torch_npu 元算子版本
- Phase 2 寄存器预加载优化因 triton 3.2.0 不支持 `tl.extract_slice` 和动态张量索引而暂不可行，后续 triton 版本升级后可重新评估
- 目标 CASE 的计算量随着 H 增大而线性增加（grid = (256, 96) = 24576 CTAs），H=96 时每个 CTA 处理 1/96 的工作负载
---

## 第二轮优化: 融合单 kernel + 重复平方截断逆（npow=3）

### 结论
**最优 = 融合单 kernel（head-merged HM=16, npow=3 截断逆）, 目标 case 14.5ms**（旧 kernel 66.5ms → **4.6×**; 相对 torch_npu 135ms → **9.3×**）。

### 关键改动
1. **融合 Phase1+2 为单 kernel**: 每 CTA 循环 HM 个 head（grid=(256,6)=1536 CTAs），避免高 CTA 数下 10-dot 链 aicore exception（根因 CANN UB 分配代码生成 bug，head-merge 后 24576→1536 CTAs 规避）。
2. **重复平方逆替代链式 23 小 dot**: (I-L)^{-1}=(I-L)(I+L²)(I+L⁴)(I+L⁸)，npow=3 → 6 个 [64,64] dot（满精度 npow=5 需 10 dot）。
3. **Mkk 直接给出对角块**: 与 K2 的 Akkd 数学一致（beta_i·k·k·exp2(g_i-g_j) 对角 16×16 strict-lower），无需读 Akkd。

### 精度（vs 真实 ground truth inter_solve_torch，真值 Akkd）
| 输出 | max_diff |
|------|----------|
| Aqk | 1.86e-08 |
| Akk | 2.09e-07 |

unified bench 目标 case max_diff=7.45e-08（旧 kernel 5.4e-04）。

### 截断级数扫描（目标 case）
| npow | dot/head/chunk | 时间 | Akk 精度 |
|------|------|------|----------|
| 5 | 12 | 19.2ms | 1.9e-7 |
| 4 | 10 | 16.2ms | 1.9e-7 |
| **3** | **8** | **14.4ms** | 1.9e-7 |
| 2 | 6 | 18.0ms（编译器代码生成异常） | 7.1e-5 |

### 验证覆盖
target (B1,H96,T16384,K128)、small (B1,H2,T1024,K64)、B2/H8/T2048/K64、B4/H4/T4096/K128 全部 PASS（Aqk/Akk < 2.1e-7）。

---

## 第三轮: 缓冲 empty 化 + 无掩码 store（结果: 中性，未提速）

**动机（对齐 K2 hm3 playbook）**: 旧驱动 `torch.zeros(B,T,H,BT)` 每调用分配 2×402MB 并触发
ZerosLike memset（K2 段曾测到 420us/次）。

**改动（`src/inter_solve_kernel.py`）**:
1. 驱动缓冲改 `torch.empty(B, TP, H, BT)`（TP=NT*BT，补齐尾 chunk），kernel 做无掩码全量写回，
   返回 `[:, :T]`。
2. kernel `_inter_solve_kernel` 加 `TP` 参数（store 用 TP 步长 `b_offA = i_b*TP`，load 仍用 T），
   去掉两个 store 的 `mask=m_t[:, None]`（loads 已行掩码，尾行写 0；Akk_out 对角尾行 1.0 被切片丢弃）。

**结果（目标 case，msprof 集成）**:
- 正确性: verify_integrated 全 PASS（Aqk 1.9e-8 / Akk 2.1e-7），与改动前一致。
- 时间: 14.44 → **14.51ms**（中性）。msprof 段内 ZerosLike 已消失（段内仅 7 次 `_inter_solve_kernel`），
  但 kernel 本体时间未降 —— 缓冲区分配不在段内计时的主要开销里。

**结论**: 对 K3 无提速收益（保留 empty+无掩码写法，正确且对非对齐 T 更稳）。

---

## 第四轮: 进一步降 dot 成本尝试（均未成功，K3 停在 ~14.5ms）

| 实验 | 结果 |
|------|------|
| fp16 逆链 dot（fp32 累加） | 14.37→**14.87ms**（更慢），Akk 精度 1.9e-7→3.0e-4（仍 <1e-2） |
| num_warps 1/2/4/8 | 14.5-14.7ms 平台，无影响 |
| 2-head 链交错（依赖重叠） | BiShengIR 编译病态（scf.for memref 地址空间/长时编译），放弃 |
| HM=1/2（24576/12288 CTA） | 触发 aicore exception（head-merge 规避的根因），设备挂起，放弃 |

**瓶颈诊断（msprof）**: K3 段 = 纯 kernel 时间（7× ~14.5ms）。aic_scalar≈49%、cube≈12.3%，
8-dot 链（Mkk/Mqk 2 大 dot + 逆 6 小 dot）约贡献 5ms，逆 dot 为串行依赖 + 小 tile，
cube 仅 ~3% 峰值 —— 非吞吐瓶颈，是标量寻址 + 依赖延迟主导。结构性优化（BT=128、
布局转置）风险高未做。

---

## 第五轮: 目标 case 最终扫参（真实 chained 输入, 2026-08-24）—— 确认收敛

本轮用 unified bench 的 torch 链输入（K1→K3 真实 Akkd）与 NPU 参考 `inter_solve_torch`，
把 K3 剩余杠杆全部扫完。**结论: K3 停在 14.4ms 是实测最优，不再有可用杠杆。**

### NP 截断级数（chained 输入, HM=16, nw=4）
| NP | dot/head/chunk | 时间 | Akk_inv max_diff |
|----|------|------|------|
| 2 | 6 | 18.18ms | ~1e-7 |
| **3** | **8** | **14.67ms** | ~1e-7 |
| 4 | 10 | 16.39ms | ~1e-7 |

（早前 NP2 的 18.2ms 结论复现：平台怪癖，dot 更多反而 pipeline 更好。）

### HM × num_warps 扫描（chained 输入, NP=3）
| HM | nw1 | nw4 |
|----|-----|-----|
| 8  | 14.76ms | 14.82ms |
| **16** | 14.78ms | **14.49ms ← 最优** |
| 32 | 14.65ms | 14.93ms |

### fp16 逆链（out_dtype=fp16 修正后, chained 输入）
f32 链 = 14.94ms (diff 7.5e-8)；f16 链 = **15.00ms（无提速）**, diff 1.4e-4。
Ascend cube fp16 双倍速在 triton-ascend 上未体现，且精度变差 —— 死路。

**结论**: K3 = NP=3 + HM=16 + nw=4 + fp32 = **14.37ms**（msprof 官方）。
