# Kernel 5 设计文档: 独立 Delta Rule H 算子

> 本文档是 `kda_test/design/Kernel5_DeltaRuleH.md` 的**独立子集重构**。
> 与原始文档的差异:
> - 只保留 `B,T,H,K` 固定长度的最小闭环, 不涉及 VARLEN / `cu_seqlens` /
>   `chunk_offsets` / `chunk_indices` 等扩展路径;
> - 只保留 `USE_GK=True` + `USE_EXP2=True` + `INPLACE_UPDATE=True` +
>   `SAVE_NEW_VALUE=True` + `USE_INITIAL_STATE=True` 路径, 删除 `USE_G`
>   (标量 gate) 分支与 `USE_EXP2=False` 分支;
> - 行号改为引用本目录的 `src/delta_rule_h_kernel.py`;
> - 增加「精度测试策略」和「性能测试思路」两节, 配套本目录测试驱动。

---

## 1. 输入输出定义

### 1.1 输入张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `k` (kg) | `[B, T, H, K]` | fp32 | 衰减后的 key: `k * beta * exp2(gk_last - gk)` (由 Kernel-4 输出) |
| `w` | `[B, T, H, K]` | fp32 | 衰减后的 w (由 Kernel-4 输出) |
| `u` | `[B, T, H, V]` | fp32 | 原始 value: `Aqk @ (v * beta)` (由 Kernel-4 输出), **V == K** |
| `gk` | `[B, T, H, K]` | fp32 | per-channel gate (log2 空间, 已 chunk-local cumsum + `RCP_LN2` 缩放) |
| `initial_state` | `[N, H, V, K]` | fp32 | 初始状态 (`N = B`, 通过 `initial_state_indices` 索引) |
| `initial_state_indices` | `[B]` | int32 | 每个 batch 条目指向 `initial_state` 的索引 |

编译期常量: `H` (value head 数), `Hg` (key head 数, GQA 分组), `K` (head 维度),
`V` (value head 维度, **恒等于 K**), `BT` (chunk 大小 = 64), `BV` (V 维 tile 大小 = 32,
env `SGLANG_GDN_CHUNK_H_BV`)。

### 1.2 输出张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `h` | `[B, NT, H, V, K]` | fp32 | 每 chunk 起始状态快照 (供 Kernel-6 读取) |
| `v_new` | `[B, T, H, V]` | fp32 | Delta Rule 残差 value: `v_new = u - w @ h[t-1]` (供 Kernel-6 读取) |
| `initial_state` | `[N, H, V, K]` | fp32 | (in-place 更新为最终状态, 供下一 batch 使用) |

`NT = cdiv(T, BT)` 为 chunk 总数。

## 2. 分核并行策略 (Grid 拓扑)

```
Grid = (cdiv(V, BV), N * H)
        ~~~~~~~~~~~  ~~~~~~
              |          |
        V 维分块 (BV=32)  (batch, head) 对
```

- `cdiv(V, BV)`: V 按 `BV=32` 切分得到 V-tile 数;
- `N * H`: 每个 (batch, head) 组合一个平面 (`N = B` 在固定长度模式下)。

每个 CTA 处理一个 `((batch, head), V-tile)`, 加载 `[BT, 64]` 的 w/k 切片
(K 维按 64 分 tile 展开), 与 `[BT, BV]` 的 v/v_new 切片, 在该 V-tile 上完成
**整段 NT 个 chunk 的串行递推**。

### 程序 ID 映射

```
i_v  = program_id(0)   V 维 tile 索引 (0 .. cdiv(V, BV)-1)
i_nh = program_id(1)   (batch, head) 联合索引 (0 .. B*H-1)
i_n  = i_nh // H       batch 索引
i_h  = i_nh %  H       head 索引
```

### 并行维度表

| 维度 | 粒度 | 并行方式 |
|------|------|----------|
| 序列 (B/N) | 每个序列 | **Grid Y 并行**: 每个序列-头对为一个 CTA |
| 头 (H) | 每个头 | **Grid Y 并行**: 序列内所有头可并行 |
| V 维度 | BV=32 个元素/tile | **Grid X 并行**: 不同 V-tile 可并行 |
| Chunk (NT) | 整个序列的所有 chunk | **串行**: 每个 CTA 内用 `for i_t in range(NT)` 顺序推进 |
| K 维度 | 64 个元素/tile | **CTA 内展开**: 4 个寄存器块 b_h1~b_h4 覆盖 K<=256 |

## 3. 计算思路

### 3.1 整体算法: Delta Rule 跨 Chunk 递推

Delta Rule 的核心思想是: 用线性注意力近似, key-value 记忆通过外积不断累积到状态矩阵 h 中。

每个 chunk t 包含 BT=64 个连续 token。状态 h 在经过每个 chunk 后更新:

```
h[t] = h[t-1] * decay[t] + K[t]^T @ V_new[t]
```

其中:
- `h[t]` 是第 t 个 chunk 结束时的状态矩阵, 形状为 `[V, K]`;
- `decay[t]` 是第 t 个 chunk 结束时的遗忘因子: `exp2(gk_last[t])` (per-channel);
- `V_new[t] = U[t] - W[t] @ h[t-1]^T` 是 Delta Rule 的核心: 残差 = 原值 - 历史预测;
- `U[t]` 是 chunk t 中经 beta 和 Aqk 变换后的 value;
- `W[t]` 是 chunk t 中经 beta 和 gate 衰减后的 key-stats;
- `K[t]` 是 chunk t 中衰减后的 key (即输入 `k`/`kg`)。

### 3.2 初始状态加载

```
initial_state 形状: [N, H, V, K]  (N = B, indices = arange(B))

每个序列通过 initial_state_indices[B] 索引到它的状态池:
  h0_ptr = initial_state + initial_state_indices[i_n] * (H * V * K) + i_h * V * K

加载当前 V-tile [i_v*BV : (i_v+1)*BV, :] 到 b_h1 ~ b_h4 (4 个 K-tile):
  b_h1 = h0[i_v*BV : (i_v+1)*BV,    0 : 64 ]   ← K tile 0
  b_h2 = h0[i_v*BV : (i_v+1)*BV,  64 : 128]   ← K tile 1 (if K > 64)
  b_h3 = h0[i_v*BV : (i_v+1)*BV, 128 : 192]   ← K tile 2 (if K > 128)
  b_h4 = h0[i_v*BV : (i_v+1)*BV, 192 : 256]   ← K tile 3 (if K > 192)
```

### 3.3 Chunk 循环体

主循环 `for i_t in range(NT)` 依次处理每个 chunk, CTA 内部串行。

```
for i_t = 0, 1, 2, ..., NT-1:

    chunk t ──────────────────────────┐
    │                                  │
    │  ① 保存状态快照 h[i_t] = state   │
    │  ② Delta Rule 计算 v_new         │
    │  ③ per-channel gate 衰减         │
    │  ④ 状态更新 h += kg^T @ v_new    │
    └──────────────────────────────────┘
         ↓
    chunk t+1 使用更新后的 h
```

#### 步骤 1: 保存状态快照 `h[i_t] = state`

```
目的: 保存当前 chunk 开始时的状态到全局内存 h[B, NT, H, V, K]
这样 Kernel-6 (output kernel) 在计算最终输出时可以直接读取 h[t-1]

为什么用 flat 1D pointer store 而不是 block_ptr?
  → triton-ascend 编译器 bug: block_ptr store 到 (V, K) 形状会损坏源寄存器
  → 变通方案: reshape 成 (BV*64,) 的一维 flat tensor, 用 1D store
  (b_h1 + 0) 强制编译器用新鲜寄存器, 防止污染
```

#### 步骤 2: Delta Rule 计算 `v_new = u - w @ h[t-1]^T`

```
Delta Rule 核心:
  v_new[t] = u[t] - W[t] @ h[t-1]^T

其中:
  - u[t]: chunk t 的原始 value     → [BT, V]
  - W[t]: chunk t 的 w 矩阵         → [BT, K]
  - h[t-1]: 上一 chunk 结束时的状态 → [V, K]
  - h[t-1]^T: 状态的转置            → [K, V]
  - W[t] @ h[t-1]^T: 历史预测       → [BT, V]
  - v_new[t]: 残差 (误差)            → [BT, V]

分块计算 (K 维度分 4 个 64 宽的 tile):
  b_v = 0  (初始化为 0)
  b_v += W_tile1 [BT, 64]  @  h1^T [64, BV]   ← K-tile 0
  b_v += W_tile2 [BT, 64]  @  h2^T [64, BV]   ← K-tile 1 (if K > 64)
  b_v += W_tile3 [BT, 64]  @  h3^T [64, BV]   ← K-tile 2 (if K > 128)
  b_v += W_tile4 [BT, 64]  @  h4^T [64, BV]   ← K-tile 3 (if K > 192)

  b_v = u[t] - b_v    ← 残差 = 原始值 - 历史预测
  (代码中 p_v 加载的是 u[t], 然后减去预测得到 v_new[t])
```

#### 步骤 3: 保存 v_new

```
SAVE_NEW_VALUE=True: 把 v_new 存到 v_new[B, T, H, V], 供 Kernel-6 output 使用
  tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
```

#### 步骤 4: Per-Channel Gate 衰减 (USE_GK + USE_EXP2)

```
USE_GK: 逐 token / 逐 channel gate (log2 空间, 已 chunk-local cumsum)

加载 gk_last (chunk 末尾的 gk):
  b_gk_last1 = gk[last_idx, 0:64]     ← [64], K-tile 0 的末尾 gate
  b_gk_last2 = gk[last_idx, 64:128]   ← [64], K-tile 1 的末尾 gate
  ...

应用 gate (采用 exp2):
  b_h1 *= exp2(b_gk_last1)[None, :]   ← [64] 广播到 [BV, 64]
  b_h2 *= exp2(b_gk_last2)[None, :]
  ...

含义:
  每个 K channel 有独立的衰减因子 exp2(gk_last)。
  这允许模型对不同维度的 key 信息有不同的保留时间。
  信息衰减快的 channel 很快被遗忘, 衰减慢的 channel 长期保留。
```

#### 步骤 5: 状态更新 `h += kg^T @ v_new`

```
状态更新 (外积累加):
  h_new = h_decayed + K[t]^T @ v_new[t]

矩阵维度:
  - K[t]   → [K, BT]   (key 矩阵, 行为 K, 列为 BT tokens)
  - v_new  → [BT, BV]  (chunk t 的 v_new, V 维度的当前 tile)
  - h_new  → [BV, K]   (当前状态, V-tile x K-tile)

分块计算:
  使用 tl.dot(b_k, b_v) 然后转置:
  b_h1 += tl.trans(tl.dot(b_k1, b_v))   ← b_k1 [64, BT] @ b_v [BT, BV] → [64, BV] → trans → [BV, 64]
  b_h2 += tl.trans(tl.dot(b_k2, b_v))
  b_h3 += tl.trans(tl.dot(b_k3, b_v))
  b_h4 += tl.trans(tl.dot(b_k4, b_v))

  注意: k = kg = k * beta * exp2(gk_last - gk) (已由 Kernel-4 计算好)
```

### 3.4 Epilogue: 最终状态写回 (INPLACE_UPDATE=True)

```
序列处理完所有 NT 个 chunk 后, 最终状态 ht 写回 initial_state (in-place):

  initial_state[initial_state_indices[i_n], i_h, i_v*BV:(i_v+1)*BV, :] = final_h

使用 flat 1D store (与 snapshot 相同的 triton-ascend bug 规避方案):
  p_ht = ht + i_v * BV * K + offset_k + tl.arange(0, BV * 64)
  tl.store(p_ht, b_h_flat.to(ht.dtype.element_ty))

目的: 为下一个 batch 提供正确的初始状态 (KV Cache 管理)。
```

### 3.5 为什么 state 需要衰减 (遗忘机制)

```
State 衰减是线性注意力模型的核心机制:

  h[t] = h[t-1] * decay + K[t]^T @ V_new[t]

如果没有 decay 项:
  → h[t] 是所有历史 token 的外积累加: h[t] = SUM_{i=0}^{t} K[i]^T @ V_new[i]
  → 远处 token 和近处 token 贡献权重相同
  → 模型无法聚焦近期信息

加入 decay 项后:
  → h[t] = decay^t * h[0] + SUM_{i=1}^{t} decay^(t-i) * K[i]^T @ V_new[i]
  → 历史越久远的状态 weight 越小 (指数衰减)
  → 类似标准 softmax attention 中因果 mask 的位置 bias
  → 不同 head / channel 可有不同的衰减速率

Gate 设计 (本目录只保留 USE_GK + USE_EXP2 路径):
  - USE_GK:  per-channel gate exp2(gk_last), 每个 channel 独立衰减
  - USE_EXP2: gk 使用 exp2 (log2 空间), 数值更稳定
  (USE_G + natural exp 的组合被删除, 因为本项目不调用)
```

### 3.6 为什么需要保存 h 快照 (供 Kernel-6 读取)

```
Kernel-6 (chunk_gla_fwd_kernel_o) 是 output kernel, 计算:
  o[t] = q[t] * exp2(g[t]) @ h[t]^T + causal_local[t]

其中 h[t] 是第 t 个 chunk 的起始状态 (即第 t-1 个 chunk 结束时的状态)。
Kernel-6 从 Kernel-5 写入的 h[B, NT, H, V, K] 中读取 h[t]:

  Kernel-5 循环中的保存时序:
    chunk 0: 保存 h[0] = initial_state,   → 计算 → h_new = updated
    chunk 1: 保存 h[1] = h_new,           → 计算 → h_new = updated
    chunk 2: 保存 h[2] = h_new,           → 计算 → h_new = updated
    ...

  Kernel-6 读取:
    o[chunk_t] 使用 h[chunk_t_index] (即 h[i_tg])
    其中 i_tg = i_b * NT + i_t

  所以 h[t] 总是记录第 t 个 chunk 开始时的状态。
```

## 4. 关键代码对应 (`src/delta_rule_h_kernel.py`)

| 功能 | 行号 | 说明 |
|------|------|------|
| CPU 参考 `delta_rule_h_ref` | 47-118 | 逐 chunk 串行, 整体 [V, K] 计算 |
| torch_npu 元算子 `delta_rule_h_torch` | 124-167 | 与 ref 数学一致, 在 NPU 上用 matmul/exp2 |
| triton kernel `_delta_rule_h_kernel` | 175-330 | 固定长度 + USE_GK + USE_EXP2 子集 |
| Python driver `delta_rule_h_triton` | 335-379 | grid 计算 / 张量分配 / 调度 |
| CTA 标识计算 | 187-188 | `i_v, i_nh = program_id(0), program_id(1)` |
| State 寄存器分配 | 195-201 | `b_h1` ~ `b_h4`, 每个 `[BV, 64]` fp32 |
| 初始状态加载 | 219-231 | `tl.make_block_ptr(h0, ...)` 分 4 个 K-tile |
| 步骤 1: 保存状态快照 | 238-257 | flat 1D store workaround |
| 步骤 2: Delta Rule (W @ h) | 260-281 | `b_v = tl.dot(...)` 累加, `b_v = u - b_v` |
| 步骤 3: 保存 v_new | 284-286 | `tl.store(p_v_new, ...)` |
| 步骤 4: per-channel gate 衰减 | 289-317 | `b_h *= exp2(b_gk_last)[None, :]` |
| 步骤 5: 状态更新 | 319-330 | `b_h += tl.trans(tl.dot(b_k, b_v))` |
| Epilogue: 写回 final state | 333-348 | flat 1D store 到 `ht` (initial_state in-place) |

## 5. 数据流图

```
═══════════════════════════════════════════════════════════════════════════════════
                        Kernel 5: Delta Rule 跨 Chunk 递推
═══════════════════════════════════════════════════════════════════════════════════

    输入: k(=kg), w, u(=v), gk                输入: initial_state per sequence
    ┌──────────────────────────┐              ┌─────────────────────────────┐
    │ from Kernel 4            │              │ from KV Cache Pool          │
    │ (recompute_w_u_fwd)      │              │ [N, H, V, K]                │
    └──────────────────────────┘              └─────────────────────────────┘
              │                                    │
              │                                    ▼
              │                       ┌─────────────────────────┐
              │                       │ Load to registers       │
              │                       │ b_h1~b_h4 [BV, 64]      │
              │                       │ = initial_state slice   │
              │                       └──────────┬──────────────┘
              │                                  │
              │              ╔═══════════════════╧═══════════════════╗
              │              ║     Chunk Loop: for i_t in 0..NT-1     ║
              │              ╠═══════════════════════════════════════╣
              │              ║  ① Save snapshot h[i_t] = b_h          ║
              │              ║  ② Delta Rule: v_new = u - W @ b_h    ║
              │              ║  ③ Save v_new to v_new[i_t]            ║
              │              ║  ④ Gate Decay: b_h *= exp2(gk_last)   ║
              │              ║  ⑤ State Update: b_h += k^T @ v_new   ║
              │              ╚═════════════════════╤═══════════════════╝
              │                                     │
              │                                     ▼
              │                       ┌─────────────────────────┐
              │                       │ Epilogue:               │
              │                       │ Write final state to    │
              │                       │ initial_state in-place  │
              │                       └─────────────────────────┘
              ▼
    输出: h [B, NT, H, V, K]              输出: initial_state (updated)
    ┌──────────────────────────┐          ┌─────────────────────────────┐
    │ consumed by Kernel 6    │          │ consumed by next batch's    │
    │ (chunk_gla_fwd_o_gk)     │          │ inference as initial_state  │
    └──────────────────────────┘          └─────────────────────────────┘

    输出: v_new [B, T, H, V]
    ┌──────────────────────────┐
    │ consumed by Kernel 6    │
    │ (chunk_gla_fwd_o_gk)     │
    └──────────────────────────┘
```

## 6. 精度 & 性能对比测试策略 (配套 `run.py` / `testcases.csv`)

参考: `test_level2_kernel_precision.py::TestDeltaRuleKernel`

- 每个 case 固定 seed, 输入分布与 level2 一致 (`q/k = normalize(randn)`,
  `w/u = randn*0.1`, `gk = randn*0.5-2.0`, `initial_state = randn*0.05`);
- 对每个 case 依次跑**两个对比方**:
  1. **torch_npu 元算子** `delta_rule_h_torch` —— 用 torch_npu 现成 matmul /
     exp2 / 广播乘法组合完成同样计算, 作为**精度基准**和**性能基准**;
  2. **triton kernel** `delta_rule_h_triton` —— 本目录实现的单 kernel 版本;
- 精度指标 (两个都满足才 PASS):
  - `max|triton_h - torch_h| < 1e-2` 与 `max|triton_v_new - torch_v_new| < 1e-2`;
  - `max|torch - CPU 参考| < 1e-2` 且 `max|triton - CPU 参考| < 1e-2`
    (CPU 参考 `delta_rule_h_ref` 为逐 chunk 串行的 ground truth)。
  实测 maxdiff 均为 fp32 累积噪声级 (≤ ~1e-5)。
- 性能指标: 预热 `--warmup`(默认 5) 次, 各跑 `--repeats`(默认 30) 次, 用
  `torch.npu.synchronize()` 包裹计时, 输出每个 case 的
  **加速比 = torch_npu_time / triton_time**。
- CSV 中共 15 个 case, 覆盖: 完整 chunk / 尾 chunk 不满
  (`T=63/65/96/100/127/193/255/256/2562`)、单/多 head (`H=1/2/3`)、
  单/多 batch (`B=1/2`)。**所有 case 固定 `K=V=64`** —— 上游
  `chunk_delta_h` kernel 的 flat 1D store 假定行步长 K=64, 在 K≠64 时会
  产生错误结果 (与 `test_level2_kernel_precision.py` 的约定一致)。

> 预期: 15/15 PASS; 加速比受 chunk 间串行依赖限制, 但 triton 单 kernel 节省
> 了 torch_npu 多 kernel 启动 + 中间张量读写开销。

## 7. 性能测试思路 (配套 `run.py`)

内存 + 计算混合型 kernel (读 `B*T*H*K` 的 kg/w/u/gk + 读 `B*H*V*K` 的
initial_state + 写 `B*NT*H*V*K` 的 h + 写 `B*T*H*V` 的 v_new + 写回 final state),
性能对比的两个对象:

- **torch_npu 元算子**: 逐 chunk 串行的 matmul / exp2 / 广播乘法 + 外积 (每
  chunk 多条 kernel, 另有 zeros / clone / copy 等), 是性能基准的下界参考;
- **triton kernel**: 单 kernel 完成快照 + Delta Rule + gate 衰减 + 外积更新 +
  写回, 避免中间张量;
  每 case 预热 5 次, 计时 30 次 (`torch.npu.synchronize()` 包裹), 报告
  `torch_ms / triton_ms / speedup`。

加速比主要来自: 单 kernel 复用 (state 寄存器 b_h1~b_h4 跨 chunk 保留) 与
tunable 的 tile 选择 (`BV=32`) —— Delta Rule 递推要求 chunk 间串行, 但每个
chunk 内部的 matmul / exp2 仍可被 triton-ascend 优化为单次 kernel 启动。

后续如做 kernel 级优化, 对照变量: `BV`(32 vs 64), `num_warps`, `num_stages`,
以及 K-tile 展开 (4 个 64-宽 tile 是否可融合为更宽的 tile)。

## 附录 A: 关键常量速查

| 常量 | 值 | 用途 |
|------|-----|------|
| `BT` | 64 | Chunk 大小 / 时间维 tile 大小 |
| `BV` | 32 | V 维 tile 大小 (env `SGLANG_GDN_CHUNK_H_BV`) |
| `num_warps` | 4 | 每 CTA warp 数 (env `SGLANG_GDN_CHUNK_H_NUM_WARPS`) |
| `num_stages` | 2 | pipeline stage 数 (env `SGLANG_GDN_CHUNK_H_NUM_STAGES`) |
| K-tile 宽 | 64 | K 维按 64 分 tile 展开, 4 块覆盖 K<=256 |

## 附录 B: 与真实 kernel 的差异对照

| 项 | `python/.../fla/chunk_delta_h.py` | 本目录独立实现 |
|----|------------------------------------|----------------|
| VARLEN / `cu_seqlens` | 支持 | 只做固定长度 |
| `USE_G` (标量 gate) | 支持 | 删除 (本项目不调用) |
| `USE_EXP2=False` (natural exp) | 支持 | 删除 (本项目恒为 True) |
| `chunk_offsets` | 支持 | 只做固定长度 |
| Autotune multi-config | 仅 1 config (避免 in-place 污染) | 单 config (env 可调) |
| 依赖 | sglang 包 + `fla.op.exp2` 等 | 仅 torch + triton + `tl.math.exp2` |
