# Triton Kernel 编程试题：Chunkwise Linear Attention

## 一、背景介绍

### 1.1 问题：长文本推理中的注意力计算瓶颈

在 Transformer 架构中，标准 Softmax Attention 的每个 token 需要与所有历史 token 计算相似度：

```
对于位置 t 的 token:
  output[t] = softmax(q[t] · k[0], q[t] · k[1], ..., q[t] · k[T-1]) @ v
```

当序列长度 T=128K 时，注意力矩阵大小为 `T × T`（约 160 亿个元素），计算复杂度 O(T²)，无法在单芯片上完成。这是长文本推理的核心瓶颈。

### 1.2 解决方案：Chunkwise Linear Attention

KDA（Kimi Delta Attention）将 Linear Attention 与分块策略结合，把 O(T²) 的计算量降为 O(T)：

**核心思路——分块处理：**

将 T 个 token 按 `chunk_size=64` 切成 NT = ceil(T/64) 个 chunk。每个 chunk 内部可以并行做矩阵乘法（O(T × 64)），chunk 之间通过一个压缩的"状态矩阵"串联传递历史信息（O(NT)）。整体复杂度从 O(T²) 降为 O(T × 64)，在 T=128K 时约 800 万次操作，远小于 160 亿次。

```
T 个 token 划分为 NT 个 chunk：
  |---- Chunk 0 (64 tokens) ----|---- Chunk 1 (64 tokens) ----|---- Chunk 2 (64 tokens) ----| ...

每个 chunk 内部再切为 4 个 sub-chunk（各 16 token）：
  | SC0 (0..15) | SC1 (16..31) | SC2 (32..47) | SC3 (48..63) |
```

**三个关键机制：**

**(a) Gate（门控衰减）：** 每个 token 在每个 key channel 上有独立的"遗忘速率"。gate 值是负数，经过 exp 后变为 (0,1] 范围的衰减因子，控制旧信息在状态矩阵中的保留比例。

```
gate[t, k] = -exp(A_log[h]) * softplus(raw_gate[t, k] + dt_bias[h, k])
```

gate < 0，故 exp(gate) < 1。gate 越负（绝对值越大），旧信息衰减越快。每个 channel 学习不同的衰减速率，使模型能灵活选择保留多久的历史。

**(b) Delta Rule（增量更新）：** 避免在状态矩阵中重复存储已知信息。先用当前 key 从状态矩阵中"预测"value，只把预测不到的残差（delta）写进去。

```
predict = k[t] @ state           ← 用 key 查询历史状态，预测 value
delta   = v[t] - predict         ← 只保留预测不到的新信息
state   = state * decay + k[t]^T @ delta   ← 衰减旧信息 + 写入新信息
```

如果当前 token 的信息可以从历史状态完全预测，delta = 0，不写入任何内容。这种"增量编码"使状态矩阵保持紧凑。

**(c) 状态矩阵 h：** 这是一个 `[K, V]` 矩阵（K 行 V 列），将 chunk c 之前所有 token 的信息压缩在其中。读取方式：`q[K] @ h[K, V] = [V]`，即用 gated query 从状态中读出跨块的历史信息。

```
h[c]  = 压缩了 chunk 0, 1, ..., c-1 中所有 token 的信息
o_cross = (q * exp2(g)) @ h[c] * scale    ← 从 h 中读取全部历史
```

### 1.3 完整计算流水线

KDA 的 6 个计算步骤形成严格的数据依赖链（每个步骤的输出是下一个的输入）：

```
Step 1: Gate Cumsum
  raw_gate → 激活为 gate → chunk 内前缀和 → log2 转换 → g_cumsum
                                          │
                                          ▼
Step 2: Token Parallel (对角线 Aqk/Akk)
  q, k, beta, g_cumsum → Aqk, Akk（对角线块）
                                          │
                                          ▼
Step 3: Inter Solve (块三对角矩阵求逆)
  Aqk, Akk → Akk_inv（解耦矩阵）
                                          │
          ┌───────────────────────────────┘
          ▼
Step 4: Recompute W/U
  Akk_inv, k, v, beta, g_cumsum → w, u, kg
                                          │
                                          ▼
Step 5: Delta Rule H（状态更新）
  w, u, kg → h（状态矩阵快照）, v_new
                                          │
                                          ▼
Step 6: GLA Output（最终输出融合）
  q, v_new, g_cumsum, Aqk, h → o（最终注意力输出）
```

**本试题选取 Step 1、Step 2、Step 6 作为三道难度递进的编程题：**

| 题目 | 对应步骤 | 核心概念 | 难度 |
|------|---------|---------|------|
| 题目一 | Step 1: Gate Cumsum | 门控激活 + chunk-local 前缀和 | L1 入门 |
| 题目二 | Step 2: Token Parallel | per-token 并行 + gated dot product + 因果掩码 | L2 中级 |
| 题目三 | Step 6: GLA Output | 双路径融合 + 矩阵乘法 + K 维循环 | L3 进阶 |

三题的输出形成数据依赖链：题目一的 `g_cumsum` → 题目二的 `Aqk` → 题目三的 `o`。

---

## 二、编程题

### 2.1 数据约定

所有题目使用以下统一约定：
- `B = 1`（单 batch，简化处理）
- `BT = 64`（chunk 大小，每个 chunk 固定 64 个 token）
- `H`, `K`, `V` 均为题目给定的编译期常量
- 所有 Triton kernel 使用 **1 个 warp (32 threads)**，即 `num_warps=1`
- 输入张量为 bf16 或 fp32，计算过程使用 fp32
- 固定长度序列（不做 VARLEN）

---

### 题目一 (L1)：Gate Chunk Cumsum — 门控激活与块内前缀和


#### 在流水线中的位置

本题对应 **Step 1: Gate Cumsum**。原始门控值 `raw_gate` 来自上游网络输出，未经任何处理。本题负责将其激活为有效的 gate 值，并在每个 chunk 内部做前缀和，为后续所有步骤提供 `g_cumsum`。

#### 题目描述

实现 `gate_chunk_cumsum` kernel，对每个 chunk 独立完成三步操作：

**操作 1：门控激活**

将原始门控值激活为 gate：

```
gate[t, h, k] = -exp(A_log[h]) * softplus(raw_gate[t, h, k] + dt_bias[h, k])
```

其中 `softplus(x) = log(1 + exp(x))`。`-exp(A_log[h])` 是一个负的 head-level 缩放因子，确保 gate < 0，从而 `exp(gate) < 1` 作为后续步骤的衰减因子。`dt_bias` 是每个 head 每个 channel 的可学习偏置。

**操作 2：Chunk-local Cumsum**

在每个 chunk 内部沿时间轴做前缀和。关键约束：**chunk 间不累积**——每个 chunk 的 cumsum 从 0 重新开始。

```
对于 chunk 0 中 channel k 的 64 个 token:
  输入 gate (激活后):  [g0,    g1,    g2,    ..., g63]
  输出 cumsum:        [g0,  g0+g1, g0+g1+g2, ..., g0+g1+...+g63]

对于 chunk 1 中的同一个 channel k (从 0 重新开始):
  输入 gate:          [g64,      g65,      ..., g127]
  输出 cumsum:        [g64,    g64+g65,   ..., g64+g65+...+g127]
```

这样设计是因为后续 chunk-wise attention 中，每个 chunk 独立计算内部衰减，不需要跨 chunk 的全局累积值。

**操作 3：Log2 空间转换**

```
output = cumsum_result * RCP_LN2
```

其中 `RCP_LN2 = 1 / ln(2) ≈ 1.442695`。因为 `ln(x) * (1/ln(2)) = log2(x)`，此操作将 natural-log 空间的值转为 log2 空间。后续 kernel 使用硬件高效的 `exp2()` 而非 `exp()` 来还原衰减因子。

#### 输入输出规格

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `raw_gate` | `[B, T, H, K]` | fp32 | 原始门控值（未激活） |
| `A_log` | `[H]` | fp32 | 每 head 的对数尺度参数 |
| `dt_bias` | `[H, K]` | fp32 | 每 head 每 channel 的偏置 |
| `scale` | scalar | fp32 | 输出缩放因子，传入 `RCP_LN2 = 1.4426950216293335` |
| `output` (返回) | `[B, T, H, K]` | fp32 | 激活后的 gate cumsum（log2 空间） |

编译期常量：`BT=64`（chunk 大小），`BS=32`（K 维度的 tile 大小）。

#### 计算示意

```
每个 CTA 处理一个 [BT, BS] = [64, 32] 的 tile：

     K 维 (BS=32 个 channel)
  ┌──────────────────────────────┐
  │ g[0,0..31]   g[0,32..63]  ...│  ← CTA 负责 t0..t63, k0..k31
  │ g[1,0..31]                   │
  │ ...                          │
  │ g[63,0..31]                  │
  └──────────────────────────────┘
  ↑ 沿时间轴做 cumsum (axis=0)

每个 channel 独立做 cumsum，chunk 边界处从 0 重新开始。
```

#### 参考实现（PyTorch，用于验证）

```python
import torch
import math

RCP_LN2 = 1.4426950216293335

def reference_gate_chunk_cumsum(raw_gate, A_log, dt_bias, chunk_size=64):
    """
    raw_gate: [B, T, H, K]  fp32
    A_log:    [H]           fp32
    dt_bias:  [H, K]        fp32
    Returns:  [B, T, H, K]  fp32
    """
    B, T, H, K = raw_gate.shape
    # Step 1: Gate activation
    gate = -torch.exp(A_log)[None, None, :, None] * torch.nn.functional.softplus(
        raw_gate + dt_bias[None, None, :, :]
    )

    # Step 2: Chunk-local cumsum
    output = torch.zeros_like(gate)
    NT = (T + chunk_size - 1) // chunk_size
    for c in range(NT):
        start = c * chunk_size
        end = min(start + chunk_size, T)
        output[:, start:end] = torch.cumsum(gate[:, start:end], dim=1)

    # Step 3: log2 space conversion
    output = output * RCP_LN2
    return output
```

#### 完成标准

Triton kernel 的输出与 reference 实现之间的 **RMSE < 1e-5**，且满足：
- 每个 chunk 的 cumsum 从 0 开始，跨 chunk 边界不传递
- softplus 对 x >= 20 使用线性近似避免数值溢出
- 输出 dtype 为 fp32

---

### 题目二 (L2)：Token Parallel — 对角线 Aqk/Akk 计算

#### 在流水线中的位置

本题对应 **Step 2: Token Parallel**。上一步输出的 `g_cumsum`（log2 空间的门控累积值）作为本步的输入，用于计算 sub-chunk 内部的 gated dot product，产出 Aqk（注意力权重矩阵的对角线块）和 Akk（key-key 依赖矩阵的对角线块）。这两个矩阵是 Step 3 求逆解耦的输入。

#### 题目描述

实现 `token_parallel` kernel，计算每个 sub-chunk 内的对角线 Aqk 和 Akk 块。

**Token-Parallel 策略**：每个 token 分配一个独立的 CTA。该 CTA 只遍历自己所在 sub-chunk 内的历史 token（j <= i），避免无效计算。

**为什么只算对角线块？** 一个 64×64 的 chunk 矩阵被划分为 4×4 个 16×16 的 sub-chunk 块。非对角线块由其他 kernel 负责。本题只计算 4 个对角线块（D00, D11, D22, D33），每个块大小为 16×16：

```
Chunk Matrix (BT × BT = 64 × 64):
        SC0     SC1     SC2     SC3
      +--------+--------+--------+--------+
SC0   |  D00   | (off)  | (off)  | (off)  |
      +--------+--------+--------+--------+
SC1   |  K10   |  D11   | (off)  | (off)  |    Dnn = 对角线块 ← 本题计算
      +--------+--------+--------+--------+    Knm = 非对角线块
SC2   |  K20   |  K21   |  D22   | (off)  |
      +--------+--------+--------+--------+
SC3   |  K30   |  K31   |  K32   |  D33   |
      +--------+--------+--------+--------+
```

**数学定义：**

对于 token i 及其 sub-chunk 内的历史 token j（j <= i）：

```
gated_k = k[j] * exp2(g[i] - g[j])

Aqk[i, j] = scale * (q[i] · gated_k)       ← j <= i 时计算，否则为 0
Akk[i, j] = k[i] * beta[i] · gated_k       ← j < i  时计算，否则为 0
```

其中 `exp2(g[i] - g[j])` 是 token j 对 i 的门控衰减因子。g 已在 log2 空间，所以用 `exp2` 直接还原。`g[i] >= g[j]`（因为 g 是负值的累积和递减序列），所以 `g[i] - g[j] <= 0`，`exp2(差值) ∈ (0, 1]`。

**Aqk 和 Akk 的区别：**
- Aqk：query 与 gated key 的点积，乘 scale 后作为注意力权重。对角线位置 j==i 时 `exp2(g[i]-g[i]) = 1`，退化为 `q[i] · k[i] * scale`。
- Akk：key 自身（乘 beta）与 gated key 的点积，用于后续构建块三对角矩阵和解耦线性系统。对角线 j==i 时必须为 0（严格上三角）。

#### 输入输出规格

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `q` | `[B, T, H, K]` | bf16 | Query 张量 |
| `k` | `[B, T, H, K]` | bf16 | Key 张量 |
| `g` | `[B, T, H, K]` | fp32 | 题目一输出的 g_cumsum（log2 空间） |
| `beta` | `[B, T, H]` | bf16 | Per-token per-head 权重系数 |
| `scale` | scalar | fp32 | 注意力缩放因子 = `1/sqrt(K)` |
| `Aqk` (输出) | `[B, T, H, BT]` | bf16 | 对角线 Aqk 块，每行填充 sub-chunk 对应列 |
| `Akk` (输出) | `[B, T, H, BC]` | fp32 | 对角线 Akk 块（fp32 保证后续求逆精度） |

编译期常量：`BT=64`（chunk 大小），`BC=16`（sub-chunk 大小），`BK=next_power_of_2(K)`。

#### 索引计算

```
给定全局 token 索引 i_tg:
  i_b = i_tg // T              ← batch 索引
  i_t = i_tg %  T              ← batch 内局部 token 索引

  i_c  = i_t // BT             ← chunk 索引
  i_s  = (i_t % BT) // BC      ← sub-chunk 索引 (0..3)
  i_ts = i_c * BT + i_s * BC   ← sub-chunk 起始 token 索引

  j 遍历范围: i_ts .. min(i_t, i_ts + BC - 1)
```

#### 计算示意

以 Chunk 0, Sub-chunk 1（token 16..31）为例：

```
D11 块内部 (16×16):
     j=16 j=17 j=18 ... j=31
i=16 [ C0    X    X  ...  X  ]  ← CTA_16: j=16  (1 个 pair)
i=17 [ C1   C2    X  ...  X  ]  ← CTA_17: j=16,17 (2 个 pair)
i=18 [ C3   C4   C5  ...  X  ]  ← CTA_18: j=16,17,18 (3 个 pair)
 ...                            ...
i=31 [ ...  ...  ... ...  Cn ]  ← CTA_31: j=16..31 (16 个 pair)

每个 CTA 的工作量随 token 在 sub-chunk 中的位置线性增长 (1..BC 个 pair)。
```

**存储布局：**

```
Aqk [B, T, H, BT]:  写入列 = j % BT    (chunk 内的绝对列位置)
Akk [B, T, H, BC]:  写入列 = j - i_ts  (sub-chunk 内的相对偏移, 0..BC-1)
```

#### 参考实现（PyTorch，用于验证）

```python
def reference_token_parallel(q, k, g, beta, scale, BT=64, BC=16):
    """
    q:    [B, T, H, K]  bf16
    k:    [B, T, H, K]  bf16
    g:    [B, T, H, K]  fp32  (题目一的输出)
    beta: [B, T, H]     bf16
    scale: scalar
    Returns: Aqk [B, T, H, BT], Akk [B, T, H, BC]
    """
    B, T, H, K = q.shape
    q = q.float()
    k = k.float()
    beta = beta.float()

    Aqk = torch.zeros(B, T, H, BT, dtype=torch.float32)
    Akk = torch.zeros(B, T, H, BC, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            for i in range(T):
                i_c = i // BT
                i_s = (i % BT) // BC
                i_ts = i_c * BT + i_s * BC

                for j in range(i_ts, min(i + 1, min(T, i_ts + BC))):
                    gated_k = k[b, j, h] * torch.exp2(g[b, i, h] - g[b, j, h])
                    aqk = scale * torch.dot(q[b, i, h], gated_k)
                    Aqk[b, i, h, j % BT] = aqk

                    if j < i:
                        akk = torch.dot(k[b, i, h] * beta[b, i, h], gated_k)
                        Akk[b, i, h, j - i_ts] = akk

    return Aqk, Akk
```

#### 完成标准

Triton kernel 的输出与 reference 实现之间的误差满足：
- Aqk: **RMSE < 1e-3**
- Akk: **RMSE < 1e-3**

且满足：
- Aqk 在对角线位置 (j == i) 可用，Akk 在对角线位置 (j == i) 必须为 0
- K 维度的点积通过循环分块累加完成（不能一次性加载整个 K）
- q[i] 和 k[i] 在整个内循环中复用，不重复加载

---

### 题目三 (L3)：GLA Output — 双路径最终输出融合

#### 在流水线中的位置

本题对应 **Step 6: GLA Output**，是 KDA 流水线的最后一步。此前的步骤已经完成了：
- Step 1: 计算出 `g_cumsum`（门控累积值）
- Step 2-3: 计算出 `Aqk`（块内注意力权重）和 `Akk_inv`（解耦矩阵）
- Step 4-5: 计算出 `h`（跨块状态矩阵）和 `v_new`（Delta Rule 校正后的 value）

本题将以上所有中间结果融合，产出最终的注意力输出 `o`。

#### 题目描述

实现 `gla_output` kernel，将跨块（cross-chunk）和块内（intra-chunk）两条计算路径融合，产生最终输出。

**为什么要双路径？** Chunkwise Linear Attention 将信息分为两个来源：
- **跨块信息**：chunk c 之前所有 token 的压缩表示，存储在状态矩阵 h 中。从 h 读取信息只需一次矩阵乘法 O(K×V)，不需要回看每个历史 token。
- **块内信息**：当前 chunk 内部的 token 间交互，精确计算（不做压缩），使用 Aqk 注意力矩阵和 v_new。

两条路径互补——跨块路径提供全局上下文（粗糙但覆盖广），块内路径提供局部精确交互（精细但范围小）。

**数学定义：**

```
对于 chunk c 中的 token t:
  o[t] = o_cross[t] + o_intra[t]

  跨块路径: o_cross[t] = (q[t] * exp2(g[t])) @ h[c] * scale
            ↑ q[t] 乘上门控因子 exp2(g[t])，放大当前 token 的有效 query
            ↑ 再与状态矩阵 h[c] (shape [V, K]) 做矩阵乘法
            ↑ 实际计算: [K] @ [K, V]^T = [K] @ [V, K] = [V]
            ↑ 即用 gated query 从跨块记忆中读出历史信息

  块内路径: o_intra[t] = Σ_{j=start..t} Aqk[t, j] * v_new[j]
            ↑ 在当前 chunk 内，用因果注意力权重对 v_new 加权求和
            ↑ Aqk[t, j] 只对 j <= t 有非零值（天然因果）
            ↑ 等价于: Aqk_chunk @ v_new_chunk，加 causal mask
```

#### 输入输出规格

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `q` | `[B, T, H, K]` | bf16 | Query 张量 |
| `v_new` | `[B, T, H, V]` | bf16 | Delta Rule 校正后的 value |
| `g` | `[B, T, H, K]` | fp32 | 题目一输出的 g_cumsum（log2 空间） |
| `Aqk` | `[B, T, H, BT]` | bf16 | 题目二输出的 Aqk 注意力矩阵 |
| `h` | `[B, NT, H, V, K]` | bf16 | 每个 chunk 的记忆状态快照（V 行 K 列） |
| `scale` | scalar | fp32 | 注意力缩放因子 = `1/sqrt(K)` |
| `o` (输出) | `[B, T, H, V]` | bf16 | 最终注意力输出 |

编译期常量：`BT=64`，`BK`、`BV`（K 和 V 维度的 tile 大小，取为 32）。

#### 计算示意

```
                  ┌─────────────────────┐
                  │   输入：Chunk c      │
                  │   BT=64, BK=BV=32   │
                  └─────────┬───────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
   ┌──────────────────┐          ┌──────────────────┐
   │  跨块路径 (K 循环) │          │  块内路径          │
   │                  │          │                  │
   │  for each BK:   │          │  b_A = load      │
   │    b_qg = load   │          │    Aqk[BT, BT]   │
   │    q[BT, BK] *   │          │  m_s = causal    │
   │    exp2(g[BT,BK])│          │    mask (下三角)   │
   │    * scale       │          │  b_A *= m_s      │
   │                  │          │                  │
   │    b_h = load    │          │  b_v = load      │
   │    h[V, K] →     │          │    v_new[BT, BV] │
   │    transpose     │          │                  │
   │    → [BK, BV]    │          │  b_o2 = b_A @    │
   │                  │          │    b_v           │
   │    b_o1 +=       │          │  → [BT, BV]      │
   │    b_qg @ b_h^T  │          └────────┬─────────┘
   │  → [BT, BV]      │                   │
   └────────┬─────────┘                   │
            │                             │
            └──────────┬──────────────────┘
                       │
                       ▼
               b_o = b_o1 + b_o2
                   [BT, BV]
                       │
                       ▼
               写入 o[chunk_start:chunk_end, V_tile]
```

**跨块路径详解：**

跨块路径的特征是 K 维度循环——h 的大小为 `[V, K]`，V 方向可以一次性加载 BV=32 个元素，但 K 方向需要分块遍历。每次循环：
1. 加载 `q[BT, BK]` 和 `g[BT, BK]`（当前 K 分块的 64 个 token × 32 个 channel）
2. 计算 gated query: `b_qg = q * exp2(g) * scale`
3. 加载 `h[BV, BK]` 的一个 tile 并转置为 `[BK, BV]`
4. 矩阵乘法累加: `b_o1 += b_qg @ b_h^T`（`[BT, BK] @ [BK, BV] = [BT, BV]`）

**块内路径详解：**

1. 加载 `Aqk[BT, BT]`（当前 chunk 的注意力权重，可能不满 BT 需 boundary check）
2. 用因果掩码（下三角）将上三角置零（虽然 Aqk 本身上三角为 0，但加载了整个 BT×BT 块）
3. 加载 `v_new[BT, BV]`（当前 chunk 的校正后 value）
4. 矩阵乘法: `b_o2 = b_A @ b_v`（`[BT, BT] @ [BT, BV] = [BT, BV]`）

#### 参考实现（PyTorch，用于验证）

```python
def reference_gla_output(q, v_new, g, Aqk, h, scale, BT=64):
    """
    q:     [B, T, H, K]  bf16
    v_new: [B, T, H, V]  bf16
    g:     [B, T, H, K]  fp32
    Aqk:   [B, T, H, BT] bf16
    h:     [B, NT, H, V, K] bf16  (V rows, K cols)
    scale: scalar
    Returns: o [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    NT = h.shape[1]
    q = q.float()
    v_new = v_new.float()
    Aqk = Aqk.float()
    h = h.float()

    o = torch.zeros(B, T, H, V, dtype=torch.float32)
    causal_mask = torch.tril(torch.ones(BT, BT))

    for b in range(B):
        for head in range(H):
            for c in range(NT):
                start = c * BT
                end = min(start + BT, T)
                n_tokens = end - start

                # Cross-chunk path
                h_c = h[b, c, head]  # [V, K]
                for t in range(n_tokens):
                    idx = start + t
                    q_gated = q[b, idx, head] * torch.exp2(g[b, idx, head]) * scale
                    o[b, idx, head] += q_gated @ h_c.T  # [K] @ [K, V] = [V]

                # Intra-chunk path
                A_chunk = Aqk[b, start:end, head, :n_tokens]  # [n, BT] -> [n, n]
                v_chunk = v_new[b, start:end, head]           # [n, V]
                mask = causal_mask[:n_tokens, :n_tokens]
                o[b, start:end, head] += (A_chunk * mask) @ v_chunk

    return o
```

#### 完成标准

Triton kernel 的输出与 reference 实现之间的 **RMSE < 1e-3**，且满足：
- 跨块路径正确实现 K 维循环（不能假设 K 能一次性加载完）
- 块内路径正确使用因果掩码（下三角有效，上三角为 0）
- 两条路径独立计算后累加，而非串行覆盖
- 使用 `tl.dot` 做矩阵乘法

---

## 三、附录

### A. 数据生成脚本

以下脚本生成完整的测试数据，覆盖三道题目：

```python
import torch
import math

def generate_test_data(T=128, H=2, K=64, V=64, seed=42):
    """生成完整测试数据集，适配三道题目。"""
    torch.manual_seed(seed)
    B = 1
    BT = 64

    # 模型参数
    A_log = torch.randn(H) * 0.5
    dt_bias = torch.randn(H, K) * 0.1

    # 题目一输入
    raw_gate = torch.randn(B, T, H, K)

    # 题目二输入
    q = torch.randn(B, T, H, K) * 0.1
    k = torch.randn(B, T, H, K) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H))

    # 题目一输出 = 题目二的 g 输入
    RCP_LN2 = 1.4426950216293335
    gate = -torch.exp(A_log)[None, None, :, None] * torch.nn.functional.softplus(
        raw_gate + dt_bias[None, None, :, :]
    )
    g_cumsum = torch.zeros_like(gate)
    NT = (T + BT - 1) // BT
    for c in range(NT):
        start = c * BT
        end = min(start + BT, T)
        g_cumsum[:, start:end] = torch.cumsum(gate[:, start:end], dim=1)
    g_cumsum *= RCP_LN2

    # 题目三输入 — v_new 和 h 用随机数据模拟
    v_new = torch.randn(B, T, H, V) * 0.1
    h = torch.randn(B, NT, H, V, K) * 0.01

    scale = K ** -0.5

    return {
        'raw_gate': raw_gate,
        'A_log': A_log,
        'dt_bias': dt_bias,
        'q': q,
        'k': k,
        'beta': beta,
        'g_cumsum': g_cumsum,  # 题目一输出 / 题目二输入
        'v_new': v_new,
        'h': h,
        'scale': scale,
        'BT': BT,
    }

# 使用示例
data = generate_test_data()

# 题目一
from reference import reference_gate_chunk_cumsum
g_out_ref = reference_gate_chunk_cumsum(data['raw_gate'], data['A_log'], data['dt_bias'])
g_out_triton = gate_chunk_cumsum(data['raw_gate'], data['A_log'], data['dt_bias'])
assert torch.allclose(g_out_ref, g_out_triton, rtol=1e-5), "题目一 failed"

# 题目二
Aqk_ref, Akk_ref = reference_token_parallel(data['q'], data['k'], data['g_cumsum'],
                                              data['beta'], data['scale'])
Aqk_tri, Akk_tri = token_parallel(data['q'], data['k'], data['g_cumsum'],
                                    data['beta'], data['scale'])
assert torch.allclose(Aqk_ref, Aqk_tri.float(), atol=1e-3), "题目二 Aqk failed"
assert torch.allclose(Akk_ref, Akk_tri.float(), atol=1e-3), "题目二 Akk failed"

# 题目三
o_ref = reference_gla_output(data['q'], data['v_new'], data['g_cumsum'],
                               Aqk_tri, data['h'], data['scale'])
o_tri = gla_output(data['q'], data['v_new'], data['g_cumsum'],
                    Aqk_tri, data['h'], data['scale'])
assert torch.allclose(o_ref, o_tri.float(), atol=1e-3), "题目三 failed"

print("All tests passed!")
```

