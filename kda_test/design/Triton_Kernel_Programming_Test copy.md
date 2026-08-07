# Triton Kernel 编程试题：Chunkwise Linear Attention

## 一、背景介绍

### 1.1 问题：长文本推理中的注意力计算瓶颈

在 Transformer 架构中，标准 Softmax Attention 的核心操作是每个 token 与所有历史 token 计算相似度：

```
对于位置 t 的 token:
  output[t] = softmax(q[t] · k[0], q[t] · k[1], ..., q[t] · k[T-1]) @ v
```

当序列长度 T=128K 时，注意力矩阵 `T × T` 约 160 亿个元素，计算复杂度 O(T²)，无法在单芯片上完成。

### 1.2 解决方案：Chunkwise Linear Attention

KDA（Kernelized Delta Attention）采用分块线性注意力策略：把 T 个 token 按 `chunk_size=64` 切割，**块内并行（矩阵乘法）、块间串行（状态传递）**。核心计算流程分为 6 个 kernel：

```
raw_gate ─[K1:Gate Cumsum]──→ g_cumsum ─[K2:Token Parallel]──→ Aqk,Akk ─[K3:Inter Solve]──→ Akk_inv
                                                                                              │
                                                    ┌─────────────────────────────────────────┘
                                                    ▼
                              ─[K4:Recompute W/U]──→ w,u,kg ─[K5:Delta Rule H]──→ h,v_new ─[K6:GLA Output]──→ o
```

每个 kernel 的输出是下一个 kernel 的输入，形成严格的流水线。

### 1.3 核心概念速览

**Gate（门控衰减）**：每个 token 在每个 key channel 上有独立的遗忘速率。

```
gate[t,k] = -exp(A_log) * softplus(raw_gate[t,k] + dt_bias[k])
```

`gate < 0` 保证 `exp(gate) < 1`，作为衰减因子控制 state 中旧信息的遗忘速度。

**Delta Rule**：只把"新信息"写入状态矩阵，避免冗余存储。

```
predict = k[t] @ state        ← 用当前 key 查询历史状态，预测 value
delta   = v[t] - predict       ← 只保留预测不到的部分
state   = state * decay + k[t]^T @ delta   ← 衰减旧信息 + 写入新信息
```

**Chunk 划分**：T 个 token 切为 NT = ceil(T/64) 个 chunk，每个 chunk 再切为 4 个 sub-chunk（各 16 token）。

```
T tokens:  |----Chunk 0 (64 tokens)----|----Chunk 1 (64 tokens)----|...
           |SC0(16)|SC1(16)|SC2(16)|SC3(16)|
```

---

## 二、编程题

下面三道题目分别对应上述流水线中三个独立、自包含、难度递进的 kernel。每道题都提供完整的输入输出规格和 PyTorch reference 实现。

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

**预计用时**：30-45 分钟
**预计代码量**：~50 行 Triton kernel + ~20 行 Python wrapper

#### 题目描述

实现 `gate_chunk_cumsum` kernel，对每个 chunk 独立完成两步操作：

1. **门控激活**：将原始门控值 `raw_gate` 激活为 gate
   ```
   gate[t,h,k] = -exp(A_log[h]) * softplus(raw_gate[t,h,k] + dt_bias[h,k])
   ```
   其中 `softplus(x) = log(1 + exp(x))`，当 x >= 20 时可用 x 近似。

2. **Chunk-local Cumsum**：在每个 chunk 内部沿时间轴做前缀和，**chunk 间不累积**（每个 chunk 从 0 重新开始）。

3. **输出缩放**：将结果乘以 `RCP_LN2 = 1.442695...`，将 natural-log 空间的值转换为 log2 空间。

#### 输入输出规格

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `raw_gate` | `[B, T, H, K]` | fp32 | 原始门控值（未激活） |
| `A_log` | `[H]` | fp32 | 每 head 的对数尺度参数 |
| `dt_bias` | `[H, K]` | fp32 | 每 head 每 channel 的偏置（本题提供为 [H,K] 而非 [H*K]） |
| `scale` | scalar | fp32 | 输出缩放因子，传入 `RCP_LN2 = 1.4426950216293335` |
| `output` (返回) | `[B, T, H, K]` | fp32 | 激活后的 gate cumsum，log2 空间 |

编译期常量：`BT=64`（chunk 大小），`BS=32`（K 维度的 tile 大小）。

#### Grid 设计要求

```
Grid = (cdiv(K, BS), cdiv(T, BT), B * H)

program_id(0) = i_s   → K 维度分块索引
program_id(1) = i_t   → chunk 索引
program_id(2) = i_bh  → (batch, head) 联合索引
```

每个 CTA 处理一个 `[BT, BS]` 大小的 tile（64 个时间步 × 32 个通道）。

#### 计算示意

```
对于 chunk 0 中的一个 channel k:

  输入 gate (激活后):  [g0, g1, g2, g3, ..., g63]
  输出 cumsum:        [g0, g0+g1, g0+g1+g2, ..., g0+g1+...+g63]

对于 chunk 1 中的同一个 channel k (从 0 重新开始):

  输入 gate:          [g64, g65, ..., g127]
  输出 cumsum:        [g64, g64+g65, ..., g64+g65+...+g127]
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

#### 评分标准

| 维度 | 分值 | 要求 |
|------|------|------|
| 正确性 | 50% | 输出与 reference 的 RMSE < 1e-5 |
| Grid 设计 | 15% | 正确使用 3D grid，索引映射正确 |
| 内存访问 | 15% | 使用 `tl.make_block_ptr` 加载/存储 tile，正确设置 boundary_check |
| Chunk 边界 | 10% | 每个 chunk 的 cumsum 从 0 开始，跨 chunk 不传递 |
| 数值稳定性 | 10% | softplus 对 x>=20 使用线性近似，避免 exp 溢出 |

#### 提示

- Triton 提供 `tl.cumsum(tensor, axis=0)` 内置函数
- 使用 `tl.make_block_ptr` 创建 2D 块指针，shape 为 `(T, K)`，strides 为 `(H*K, 1)`
- `A_log` 是单一 head 的标量，需要用 `tl.load(A_log + i_h)` 加载
- 输出 dtype 为 fp32

---

### 题目二 (L2)：Token Parallel — 对角线 Aqk/Akk 计算

**预计用时**：45-60 分钟
**预计代码量**：~80 行 Triton kernel + ~30 行 Python wrapper

#### 题目描述

实现 `token_parallel` kernel，基于 Token-Parallel 策略计算每个 sub-chunk 内的对角线 Aqk 和 Akk 块。

**核心设计**：每个 token 分配一个独立的 CTA。该 CTA 遍历自己所在 sub-chunk 内的历史 token，计算与该 token 的 gated dot product。

#### 数学定义

对于 token i 及其 sub-chunk 内的 token j（j <= i）：

```
gated_k = k[j] * exp2(g[i] - g[j])        ← j 的 key 用门控差衰减

Aqk[i,j] = scale * (q[i] · gated_k)       ← 注意力权重 (j <= i, 否则 0)
Akk[i,j] = k[i]*beta[i] · gated_k         ← key-key 依赖 (j < i,  否则 0)
```

其中 `g` 是**题目一输出的 g_cumsum**（已在 log2 空间），`exp2(g[i]-g[j])` 计算 token j 对 i 的衰减因子。

#### 输入输出规格

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `q` | `[B, T, H, K]` | bf16 | Query 张量 |
| `k` | `[B, T, H, K]` | bf16 | Key 张量 |
| `g` | `[B, T, H, K]` | fp32 | 题目一输出的 g_cumsum（log2 空间） |
| `beta` | `[B, T, H]` | bf16 | Per-token per-head 权重系数 |
| `scale` | scalar | fp32 | 注意力缩放因子 = `1/sqrt(K)` |
| `Aqk` (输出) | `[B, T, H, BT]` | bf16 | 对角线 Aqk 块，每行填充 sub-chunk 对应列 |
| `Akk` (输出) | `[B, T, H, BC]` | fp32 | 对角线 Akk 块，每行填充 sub-chunk 对应列 |

编译期常量：`BT=64`（chunk 大小），`BC=16`（sub-chunk 大小），`BK=next_power_of_2(K)`。

#### Grid 设计要求

```
Grid = (B * T, cdiv(H, 1))

program_id(0) = i_tg  → 全局 token 索引 (0 .. B*T-1)
program_id(1) = i_hg  → head group 索引 (本题 BH=1，即 0 .. H-1)
```

每个 CTA 处理一个 `(token, head)` 对，确定该 token 所属的 sub-chunk，遍历 sub-chunk 内所有 `j <= token_i` 的历史 token。

#### 索引计算

```
给定全局 token 索引 i_tg:
  i_b = i_tg // T     ← batch 索引
  i_t = i_tg %  T     ← batch 内局部 token 索引 (0 .. T-1)

  i_c  = i_t // BT          ← chunk 索引 (0 .. NT-1)
  i_s  = (i_t % BT) // BC   ← sub-chunk 索引 (0 .. 3)
  i_ts = i_c * BT + i_s * BC  ← sub-chunk 起始 token 索引

  j 遍历范围: i_ts .. min(i_t, i_ts + BC - 1)
```

#### 计算示意

```
Chunk 0, Sub-chunk 1 (token 16-31):

  token 16: j=16            → Aqk[16,16], Akk[16,16]=0 (j==i)
  token 17: j=16,17         → Aqk[17,16..17], Akk[17,16..16]
  token 18: j=16,17,18      → Aqk[18,16..18], Akk[18,16..17]
  ...
  token 31: j=16..31        → Aqk[31,16..31], Akk[31,16..30]

存储布局:
  Aqk [B,T,H,BT]: 每行只写 sub-chunk 对应 BT 范围内的列
    Aqk[i_t, j % BT]  ← 写目标列
  Akk [B,T,H,BC]: 每行写 sub-chunk 内的偏移列
    Akk[i_t, j - i_ts]  ← 写目标列 (0..BC-1)
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

#### 评分标准

| 维度 | 分值 | 要求 |
|------|------|------|
| 正确性 | 45% | Aqk RMSE < 1e-3, Akk RMSE < 1e-3 |
| Token-Parallel 设计 | 15% | 每个 token 一个 CTA，Grid = (B*T, H) |
| 内循环遍历 | 15% | 正确遍历 sub-chunk 内 j<=i 的 token |
| 因果掩码 | 10% | Aqk[j==i] 允许，Akk[j==i] 为 0 |
| 内存布局 | 10% | Aqk 写入 j%BT 列，Akk 写入 (j-i_ts) 列 |
| 数值稳定性 | 5% | exp2 的输入范围控制 |

#### 提示

- `g[i] - g[j]` 可能较大，注意 exp2 的数值范围（Triton 的 exp2 内部有 clamp 处理）
- K 维度的循环使用 `for i_k in range(tl.cdiv(K, BK))`，每次加载 `[1, BK]` 的 q/k/g tile（1 个 token 的 BH 个 head、BK 个 channel）
- q[i] 和 k[i] 在整个内循环中不变，可以提前加载

---

### 题目三 (L3)：GLA Output — 双路径最终输出融合

**预计用时**：60-90 分钟
**预计代码量**：~100 行 Triton kernel + ~35 行 Python wrapper

#### 题目描述

实现 `gla_output` kernel，将跨块（cross-chunk）和块内（intra-chunk）两条计算路径融合，产生最终的注意力输出。

#### 数学定义

对于 chunk c 中的 token t：

```
o[t] = o_cross[t] + o_intra[t]

跨块路径：o_cross = (q[t] * exp2(g[t])) @ h[c] * scale
        ↑ 用 gated query 从"记忆矩阵" h[c] 中读取所有历史信息

块内路径：o_intra = Aqk[t, :t+1] @ v_new[chunk_start : t+1]
        ↑ 用因果注意力矩阵计算当前 chunk 内的精确交互
```

其中 `h[c]` 是一个 `[V, K]` 矩阵（V 行 K 列），把 chunk c 之前所有历史 token 的信息压缩在里面。——注：此处 `h` 的布局为 `[V, K]`（与 KDA 代码中的实际存储一致，见附录说明）。

#### 输入输出规格

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `q` | `[B, T, H, K]` | bf16 | Query 张量 |
| `v_new` | `[B, T, H, V]` | bf16 | 校正后的 value（来自 Delta Rule） |
| `g` | `[B, T, H, K]` | fp32 | g_cumsum（log2 空间） |
| `Aqk` | `[B, T, H, BT]` | bf16 | 题目二输出的 Aqk 注意力矩阵 |
| `h` | `[B, NT, H, V, K]` | bf16 | 每个 chunk 的记忆状态快照（V 行 K 列） |
| `scale` | scalar | fp32 | 注意力缩放因子 = `1/sqrt(K)` |
| `o` (输出) | `[B, T, H, V]` | bf16 | 最终注意力输出 |

编译期常量：`BT=64`（chunk 大小），`BK`、`BV`（K 和 V 维度的 tile 大小，通常为 32）。

#### Grid 设计要求

```
Grid = (cdiv(V, BV), NT, B * H)

program_id(0) = i_v  → V 维度分块索引
program_id(1) = i_t  → chunk 索引
program_id(2) = i_bh → (batch, head) 联合索引
```

每个 CTA 处理一个 `(chunk, head, V-tile)`，对 chunk 内的 64 个 token 同时计算一个 V 分块的输出。

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
   │  for each BK:   │          │  b_A = load Aqk  │
   │    b_qg = load   │          │    [BT, BT]      │
   │    q[BT,BK] *    │          │  m_s = causal    │
   │    exp2(g[BT,BK])│          │    mask (下三角)   │
   │    * scale       │          │  b_A *= m_s      │
   │                  │          │                  │
   │    b_h = load    │          │  b_v = load      │
   │    h[V,K]^T      │          │    v_new[BT,BV]  │
   │    → [BK,BV]     │          │                  │
   │                  │          │  b_o2 = b_A @    │
   │    b_o1 +=       │          │    b_v [BT,BV]   │
   │    b_qg @ b_h^T  │          │  → [BT,BV]       │
   │  → [BT,BV]       │          └────────┬─────────┘
   └────────┬─────────┘                   │
            │                             │
            └──────────┬──────────────────┘
                       │
                       ▼
               b_o = b_o1 + b_o2
                   [BT, BV]
                       │
                       ▼
               写入 o[BT, V_tile]
```

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
                    o[b, idx, head] += q_gated @ h_c.T  # [K] @ [K,V] = [V]

                # Intra-chunk path
                A_chunk = Aqk[b, start:end, head, :n_tokens]  # [n, BT] -> [n, n]
                v_chunk = v_new[b, start:end, head]           # [n, V]
                mask = causal_mask[:n_tokens, :n_tokens]
                o[b, start:end, head] += (A_chunk * mask) @ v_chunk

    return o
```

#### 评分标准

| 维度 | 分值 | 要求 |
|------|------|------|
| 正确性 | 40% | 输出与 reference 的 RMSE < 1e-3 |
| 3D Grid 设计 | 10% | cdiv(V,BV) × NT × B*H，维度分配合理 |
| 跨块路径 | 15% | 正确实现 K 维循环 + `q*exp2(g)*scale @ h^T` |
| 块内路径 | 15% | 正确实现 causal mask + `Aqk @ v_new` |
| 双路融合 | 10% | 两条路径同时计算并加到同一个输出 |
| 内存访问 | 10% | block_ptr 的 shape/stride/order 设置正确 |

#### 提示

- h 的布局为 `h[NT, H, V, K]`，加载时以 V 为行、K 为列，跨块计算需要 h 的转置 `h^T[K,V]`
- Aqk 每行只有前 BT 列有数据（对应 chunk 内的 64 列），加载 `[BT, BT]` 块时注意 `boundary_check`
- causal mask: `m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]`
- K 维度的循环要放在跨块路径，因为 h 需要按 K 分块加载
- 跨块部分先累加完，再加块内部分，最后一次性 store

---

## 三、附录

### A. 数据生成脚本

考生可用以下脚本生成测试数据，验证自己的实现：

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

### B. `h` 矩阵布局说明

`h[B, NT, H, V, K]` 的最后一维是 K，倒数第二维是 V。在代码中加载：

```python
# block_ptr for h[nt, head, :, :]  → shape=(V, K), strides=(K, 1)
p_h = tl.make_block_ptr(
    h + (i_tg * H + i_h) * V * K,
    (V, K),           # shape
    (K, 1),           # strides: 按行遍历 V 时步长为 K, 按列遍历 K 时步长为 1
    (i_v * BV, i_k * BK),  # offset
    (BV, BK),         # block shape
    (1, 0),           # order: K 连续
)
# b_h: [BV, BK]  = h 的一个 tile
# 跨块计算使用 tl.trans(b_h) 转置为 [BK, BV] 后与 q_gated 做 dot
```

这样在论文推导中用 `[K, V]` 概念布局便于理解 `q[K] @ state[K,V] = [V]`，而代码中按 `[V, K]` 存储，因为 K==V 时两者等价（都约等于 64×64 方阵的转置）。

### C. Triton API 速查

```python
# 程序索引
pid = tl.program_id(axis)         # 0/1/2

# Block pointer
p = tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)
data = tl.load(p, boundary_check=(axis0, axis1), padding_option="zero")
tl.store(p, data, boundary_check=(axis0, axis1))

# 数学运算
tl.exp(x), tl.exp2(x)             # 指数 / 2的幂指数
tl.cumsum(x, axis=0)              # 前缀和
tl.dot(a, b)                      # 矩阵乘法 [M,K] @ [K,N] → [M,N]
tl.trans(x)                       # 转置
tl.sum(x, axis=1)                 # 沿轴求和
tl.where(cond, a, b)              # 条件选择

# 数组构造
tl.arange(0, N)                   # [0, 1, ..., N-1]
tl.zeros([M, N], dtype=tl.float32)
tl.full([M, N], value, dtype=tl.int32)

# 常量类型
tl.constexpr                     # 编译期常量标记
```

### D. 难度递进关系

```
题目一 (L1)                    题目二 (L2)                      题目三 (L3)
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│ Gate Cumsum     │      │ Token Parallel       │      │ GLA Output          │
│                 │      │                     │      │                     │
│ · 2D tile       │      │ · per-token CTA     │      │ · 3D grid           │
│ · tl.cumsum     │  ──→ │ · causal inner loop │ ──→  │ · dual-path fusion  │
│ · softplus      │      │ · exp2 gating       │      │ · tl.dot matmul     │
│ · block_ptr     │      │ · triangle masking  │      │ · K-dim loop        │
│                 │      │                     │      │ · tl.trans          │
│ 概念: 门控+前缀和 │      │ 概念: Token 级并行    │      │ 概念: 双路融合输出    │
└─────────────────┘      └─────────────────────┘      └─────────────────────┘
      ~50 行                    ~80 行                        ~100 行
      入门级                     中级                         进阶级
```

三道题的输出形成严格的数据依赖链条：题目一的 `g_cumsum` → 题目二的 `Aqk` → 题目三的 `o`。
