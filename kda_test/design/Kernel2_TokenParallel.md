# Kernel 2 核函数设计文档: Token Parallel (Diagonal Aqk/Akk)

## 1. 概述

`chunk_kda_fwd_kernel_intra_token_parallel` 是 KDA (Kernelized Delta Attention) intra-chunk 阶段的 Token Parallel 实现，用于计算每个 sub-chunk 内的对角线 Aqk 和 Akk 块。

源文件: `python/sglang/kernels/ops/attention/fla/chunk_intra_token_parallel.py`

**核心设计理念**: 每个 token 分配一个独立的 CTA (Cooperative Thread Array)，在自己所属的 sub-chunk 内向前遍历历史 token，计算 token 级别的 gated dot product。这种 Token Parallel 策略实现了最大粒度的并行，尤其适合小 batch 场景和变长序列 (VARLEN)，避免了在 padding token 上浪费计算资源。

---

## 2. 输入输出定义

| 参数 | 形状/类型 | 方向 | 说明 |
|------|-----------|------|------|
| `q` | `[B, T, H, K]` | 输入 | Query 张量 |
| `k` | `[B, T, H, K]` | 输入 | Key 张量 |
| `g` | `[B, T, H, K]` | 输入 | Gate 的 chunk-local cumsum 结果 (log2 空间) |
| `beta` | `[B, T, H]` | 输入 | Per-token per-head 的 scalar 权重系数 |
| `Aqk` | `[B, T, H, BT]` | 输出 | Diagonal 区域的 Aqk 结果，写入指定 sub-chunk columns 内 |
| `Akk` | `[B, T, H, BC]` | 输出 | Diagonal 区域的 Akk 结果 (fp32)，写入当前 token 对应 row |
| `scale` | scalar | 输入 | Attention scale 因子，通常为 `K^{-0.5}` |
| `cu_seqlens` | `[N+1]` or None | 输入 | 变长序列累积长度，None 表示定长 |
| `N` | int | 输入 | 变长模式下的序列数 / 定长模式下的 batch size |
| `T` | int (constexpr) | 输入 | 序列长度 (定长模式) 或 padded 最大长度 (变长模式) |
| `H` | constexpr | 输入 | Head 数量 |
| `K` | constexpr | 输入 | 每个 head 的维度 |
| `BT` | constexpr | 输入 | Chunk 大小 (默认 64) |
| `BC` | constexpr | 输入 | Sub-chunk 大小 (默认 16) |
| `BK` | constexpr | 输入 | K 维度的 block size，值为 `next_power_of_2(K)` |
| `BH` | constexpr | 输入 | 每个 CTA 处理的 head 数量，固定为 1 |

**注意**: BT=64, BC=16 意味着每个 chunk 被划分为 4 个 sub-chunk (NC = BT/BC = 4)。

---

## 3. 分核并行策略

### 3.1 Grid 布局

```
Grid = (B * T, cdiv(H, BH))  其中 BH = 1
       +---------------+-----------+
       | program_id(0) | program_id(1) |
       +===============+===============+
       | token index   | head group     |
       +---------------+---------------+
       | i_tg          | i_hg           |
       +---------------+---------------+

总 CTA 数 = B * T * cdiv(H, 1) = B * T * H
(每个 (batch, token, head) 三元组分配一个 CTA)
```

### 3.2 Token-Parallel 概念图

```
Sequence of length T (e.g. T=256), BT=64, BC=16:

Token:    0  1 ... 15 | 16 17 ... 31 | 32 33 ... 47 | 48 49 ... 63 | ... 255
          +-----------+--------------+--------------+--------------+-----+
          | Sub-chunk | Sub-chunk    | Sub-chunk    | Sub-chunk    | ... |
          | SC0       | SC1          | SC2          | SC3          |     |
          +-----------+--------------+--------------+--------------+-----+
          |<------------------ Chunk 0 (BT=64) ------------------->|

每个 token i 的 CTA 独立工作, 在自己的 sub-chunk 内:
  - 定位 sub-chunk 起始位置 i_ts
  - 遍历 sub-chunk 内 j = i_ts, i_ts+1, ..., min(i, i_ts+BC-1)
  - 计算 (i, j) 的 gated dot product

Token-parallel 示意:
   CTA_i:  token[0] 的 CTA 遍历 SC0 中 j=0
   CTA_j:  token[1] 的 CTA 遍历 SC0 中 j=0..1
   CTA_k:  token[2] 的 CTA 遍历 SC0 中 j=0..2
   ...
   CTA_m:  token[15] 的 CTA 遍历 SC0 中 j=0..15
   CTA_n:  token[16] 的 CTA 遍历 SC1 中 j=16
   ...

每个 CTA 自己的工作负载随 token 在 sub-chunk 中的位置线性增长:
  - Sub-chunk 内第 pos 个 token 需要计算 pos+1 个 (i,j) 对
  - 最坏情况 (pos=BC-1): 需要计算 BC 个 (i,j) 对
```

### 3.3 Chunk/Sub-chunk/Token 三维索引映射

```
输入: i_tg = program_id(0)  (全局 token 索引: 0 .. B*T-1)

定长模式:
  bos = (i_tg // T) * T     -- batch 内起始 token 偏移
  i_t = i_tg % T            -- batch 内局部 token 索引

变长模式 (IS_VARLEN=True):
  二分查找 cu_seqlens 定位 batch index i_n
  bos = cu_seqlens[i_n]
  T   = cu_seqlens[i_n+1] - bos  (本序列的实际长度)
  i_t = i_tg - bos           -- 序列内局部 token 索引

Sub-chunk 定位:
  i_c  = i_t // BT           -- chunk 索引 (0, 1, 2, ...)
  i_s  = (i_t % BT) // BC    -- sub-chunk 索引 (0, 1, 2, 3)
  i_tc = i_c * BT            -- chunk 起始 token
  i_ts = i_tc + i_s * BC     -- sub-chunk 起始 token

以 T=256, BT=64, BC=16 为例, token 50 的定位:
  i_c  = 50 // 64 = 0       --> Chunk 0
  i_s  = (50 % 64) // 16 = 3 --> Sub-chunk 3
  i_tc = 0 * 64 = 0
  i_ts = 0 + 3 * 16 = 48    --> Sub-chunk 起始于 token 48

Token 50 的 CTA 会遍历 j = 48, 49, 50 (同 sub-chunk 内所有先前 token)
```

---

## 4. 计算思路

### 4.1 Token-to-Subchunk 映射

```
Time Axis:  -------------------------------------------------------------->
            0              15 16             31 32             47 48             63
            |<--- SC0 ---->| |<--- SC1 ---->| |<--- SC2 ---->| |<--- SC3 ---->|
            |<--------------------------- Chunk 0 (BT=64) ------------------->|

Chunk Matrix (BT x BT, 每个 chunk 独立):

          SC0      SC1      SC2      SC3
        +--------+--------+--------+--------+
  SC0   |  D00   | (off)  | (off)  | (off)  |
        +--------+--------+--------+--------+
  SC1   |  K10   |  D11   | (off)  | (off)  |    Dnn = 对角线块 (本 kernel 计算)
        +--------+--------+--------+--------+    Knm = 非对角线块
  SC2   |  K20   |  K21   |  D22   | (off)  |
        +--------+--------+--------+--------+
  SC3   |  K30   |  K31   |  K32   |  D33   |
        +--------+--------+--------+--------+

本 kernel 只计算 D00, D11, D22, D33 四个对角线 sub-chunk 块。
每个 Dnn 块大小为 BC x BC = 16 x 16。

Dnn 块内部 (Token Parallel 分工):
  以 SC0 = D00 为例 (token 0..15):

      j=0  j=1  j=2  ... j=15
     +----+----+----+--+----+
i=0  | C0 | X  | X  |..| X  |  <- CTA_0 计算 token 0 vs j=0 (1个pair)
     +----+----+----+--+----+
i=1  | C1 | C2 | X  |..| X  |  <- CTA_1 计算 token 1 vs j=0,1 (2个pair)
     +----+----+----+--+----+
i=2  | C3 | C4 | C5 |..| X  |  <- CTA_2 计算 token 2 vs j=0,1,2 (3个pair)
     +----+----+----+--+----+
     ...                          ...
     +----+----+----+--+----+
i=15 | .. | .. | .. |..| Cn |  <- CTA_15 计算 token 15 vs j=0..15 (16个pair)
     +----+----+----+--+----+

  每个 CTA_i 负责填满第 i 行中 j <= i 的列。
  Aqk: j <= i (上三角含对角线)     -- 用 for j=range(.., min(i_t+1, ..))
  Akk: j < i  (严格上三角不含对角线) -- 用 tl.where(j < i_t, 1.0, 0.0)
```

### 4.2 Inner Loop: 遍历同一 Sub-chunk 的先前 Token

```python
for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
```

循环范围分析:

| Token 位置 | i_ts | 循环范围 | 迭代次数 |
|-----------|------|---------|---------|
| sub-chunk 第 0 个 (pos=0) | i_ts | `[i_ts, i_t]` | 1 |
| sub-chunk 第 1 个 (pos=1) | i_ts | `[i_ts, i_t]` | 2 |
| ... | ... | ... | ... |
| sub-chunk 最后个 (pos=BC-1) | i_ts | `[i_ts, i_ts+BC-1]` | BC (16) |

循环内的操作 (对每个 j):

```
Step G:  Gate 衰减因子计算
  exp2(g[i] - g[j])     -- g[i] 是当前 token i 的 gate cumsum
                             g[j] 是历史 token j 的 gate cumsum
                             差值的 exp2 实现了指数衰减

Step K:  加载 k[j] 和 g[j]
  p_kj = block_ptr(k + j*H*K, ...)    --> b_kj: [BH, BK]
  p_gj = block_ptr(g + j*H*K, ...)    --> b_gj: [BH, BK]

Step G:  计算 gated key product
  b_kgj = b_kj * exp2(b_g - b_gj)    --> [BH, BK]
  其中:
    b_g  = g[i]  [BH, BK]  (当前 token 的 gate, 每次循环复用)
    b_gj = g[j]  [BH, BK]  (历史 token 的 gate)

Step Q:  Aqk dot product
  b_Aqk = sum(b_q * b_kgj, axis=1) * scale   --> [BH]
         = sum(q[i] * k[j] * exp2(g[i] - g[j])) * scale

Step K:  Akk dot product (含 beta)
  b_Akk = sum(b_k * b_kgj, axis=1) * (j < i_t ? 1.0 : 0.0)   --> [BH]
         = sum(k[i]*beta[i] * k[j] * exp2(g[i] - g[j])) if j < i else 0
```

### 4.3 Gated Dot Product 计算详解

KDA 的 gate 机制源于 Delta Rule 的离散化。`g` 是 chunk-local 的 gate log-cumsum 值 (在 log2 空间)。对于任意两个 token (i, j)，衰减因子为:

```
decay(i, j) = exp2(g[i] - g[j])
```

其中 `exp2` 为以 2 为底的指数函数，在 Triton 中直接调用硬件指令实现。

**Aqk 公式:**
```
Aqk[i, j] = <q[i], k[j] * decay(i, j)> * scale
          = sum_d( q[i,d] * k[j,d] * exp2(g[i,d] - g[j,d]) ) * scale
```
实现为: `tl.sum(b_q * b_kj * exp2(b_g - b_gj), axis=1) * scale`
维度: `[BH, BK] * [BH, BK] * [BH, BK] -> sum over BK -> [BH]`

**Akk 公式 (含 beta 加权):**
```
Akk[i, j] = <k[i]*beta[i], k[j] * decay(i, j)>
          = sum_d( k[i,d]*beta[i] * k[j,d] * exp2(g[i,d] - g[j,d]) )
```
实现为: `tl.sum(b_k * b_kj * exp2(b_g - b_gj), axis=1)`
其中 b_k 已在循环外预乘 beta: `b_k = load(k[i]) * beta[i][:, None]`

### 4.4 Causal Masking

```
Aqk 矩阵 (上三角含对角线):
+-----+-----+-----+-----+
|  v  |  0  |  0  |  0  |  j <= i: 计算并存储
+-----+-----+-----+-----+
|  v  |  v  |  0  |  0  |  j > i:  不在循环范围内 (天然被跳过)
+-----+-----+-----+-----+
|  v  |  v  |  v  |  0  |
+-----+-----+-----+-----+
|  v  |  v  |  v  |  v  |
+-----+-----+-----+-----+

Akk 矩阵 (严格上三角, 对角线为 0):
+-----+-----+-----+-----+
|  0  |  0  |  0  |  0  |  j < i:  计算并存储
+-----+-----+-----+-----+
|  v  |  0  |  0  |  0  |  j == i: tl.where(j < i_t, 1.0, 0.0) 置零
+-----+-----+-----+-----+
|  v  |  v  |  0  |  0  |  j > i:  不在循环范围内
+-----+-----+-----+-----+
|  v  |  v  |  v  |  0  |
+-----+-----+-----+-----+
```

实现方式:
1. **Aqk 对角线 (j == i)**: 由循环上界 `min(i_t + 1, ...)` 自然包含——计算 `q[i] · k[i] * exp2(g[i]-g[i]) = q[i] · k[i]` 并存储。
2. **Akk 对角线 (j == i)**: `tl.where(j < i_t, 1.0, 0.0)` 将 j == i 时的值强制归零。
3. **超范围 (j > i)**: 不在 `range(i_ts, min(i_t+1, ..))` 内，循环不执行。

### 4.5 存储到 Aqk [B,T,H,BT] 和 Akk [B,T,H,BC]

```
Aqk 的存储布局:
  base = Aqk + bos*H*BT                                (batch 偏移)
  addr = base + i_t * H * BT                           (token 偏移)
       + (i_hg * BH + o_h) * BT                        (head 偏移: BT stride)
       + j % BT                                         (column: j 在 chunk 内的位置)

  语义: Aqk[batch_idx, i_t, head_idx, j % BT] = computed_value

  注意 Aqk 按 [B, T, H, BT] 排列:
  - BT=64 列对应 chunk 内 64 个历史 token 位置
  - 对角线块的 column 索引落在 [i_s*BC, (i_s+1)*BC-1] 范围内
  - 非对角线 column 保持不变 (由 inter_solve 核函数后续填入)

Akk 的存储布局:
  base = Akk + bos*H*BC                                (batch 偏移)
  addr = base + i_t * H * BC                           (token 偏移)
       + (i_hg * BH + o_h) * BC                        (head 偏移: BC stride)
       + j - i_ts                                       (column: 在 sub-chunk 内的相对位置)

  语义: Akk[batch_idx, i_t, head_idx, j - i_ts] = computed_value

  注意 Akk 按 [B, T, H, BC] 排列:
  - BC=16 列对应当前 sub-chunk 内 16 个 token 位置
  - 列 j - i_ts 是 sub-chunk 内的相对偏移 (0..BC-1)
  - 对角线 sub-chunk 外的 Akk 值由 inter_solve 核函数负责计算
```

---

## 5. 关键代码对应

| 设计逻辑 | 代码行号 | 代码片段/说明 |
|---------|---------|-------------|
| Grid 定义: B*T 个 token CTA | 175-176 | `def grid(meta): return (B * T, triton.cdiv(H, meta["BH"]))` |
| 全局 token 索引解析 | 47 | `i_tg, i_hg = tl.program_id(0), tl.program_id(1)` |
| 变长序列 batch 定位 (二分查找) | 49-68 | `for _ in range(20): ...` 在 `cu_seqlens` 上二分查找 |
| 定长模式索引计算 | 71-72 | `bos = (i_tg // T) * T; i_t = i_tg % T` |
| Sub-chunk 定位 | 77-80 | `i_c = i_t // BT; i_s = (i_t % BT) // BC; i_ts = i_tc + i_s * BC` |
| Batch/head 偏移设置 | 82-86 | `q += bos * H * K` 等 6 行指针偏移 |
| 当前 token q/k/g/beta 加载 | 105-108 | `load(b_q, b_k, b_g, b_beta)` |
| Key 预乘 beta | 108 | `b_k = b_k * b_beta[:, None]` |
| Inner loop: 遍历历史 token | 110 | `for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):` |
| 历史 token k[j]/g[j] 加载 | 111-119 | `load(p_kj, p_gj)` |
| Gate 衰减因子: exp2(g[i]-g[j]) | 121 | `b_kgj = b_kj * exp2(b_g - b_gj)` |
| 无效 K 维度掩码 | 123 | `b_kgj = tl.where(m_k[None, :], b_kgj, 0.0)` |
| Aqk dot product | 125 | `b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale` |
| Akk dot product (严格上三角) | 126 | `b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)` |
| Aqk 存储 | 128-132 | `store(Aqk + i_t*H*BT + ... + j % BT, ...)` |
| Akk 存储 | 133-137 | `store(Akk + i_t*H*BC + ... + j - i_ts, ...)` |
| BH=1, 每个 CTA 处理 1 个 head | 20-21 | Autotune config: `for BH in [1]` |
| BK 向上取整到 2 的幂 | 178 | `BK = triton.next_power_of_2(K)` |

---

## 6. 数据流图

```
                            +-------------------+
                            | q [B,T,H,K]       |----+
                            | k [B,T,H,K]       |--+ |
                            | g [B,T,H,K]       |-+| |
                            | beta [B,T,H]      ||| |
                            +-------------------+-++-+
                                                  |||
                    Grid: (B*T, cdiv(H, BH))      |||    BH=1
                    each CTA = 1 token x 1 head   |||
                                                  vvv
                   +=====================================+
                   |  CTA_{i_t, i_h}:                     |
                   |                                       |
                   |  Step 1: 索引计算                     |
                   |    i_t = token 全局 ID -> 局部 ID     |
                   |    bos = batch 偏移                   |
                   |    i_ts = sub-chunk 起始 token         |
                   |                                       |
                   |  Step 2: 加载当前 token (i) 的数据        |
                   |    b_q = q[i]      [BK]               |
                   |    b_k = k[i]      [BK]               |
                   |    b_g = g[i]      [BK]               |
                   |    b_b = beta[i]   [1]                |
                   |    b_k = b_k * b_b  (预乘 beta)       |
                   |                                       |
                   |  Step 3: Inner Loop                   |
                   |    for j in [i_ts, i_t]:              |
                   |                                        |
                   |      +----------+                      |
                   |      | 加载 k[j]  |  Memory Load        |
                   |      | 加载 g[j]  |  (H, K)             |
                   |      +----+-----+                      |
                   |           |                            |
                   |      +----v------+                     |
                   |      | exp2(     |                     |
                   |      |  b_g -    |  Gate 衰减因子        |
                   |      |  b_gj     |                     |
                   |      +----+------+                     |
                   |           |                            |
                   |      +----v------+                     |
                   |      | b_kgj =   |  Gated Key           |
                   |      | b_kj *    |  Product             |
                   |      | exp2(...) |                     |
                   |      +----+------+                     |
                   |           |                            |
                   |         +-+--+                         |
                   |         |    |                         |
                   |    +----v-+  +---v---+                 |
                   |    | Aqk   |  | Akk  |                 |
                   |    | sum(  |  | sum( |  Dot Product     |
                   |    | q *   |  | k *  |                 |
                   |    | kgj)  |  | kgj) |                 |
                   |    +--+---+  +---+--+                 |
                   |       |          |                     |
                   |  +----v------+   |                     |
                   |  | * scale   |   | Aqk 乘 scale         |
                   |  +----+------+   |                     |
                   |       |          |                     |
                   |  +----v------+ +-v------------+       |
                   |  | Store to  | | Store to     |       |
                   |  | Aqk[i,    | | Akk[i,       |       |
                   |  |  H, BT]   | |  H, BC]      |       |
                   |  +-----------+ +--------------+       |
                   +=====================================+

                            |            |
                            v            v
                   +----------------+  +----------------+
                   | Aqk [B,T,H,BT] |  | Akk [B,T,H,BC] |
                   | (dtype of q)   |  | (fp32)         |
                   +----------------+  +----------------+

                            |            |
                            v            v
                   (传递给 inter_solve 和后续 chunk-level 核函数)
```

---

## 7. VARLEN 变长序列支持

当 `cu_seqlens` 不为 None 时，kernel 需要将全局 token 索引 `i_tg` 映射到正确的 batch 索引和序列内局部偏移:

```
全局 token 索引 i_tg = 0..total_tokens-1, 所有序列展开到一维

Sequences:   Seq 0:  [t0 t1 t2 t3]
             Seq 1:  [t4 t5]
  cu_seqlens = [0, 4, 6]

二分查找:
  i_tg=0: left=0,right=2 -> mid=1 -> 0<6 -> right=1 -> mid=0 -> 0<4 -> right=0 -> left=0
         -> i_n=0, bos=0, i_t=0  (Seq 0, token 0)

  i_tg=5: left=0,right=2 -> mid=1 -> 5<6 -> right=1 -> mid=0 -> 5>=4 -> left=1 -> left=1
         -> i_n=1, bos=4, i_t=1  (Seq 1, token 1)

  i_tg=6: (>= total_tokens) -> i_t >= T -> return (边界检查跳过)
```

二分查找迭代次数限制为 20 次，足够覆盖 `B <= 2^20` 约 100 万条序列的场景。

---

## 8. 设计要点与权衡

| 方面 | 决策 | 原因 |
|------|------|------|
| Token-parallel 策略 | 每个 token 一个 CTA | 最大并行度: B*T*H 个 CTA，对小 batch 友好 |
| BH=1 | 每个 CTA 处理 1 个 head | 1 warp (32 threads) 足够处理 1 head 的 BC 次迭代 |
| BC=16 | Sub-chunk 大小固定 16 | 平衡: 太小则对角线块覆盖范围不足，太大则单个 CTA 工作量大 |
| Ack in Akkd (fp32) | 对角线 Akk 块存为 fp32 | 后续 forward substitution 需要高精度保证数值稳定性 |
| 循环次数 vs 内存 | 每次迭代从 global memory 加载 k[j], g[j] | 避免大寄存器压力，利用 L1 cache 缓存最近 sub-chunk 的数据 |
| VARLEN 二分查找 | 展开的串行二分查找 | 避免 warp divergence; 在 CUDA 上实际执行速度很快 |
| scale 对 Aqk 乘 | Aqk *= scale, Akk 不乘 | Akk 参与后续矩阵求逆和 dot product，保持原始数值范围 |
