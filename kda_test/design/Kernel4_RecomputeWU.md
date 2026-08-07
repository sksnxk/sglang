# Kernel 4 核函数设计文档: Recompute W/U

## 1. 输入输出定义

### 输入

| 名称 | 形状 | 语义 |
|------|------|------|
| `k` | `[B, T, H, K]` | 原始 Key 张量 |
| `v` | `[B, T, H, V]` | 原始 Value 张量 |
| `beta` | `[B, T, H]` | Gated Delta Rule 的 beta 门控系数 |
| `A` | `[B, T, H, BT]` | Akk_inv：chunk 内 KKT 矩阵的逆矩阵 |
| `gk` | `[B, T, H, K]` | Key 方向的 gate cumsum（log2 空间），预乘以 exp2 使用 |
| `cu_seqlens` | `[N+1]` | 变长序列的累积序列长度（可选，IS_VARLEN 模式） |
| `chunk_indices` | `[NT, 2]` | 每个 chunk 的 (seq_id, chunk_start) 索引（可选） |

### 输出

| 名称 | 形状 | 语义 |
|------|------|------|
| `w` | `[B, T, H, K]` | 解耦后的 Key 表示：`A_inv @ (k * beta * exp2(gk))` |
| `u` | `[B, T, H, V]` | 解耦后的 Value 表示：`A_inv @ (v * beta)` |
| `kg` | `[B, T, H, K]` | 时间对齐的 Key：`k * exp2(gk_last - gk)`（可选，当 gk 非空时输出） |

### 固定形状参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `BT` | chunk_size（通常 64） | 每个 chunk 的 token 数，等于 `A.shape[-1]` |
| `BK` | 32 | K 维度的 tile 大小（autotune 配置值） |
| `BV` | 32 | V 维度的 tile 大小（autotune 配置值） |

---

## 2. 分核并行策略

### Grid 定义

```
Grid = (NT, B * H)
  ^      ^
  |      +-- batch * head：每个 (batch, head) 对包含该序列的所有 chunk
  +--------- NT = ceil(T / BT)：chunk 个数
```

每个 CTA 负责**一个 (chunk, head) 对**，处理一个 `[BT, K]` 的 Key 块和一个 `[BT, V]` 的 Value 块。

### V-dim / K-dim 循环嵌套

kernel 内部的 tile 遍历是顺序的（单 CTA 串行遍历）：

```
+------------------------------------------------------------------+
| CTA(i_t, i_bh)                                                    |
|                                                                   |
|  /* 公共加载：每个 CTA 在 loop 之前加载一次 */                        |
|  1. 加载 beta[BT]              形状 [BT]                          |
|  2. 加载 A_inv[BT, BT]         形状 [BT, BT]                      |
|                                                                   |
|  /* V 维度循环 */                                                  |
|  for i_v in 0 .. ceil(V / 32):                                    |
|    3. 加载 v[BT, 32]            形状 [BT, BV]                      |
|    4. v' = v * beta             逐元素乘 [BT, BV]                  |
|    5. u = A_inv @ v'            dot: [BT,BT] x [BT,BV] -> [BT,BV] |
|    6. 存出 u[BT, 32]                                              |
|                                                                   |
|  /* K 维度循环 */                                                  |
|  for i_k in 0 .. ceil(K / 32):                                    |
|    7. 加载 k[BT, 32]            形状 [BT, BK]                      |
|    8. k' = k * beta             逐元素乘 [BT, BK]                  |
|    9. 加载 gk[BT, 32]           形状 [BT, BK]                      |
|   10. k' = k' * exp2(gk)        [BT, BK]                          |
|   11. 若 STORE_KG:                                                |
|        a. 加载 gk_last[BK]      最后 token 的 gk 值                |
|        b. kg = k * exp2(gk_last - gk)  时间对齐                    |
|        c. 存出 kg[BT, 32]                                         |
|   12. w = A_inv @ k'            dot: [BT,BT] x [BT,BK] -> [BT,BK] |
|   13. 存出 w[BT, 32]                                              |
+------------------------------------------------------------------+
```

### 访存分析

每个 CTA 的内存访问模式：

```
Offsets: bos = seq_start, 即 batch 和 sequence 的起始偏移

+-----------+----------------------------+----------+--------------------------+
| Tensor    | Base Offset Formula        | Shape    | 跨度说明                  |
+-----------+----------------------------+----------+--------------------------+
| beta      | bos*H + i_h                | [T,]     | stride=H, 顺序连续         |
| A(Akk_inv)| (bos*H + i_h) * BT         | [T, BT]  | stride=(H*BT, 1), 行主序  |
| k         | (bos*H + i_h) * K          | [T, K]   | stride=(H*K, 1), 行主序   |
| v         | (bos*H + i_h) * V          | [T, V]   | stride=(H*V, 1), 行主序   |
| gk        | (bos*H + i_h) * K          | [T, K]   | stride=(H*K, 1), 行主序   |
| w         | (bos*H + i_h) * K          | [T, K]   | stride=(H*K, 1), 行主序   |
| u         | (bos*H + i_h) * V          | [T, V]   | stride=(H*V, 1), 行主序   |
| kg        | (bos*H + i_h) * K          | [T, K]   | stride=(H*K, 1), 行主序   |
+-----------+----------------------------+----------+--------------------------+
```

---

## 3. 计算思路

### 3a. w 的计算：`A_inv @ (k * beta * exp2(gk))`

**数学公式：**

```
w_i = sum_j A_inv(i, j) * k_j * beta_j * exp2(gk_j)
```

其中 `i, j` 是 chunk 内的 token 索引（0 到 BT-1），`beta_j` 和 `gk_j` 是标量和向量。

**分块矩阵乘法图示：**

```
  [BT, K]                        [BT, BT]          [BT, K]
+-----------+                  +-----------+     +-----------+
| k_0       |                  | A00..A0,BT|     | w_0       |
| k_1       |  元素乘             | A10..A1,BT|     | w_1       |
| ...       |  beta[j] *       | ...  ...  |  @  | ...       |
| k_{BT-1}  |  exp2(gk_j)       | ABT,0..ABT|     | w_{BT-1}  |
+-----------+                  +-----------+     +-----------+
     K dim                          BT x BT           K dim
```

逐 K-tile（BK=32）分块：

```
for i_k in range(ceil(K / 32)):
    k_tile = load([BT, 32])                  # 加载一个 K-tile
    gk_tile = load([BT, 32])                 # 对应 gk-tile
    k_scaled = k_tile * beta[:, None]        # [BT, BK] * [BT, 1]
    k_scaled *= exp2(gk_tile)                # gate 调制
    w_tile = tl.dot(A_inv, k_scaled)         # [BT, BT] @ [BT, BK]
    store(w_tile)                             # 存出 [BT, BK]
```

### 3b. u 的计算：`A_inv @ (v * beta)`

**数学公式：**

```
u_i = sum_j A_inv(i, j) * v_j * beta_j
```

**分块矩阵乘法图示：**

```
  [BT, V]                     [BT, BT]          [BT, V]
+-----------+               +-----------+     +-----------+
| v_0       |               | A00..A0,BT|     | u_0       |
| v_1       |  元素乘         | A10..A1,BT|     | u_1       |
| ...       |  beta[j]       | ...  ...  |  @  | ...       |
| v_{BT-1}  |               | ABT,0..ABT|     | u_{BT-1}  |
+-----------+               +-----------+     +-----------+
     V dim                       BT x BT           V dim
```

逐 V-tile（BV=32）分块：

```
for i_v in range(ceil(V / 32)):
    v_tile = load([BT, 32])                # 加载一个 V-tile
    v_scaled = v_tile * beta[:, None]      # [BT, BV] * [BT, 1]
    u_tile = tl.dot(A_inv, v_scaled)       # [BT, BT] @ [BT, BV]
    store(u_tile)                            # 存出 [BT, BV]
```

**注意：** u 计算没有 gk 调制，因为 Value 不受 Linear Attention 的 gate 影响

### 3c. kg 的计算：`k * exp2(gk_last - gk)`

**数学公式：**

```
kg(i) = k_i * exp2(gk_last - gk_i)
```

其中 `gk_last` 是 chunk 内最后一个有效 token 的 gk 值。

**物理含义：** 将 chunk 内每个 token 的 Key 从"各自时间戳"对齐到"chunk 末尾时间戳"。

**逐元素操作：**

```
如果启用 STORE_KG:
    last_idx = min(i_t * BT + BT, T) - 1      # chunk 最后一个有效 token
    gk_last = load(gk[last_idx, :BK])          # [BK] 向量
    kg_tile = k_tile * exp2(gk_last - gk_tile) # [BT, BK] 逐元素
    store(kg_tile)
```

**为何 kg 在 K 循环中而非独立循环：** 复用已加载的 k_tile 和 gk_tile，减少重复访存。

### 3d. 为什么需要用 Akk_inv（chunk 内因果依赖解耦）

#### 背景：Chunk-wise Linear Attention 的因果依赖

在 Linear Attention 中，对于 chunk 内的 token `i`，其输出为：

```
o_i = q_i^T @ sum_{j <= i} k_j v_j^T   (因果掩码)
```

其中 `sum_{j <= i} k_j v_j^T` 是一个累积的 KV 状态，在 chunk 内依赖因果顺序。

#### Chunk 分解策略

为了并行化，将序列切成大小为 BT 的 chunk。单个 chunk 内的计算可以分解为：

```
KV_state(i) = sum_{j <= i, j in same chunk} k_j v_j^T
```

这里的核心问题：chunk 内的 token 之间仍然存在因果依赖（`j <= i`），不能直接做矩阵乘法。

#### Akk 矩阵的引入

引入 chunk 内的 pairwise KKT 矩阵（带 gate 对齐）：

```
Akk_{m,n} = beta_n * k_m^T * k_n * exp2(gk_n - gk_last)
```

其中因果性由 `Akk_{m,n} = 0 for m > n` 来体现（下三角矩阵）。

#### 为什么需要求逆

对于 chunk 内位置 `i` 的累积 KV 状态，有：

```
KV_state(i) = sum_j A_{i,j} * beta_j * k_j v_j^T
```

其中 A 是 Akk 矩阵的下三角因子。当我们知道完整的 Akk（包含因果关系的 KKT 矩阵）后，可以通过求解下三角方程组得到其逆 `Akk_inv`。

用 `Akk_inv` 乘以 beta 调制后的 k/v，就可以得到**解耦的** `w` 和 `u`：

```
w = Akk_inv @ (k * beta * exp2(gk))       # Key 解耦表示
u = Akk_inv @ (v * beta)                  # Value 解耦表示
```

**核心洞察：** `w` 和 `u` 不再包含 chunk 内的因果依赖，它们可以直接与外积形式 `w * u^T` 参与跨 chunk 的递推计算。这就是"recompute"的含义——从原始 k/v 通过 Akk_inv 重新计算出适合跨 chunk 传播的 w/u。

#### 数据流总结

```
               chunk 内 KKT 计算               求逆/三角求解              重新计算
raw k,v,gk  ──────────────────> Akk [BT,BT]  ──────────────> Akk_inv [BT,BT]  ──> w [BT,K], u [BT,V]
(raw)                              (带因果的KKT)          (去掉因果的解耦矩阵)       (解耦表示，可跨chunk传播)
```

---

## 4. 关键代码对应

以下表格将上文分析的计算步骤映射到 `kda.py` 中的具体代码行：

| 计算步骤 | 代码行号 | 代码要点 |
|----------|----------|----------|
| CTA 索引（chunk_id, batch_head） | 550-551 | `i_t, i_bh = tl.program_id(0), tl.program_id(1)` |
| VARLEN 序列范围计算 | 552-561 | 从 `chunk_indices` 和 `cu_seqlens` 获取 (bos, eos, T) |
| 加载 beta [BT] | 564-565 | `p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), ...)` |
| 加载 A(Akk_inv) [BT, BT] | 567-570 | `p_A = tl.make_block_ptr(A + (bos*H + i_h)*BT, (T, BT), ...)` |
| V 维度循环头 | 572 | `for i_v in range(tl.cdiv(V, BV)):` |
| 加载 v tile | 573-580 | `p_v = tl.make_block_ptr(v + (bos*H + i_h)*V, (T, V), ...)` |
| v * beta | 590 | `b_vb = (b_v * b_b[:, None]).to(b_v.dtype)` |
| u = A_inv @ (v * beta) | 591 | `b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)` |
| K 维度循环头 | 594 | `for i_k in range(tl.cdiv(K, BK)):` |
| 加载 k tile | 603-610 | `p_k = tl.make_block_ptr(k + (bos*H + i_h)*K, (T, K), ...)` |
| k * beta | 612 | `b_kb = b_k * b_b[:, None]` |
| 加载 gk tile | 614-621 | `p_gk = tl.make_block_ptr(gk + (bos*H + i_h)*K, (T, K), ...)` |
| k * beta * exp2(gk) | 623 | `b_kb *= exp2(b_gk)` |
| 计算 kg（时间对齐） | 625-641 | `b_kg = b_k * exp2(b_gn - b_gk)` |
| w = A_inv @ (k * beta * exp2(gk)) | 644 | `b_w = tl.dot(b_A, b_kb.to(b_k.dtype))` |
| 存出 w | 645 | `tl.store(p_w, b_w.to(p_w.dtype.element_ty), ...)` |

### Python 包装层关键代码

```python
# kda.py 第 648-687 行

def recompute_w_u_fwd(k, v, beta, A, gk, cu_seqlens, chunk_indices):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]

    # 准备 chunk 索引
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # 输出 buffer 分配
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k) if gk is not None else None

    # 启动 kernel：(NT, B*H) 个 CTA
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k, kg=kg, v=v, beta=beta, w=w, u=u, A=A, gk=gk,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, BT=BT,
        STORE_KG=kg is not None,         # 是否计算 kg
        IS_VARLEN=cu_seqlens is not None, # 是否变长序列
        DOT_PRECISION="tf32",            # 使用 TF32 精度做 dot
    )
    return w, u, kg
```

### Autotune 配置

```python
# 第 517-526 行
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32]
        for BV in [32]
        for num_warps in [1]
        for num_stages in [1]
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
)
```

当前配置固定为 BK=32, BV=32, num_warps=1, num_stages=1。这种轻量配置适合该 kernel 的访存密集型特性（每个 CTA 只有少量计算，主要开销在 HBM 读写）。

---

## 5. 数据流图

### 5a. 单个 CTA 执行流程图

```
                         +------------------+
                         | CTA(i_t, i_bh)   |
                         | chunk=i_t        |
                         | bh=i_bh          |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
            +-------+-------+           +-------+-------+
            | load beta[BT] |           | load A[BT,BT] |
            | from HBM      |           | from HBM      |
            +-------+-------+           +-------+-------+
                    |                           |
                    v                           v
              b_b: [BT]                   b_A: [BT,BT]
              (在 SRAM)                    (在 SRAM)
                    |                           |
                    +-------+-------+-----------+
                            |       |
                            |       +-------------------------------+
                            |                                       |
                            v                                       v
                +-----------+-----------+               +-----------+-----------+
                | V-dim loop              |               | K-dim loop              |
                | for i_v in 0..V/32:     |               | for i_k in 0..K/32:     |
                |                         |               |                         |
                | +-----> load v[BT,32]   |               | +-----> load k[BT,32]   |
                | |        from HBM       |               | |        from HBM       |
                | |                       |               | |                       |
                | |   v' = v * beta      |               | |   k' = k * beta       |
                | |   [BT,32] * [BT,1]   |               | |   [BT,32] * [BT,1]    |
                | |                       |               | |                       |
                | |   u = A @ v'         |               | |   load gk[BT,32]      |
                | |   tl.dot(            |               | |   from HBM             |
                | |     [BT,BT], [BT,32] |               | |                       |
                | |   ) -> [BT,32]       |               | |   k' *= exp2(gk)      |
                | |                       |               | |   [BT,32]              |
                | |   store u[BT,32]     |               | |                       |
                | |   to HBM             |               | |   if STORE_KG:         |
                | |                       |               | |     load gk_last[BK]  |
                | +--(next i_v)----------+               | |     kg = k * exp2(     |
                |                                       | |       gk_last - gk)     |
                +-------+-------------------------------+ |     store kg[BT,32]    |
                        |                                 |                         |
                        v                                 |   w = A @ k'           |
                   (done V)                               |   tl.dot(              |
                                                          |     [BT,BT], [BT,32]  |
                                                          |   ) -> [BT,32]         |
                                                          |                         |
                                                          |   store w[BT,32]       |
                                                          |   to HBM               |
                                                          |                         |
                                                          +--(next i_k)------------+
                                                                       |
                                                                       v
                                                                  (done K)
                                                                       |
                                                                       v
                                                                  CTA 完成
```

### 5b. Grid 级别并行拓扑

```
Batch=2, H=3, T=256, BT=64, NT=4

Grid: (4, 6) = (NT, B*H)

    H0     H1     H2    H0     H1     H2
    B0     B0     B0    B1     B1     B1
  +------+------+------+------+------+------+
  | CTA  | CTA  | CTA  | CTA  | CTA  | CTA  |  chunk 0 (tokens 0..63)
  |(0,0) |(0,1) |(0,2) |(0,3) |(0,4) |(0,5) |
  +------+------+------+------+------+------+
  | CTA  | CTA  | CTA  | CTA  | CTA  | CTA  |  chunk 1 (tokens 64..127)
  |(1,0) |(1,1) |(1,2) |(1,3) |(1,4) |(1,5) |
  +------+------+------+------+------+------+
  | CTA  | CTA  | CTA  | CTA  | CTA  | CTA  |  chunk 2 (tokens 128..191)
  |(2,0) |(2,1) |(2,2) |(2,3) |(2,4) |(2,5) |
  +------+------+------+------+------+------+
  | CTA  | CTA  | CTA  | CTA  | CTA  | CTA  |  chunk 3 (tokens 192..255)
  |(3,0) |(3,1) |(3,2) |(3,3) |(3,4) |(3,5) |
  +------+------+------+------+------+------+

  所有 CTA 完全独立，无同步 —— 各 chunk/head 之间无数据依赖。
```

### 5c. 调用方上下文：Chunk Intra 中的位置

```
chunk_gated_delta_rule_fwd()  调用链:

  +-- Step 1: chunk_kda_scaled_dot_kkt_fwd()
  |      计算 Akk = beta * K_gated * K_gated^T     (chunk 内 KKT 矩阵)
  |
  +-- Step 2: chunk_kda_fwd_kernel_inter_solve_fused()
  |      下三角求解：Akk -> Akk_inv                 (求逆，解除因果依赖)
  |
  +-- Step 3: recompute_w_u_fwd()    <-- Kernel 4 (本文档)
  |      w = Akk_inv @ (k * beta * exp2(gk))
  |      u = Akk_inv @ (v * beta)
  |      kg = k * exp2(gk_last - gk)
  |      将因果解耦后的 w, u 输出，供后续跨 chunk 递推使用
  |
  +-- Step 4: chunk_gla_fwd_kernel()
         利用 w, u, kg 做跨 chunk 的矩阵乘累加得到最终输出 o
```

### 5d. 端到端 KDA 前向数据流

```
  raw_k[T,K]   raw_v[T,V]     g[T,K]
       |            |             |
       v            v             v
  +---------+    +---------+    +---------+
  | cumsum  |    | (直接   |    | cumsum  |
  | (g->gk) |    |  使用)  |    |         |
  +----+----+    +----+----+    +----+----+
       |              |             |
       v              v             v
  gk[T,K]        raw_v[T,V]     raw_g[T,K]
       |              |             |
       v              |             v
+------------------+  |  +-------------------+
| inter_solve      |  |  | beta [T,H]       |
|  Akk = K*K^T     |  |  | (from model)     |
|  Akk -> Akk_inv  |  |  +-------------------+
+--------+---------+  |          |
         |            |          |
         v            |          v
    Akk_inv[BT,BT]    |     beta[BT]
         |            |          |
         +-----+------+----+-----+
               |           |
               v           v
      +--------+-----------+--------+
      |  Kernel 4: recompute_w_u   |
      |                            |
      |  w = A_inv @ (k*b*e^gk)    |
      |  u = A_inv @ (v*b)         |
      |  kg = k * e^(gk_n - gk)    |
      +--------+-----------+--------+
               |           |
               v           v
          w[T,K]       u[T,V]       kg[T,K]
                \       /                \
                 \     /                  \
                  v   v                    v
           +-------+-------+      +--------+--------+
           | chunk recurr.  |      | chunk_gla_fwd_o  |
           | (跨chunk累积)  |      | (利用kg做最终输出) |
           +-------+-------+      +-------------------+
                   |
                   v
              o[T,V]  (最终输出)
```
