# Kernel 5 核函数设计文档: Delta Rule H

## 1. 输入输出定义

| 参数名 | 类型 | 形状 | 语义 |
|--------|------|------|------|
| `k` (输入) | float16/bf16 | `[B, T, Hg, K]` | 衰减后的 key 张量: `k * beta * exp2(gk_last - gk)` (即 kg，由 Kernel 4 recompute_w_u_fwd 计算) |
| `v` (输入) | float16/bf16 | `[B, T, H, V]` | 原始 value 张量 u (由 Kernel 4 计算 `u = Aqk @ (v * beta)`) |
| `w` (输入) | float16/bf16 | `[B, T, H, K]` | 衰减后的 w 张量 (由 Kernel 4 recompute_w_u_fwd 计算) |
| `g` (输入, 可选) | float32 | `[B, T, H]` | 逐 token 标量 gate (natural log 空间) |
| `gk` (输入, 可选) | float32 | `[B, T, H, K]` | 逐 token / 逐 channel 的 per-channel gate (log2 空间，已 cumsum + scale) |
| `h` (输出) | float16/bf16 | `[B, NT, H, V, K]` | 每个 chunk 结束时状态的快照，供 Kernel 6 (output) 读取 |
| `initial_state` (输入/输出) | float16/bf16 | `[N, H, V, K]` | 初始状态（输入），被 in-place 更新为最终状态（输出） |
| `initial_state_indices` (输入) | int32 | `[B]` | 每个 batch 条目指向 `initial_state` 的索引 |
| `v_new` (输出, 可选) | float16/bf16 | `[B, T, H, V]` | Delta Rule 更新后的残差 value: `v_new = u - w @ h[t-1]` (供 Kernel 6 output 使用) |
| `cu_seqlens` (输入, 可选) | int32 | `[N+1]` | 可变长度序列的累积长度 |
| `chunk_offsets` (输入, 可选) | int32 | `[N+1]` | 每个序列的 chunk 起始偏移 |
| `T` (标量) | int | — | 每个序列的总 token 数 |
| `H` (constexpr) | int | — | value 的 head 数 |
| `Hg` (constexpr) | int | — | key 的 head 数（GQA / MQA 分组） |
| `K` (constexpr) | int | — | head 维度（key 维度） |
| `V` (constexpr) | int | — | head 维度（value 维度），**本项目中恒等于 K** |
| `BT` (constexpr) | int | — | 每个 chunk 的 token 数，恒为 64 |
| `BV` (constexpr) | int | — | V 维度的 tile 大小，默认 32（环境变量 `SGLANG_GDN_CHUNK_H_BV`） |
| `USE_G` (constexpr) | bool | — | 是否启用标量 gate 衰减 |
| `USE_GK` (constexpr) | bool | — | 是否启用 per-channel gate 衰减 |
| `USE_INITIAL_STATE` (constexpr) | bool | — | 是否有初始状态需要加载 |
| `INPLACE_UPDATE` (constexpr) | bool | — | 是否原地写回 final state，当前恒为 True |
| `SAVE_NEW_VALUE` (constexpr) | bool | — | 是否保存 v_new（供 Kernel 6 读取） |
| `USE_EXP2` (constexpr) | bool | — | gk 使用 exp2 还是 exp (natural)，本项目使用 exp2 |
| `NT_BUCKET` (constexpr) | int | — | NT 分段桶编号（0: NT<=32, 1: NT<=128, 2: 更大），为 autotune 预留 |

**注意**: `k` / `w` / `v` 等参数的 view 窗口由 `cu_seqlens` 按序列实时计算，不是直接使用整个 B 维度（varlen 模式）。

---

## 2. 分核并行策略

### 2.1 Grid 布局

```
Grid = ( cdiv(V, BV),  N * H )

               N * H 个 CTA (序列-头对)
           ┌───────────────────────────────────────────────────────────────┐
           │ CTA(B,0,0) │ CTA(B,0,1) │ ... │ CTA(B,0,H-1) │               │
           │ CTA(B,1,0) │ CTA(B,1,1) │ ... │ CTA(B,1,H-1) │               │
  cdiv     │    ...      │    ...     │     │     ...       │               │
  (V,BV)   ├─────────────┴────────────┴─────┴───────────────┘               │
           │                                                               │
           │ 每个 CTA 的 i_v = program_id(0)，表示当前处理哪个 V-tile      │
           │ 每个 CTA 的 i_nh = program_id(1)，其中 i_n = i_nh // H，      │
           │                            i_h = i_nh % H                     │
           └───────────────────────────────────────────────────────────────┘
```

### 2.2 并行维度

| 维度 | 粒度 | 并行方式 |
|------|------|----------|
| 序列 (B/N) | 每个序列 | **Grid Y 并行**: 每个序列-头对为一个 CTA |
| 头 (H) | 每个头 | **Grid Y 并行**: 序列内所有头可并行 |
| V 维度 | BV=32 个元素/tile | **Grid X 并行**: 不同 V-tile 可并行 |
| Chunk (NT) | 整个序列的所有 chunk | **串行**: 每个 CTA 内用 `for i_t in range(NT)` 顺序推进 |
| K 维度 | 64 个元素/tile | **CTA 内展开**: 4 个寄存器块 b_h1~b_h4 覆盖 K<=256 |

### 2.3 关键约束

```
Grid X 并行（V 维度分块）可以并行:
  - 因为每个 V-tile 的状态 b_h 相互独立（不同 V 行对不同的 V 输出维度做外积）
  - 但每个 CTA 内部的 h(state) 只覆盖 [i_v*BV : (i_v+1)*BV] 行

Grid Y 并行（序列-头对）完全独立:
  - 不同头和不同序列之间的 Delta Rule 状态相互隔离

K 维度不并行（CTA 内串行展开）:
  - 因为每个 BK=64 tile 的内积必须累加到同一个 state 上
  - 通过 4 个独立的寄存器块 b_h1~b_h4 实现，无需跨 CTA 通信
```

---

## 3. 计算思路

### 3.1 整体算法: Delta Rule 跨 Chunk 递推

Delta Rule 的核心思想是: 用线性注意力近似，key-value 记忆通过外积不断累积到状态矩阵 h 中。

每个 chunk t 包含 BT=64 个连续的 token。状态 h 在经过每个 chunk 后更新:

```
h[t] = h[t-1] * decay[t] + K[t]^T @ V_new[t]
```

其中:
- `h[t]` 是第 t 个 chunk 结束时的状态矩阵，形状为 `[V, K]`
- `decay[t]` 是第 t 个 chunk 结束时的遗忘因子
- `V_new[t] = U[t] - W[t] @ h[t-1]` 是 Delta Rule 的核心: 用当前状态的线性预测减去原始 value
- `U[t]` 是 chunk t 中经 beta 和 Aqk 变换后的 value
- `W[t]` 是 chunk t 中经 beta 和 gate 衰减后的 key-stats（作为线性预测的权重）
- `K[t]` 是 chunk t 中衰减后的 key

**为什么叫 Delta Rule?** 因为它不是直接使用原始 value U[t]，而是使用 "当前输入与基于历史记忆的预测之间的差异" ΔV = U - W @ h。这类似于 Delta Rule 在神经网络中的更新方式: 权重变化 = 学习率 x 误差 x 输入。

### 3.2 初始状态加载（第 125-142 行）

```
initial_state 形状: [cache_pool_size, H, V, K]

每个序列通过 initial_state_indices[B] 索引到它的状态池:
  h0_ptr = initial_state + initial_state_indices[i_n] * (H * V * K) + i_h * V * K

加载当前 V-tile [i_v*BV : (i_v+1)*BV, :] 到 b_h1 ~ b_h4:

  b_h1 = h0[ i_v*BV : (i_v+1)*BV,    0 : 64  ]   ← K tile 0
  b_h2 = h0[ i_v*BV : (i_v+1)*BV,  64 : 128 ]   ← K tile 1 (if K > 64)
  b_h3 = h0[ i_v*BV : (i_v+1)*BV, 128 : 192 ]   ← K tile 2 (if K > 128)
  b_h4 = h0[ i_v*BV : (i_v+1)*BV, 192 : 256 ]   ← K tile 3 (if K > 192)
```

```
ASCII 图示: 初始状态矩阵 h0 (V x K) 与分块加载

                   K dimension (up to 256)
                ┌──────────────────────────────────┐
                │ ←── 64 ──→│←── 64 ──→│←── 64 ──→ │
                │           │          │           │
          BV    │  b_h1     │  b_h2    │  b_h3 ... │  ← 当前 CTA 加载的
  V rows  rows  │ (loaded)  │ (loaded) │ (loaded)  │     V 行切片
                │           │          │           │
                ├───────────┴──────────┴───────────┤
                │  (其他 V 行由其他 CTA 处理)       │
                │                                 │
                └─────────────────────────────────┘

  加载方式: tl.make_block_ptr(h0, (V, K), (K, 1), (i_v*BV, 0), (BV, 64), (1, 0))
  - 形状 (V, K)，步长 (K, 1)：行优先存储，每行 K 个连续元素
  - 偏移 (i_v*BV, 0)：从第 i_v 个 V-tile 开始，K 维度偏移 0
  - Block 形状 (BV, 64)：一次加载 BV 行 x 64 列的矩形
```

### 3.3 Chunk 循环体（第 145-288 行）

主循环 `for i_t in range(NT)` 依次处理每个 chunk。每个 CTA 内部的循环是**完全串行**的。

```
for i_t = 0, 1, 2, ..., NT-1:

    chunk t ──────────────────────────┐
    │                                 │
    │  ① 保存状态快照                  │
    │  ② Delta Rule 计算 v_new        │
    │  ③ Gate 衰减 (可选)             │
    │  ④ 状态更新 h += kg^T @ v_new   │
    └─────────────────────────────────┘
         ↓
    chunk t+1 使用更新后的 h
```

#### 步骤 1: 保存状态快照 `h[i_t] = state` (第 149-167 行)

```
目的: 保存当前 chunk 开始时的状态到全局内存 h[NT, H, V, K]
这样 Kernel 6 (output kernel) 在计算最终输出时可以直接读取 h[t-1]

为什么用 flat 1D pointer store 而不是 block_ptr？
  → triton-ascend 编译器 bug: block_ptr store 到 (V, K) 形状会损坏源寄存器
  → 变通方案: reshape 成 (BV*64,) 的一维 flat tensor，用 1D store

ASCII: h 张量的内存布局和快照存储

  h[B, NT, H, V, K] 的布局:
    dim 0: batch (B)
      dim 1: chunks (NT)
        dim 2: heads (H)
          dim 3: V dimension (V)
            dim 4: K dimension (K) — 每行 K 个连续元素

  存储到 h 的地址:
    p_h = h + i_t * (H*V*K) + i_v * BV * K + offset_k + tl.arange(0, BV*64)

  各 K-tile 的偏移:
    b_h1 → offset_k = 0
    b_h2 → offset_k = 64    (if K > 64)
    b_h3 → offset_k = 128   (if K > 128)
    b_h4 → offset_k = 192   (if K > 192)
```

#### 步骤 2: Delta Rule 计算 `v_new = u - w @ h[t-1]` (第 169-195 行)

```
Delta Rule 核心:
  v_new[t] = u[t] - W[t] @ h[t-1]^T

其中:
  - u[t]: chunk t 的原始 value     → [BT, V]  形状
  - W[t]: chunk t 的 w 矩阵         → [BT, K]  形状
  - h[t-1]: 上一 chunk 结束时的状态 → [V, K]   形状
  - h[t-1]^T: 状态的转置            → [K, V]   形状
  - W[t] @ h[t-1]^T: 历史预测       → [BT, V]  形状
  - v_new[t]: 残差 (误差)            → [BT, V]  形状

PS: 代码中 v 参数实际传入的是 u (由 chunk_kda_fwd 调用时 v=u)，
    所以 p_v 加载的是原始 u[t]，然后减去预测得到 v_new[t]。

分块计算 (K 维度分 4 个 64 宽的 tile):

  b_v = 0  (初始化为 0)
  b_v += W_tile1 [BT, 64]  @  h1^T [64, BV]   ← K-tile 0
  b_v += W_tile2 [BT, 64]  @  h2^T [64, BV]   ← K-tile 1 (if K > 64)
  b_v += W_tile3 [BT, 64]  @  h3^T [64, BV]   ← K-tile 2 (if K > 128)
  b_v += W_tile4 [BT, 64]  @  h4^T [64, BV]   ← K-tile 3 (if K > 192)

  b_v = u[t] - b_v    ← 残差 = 原始值 - 历史预测
```

```
ASCII: Delta Rule 的分块矩阵乘法 (V-tile x K-tile)

   W[BT, K]                     h_state^T [K, BV]
  ┌────────────┐              ┌────────┐
  │            │              │        │
  │ w_tile_1   │              │ h1^T   │  ← [64, BV]
  │ [BT, 64]   │       @      │        │
  │            │              ├────────┤
  ├────────────┤              │        │
  │            │              │ h2^T   │  ← [64, BV] (if K > 64)
  │ w_tile_2   │              │        │
  │ [BT, 64]   │       @      ├────────┤
  │            │              │ ...    │
  ├────────────┤              │        │
  │ w_tile_3   │              │        │
  │ [BT, 64]   │              └────────┘
  │            │
  ├────────────┤          Accumulate into:
  │ w_tile_4   │          b_v [BT, BV] = SUM(w_i @ h_i^T)
  │ [BT, 64]   │
  │            │          Then: b_v = u[t] - b_v → v_new[t]
  └────────────┘

  PS: h 在代码中的布局确实是 [BV, K] / [BV, 64]，
      所以 tl.dot(b_w, tl.trans(b_h1)) 就是 W [BT, 64] @ h1^T [64, BV]
      = 累加 W 与 h 各 K-tile 分块的内积
```

#### 步骤 3: Gate 衰减 (第 203-263 行)

##### 3.3.1 标量 Gate (USE_G)

```
USE_G: 逐 token 标量 gate (natural log 空间，对应 chunk t 内各 token)

last_idx = min((i_t+1) * BT, T) - 1   ← chunk 最后一个有效 token

加载 g_last (chunk 末尾的 gate) 和 g[tokens] (chunk 内各 token 的 gate):

  b_g_last = g[last_idx]              ← 标量，chunk 末尾 gate
  b_g      = g[i_t*BT : (i_t+1)*BT]   ← [BT]，chunk 内各 token 的 gate

应用 gate:
  b_v    = b_v * exp(b_g_last - b_g)[:, None]   ← 广播到 V 列
  b_h1   = b_h1 * exp(b_g_last)                  ← 状态衰减 (标量广播)
  b_h2   = b_h2 * exp(b_g_last)
  b_h3   = b_h3 * exp(b_g_last)
  b_h4   = b_h4 * exp(b_g_last)

为什么对 v_new 乘 exp(b_g_last - b_g)?
  这是一个补偿项: 让每个 token 的 v_new 与它的 gate 值对齐。
  b_g_last 是 chunk 末尾的定位锚点，exp(b_g_last - b_g) 对 chunk 尾部 token
  接近 1.0 (self-match)，对开头更小 (更早遗忘)。

为什么对状态 h 乘 exp(b_g_last)?
  遗忘机制: 状态逐年衰减，越早的状态 influence 越小。
  乘以 exp(b_g_last) 等价于让当前 chunk 之后的状态按该 chunk 末端的 gate 衰减。
```

##### 3.3.2 Per-Channel Gate (USE_GK)

```
USE_GK: 逐 token / 逐 channel gate (log2 空间，已 chunk-cumsum)

加载 gk_last (chunk 末尾的 gk):
  b_gk_last1 = gk[last_idx, 0:64]     ← [64]，K-tile 0 的末尾 gate
  b_gk_last2 = gk[last_idx, 64:128]   ← [64]，K-tile 1 的末尾 gate
  ...

应用 gate (采用 exp2):
  b_h1 *= exp2(b_gk_last1)[None, :]   ← [64] 广播到 [BV, 64]
  b_h2 *= exp2(b_gk_last2)[None, :]
  ...

含义:
  每个 K channel 有独立的衰减因子 exp2(gk_last)。
  这允许模型对不同维度的 key 信息有不同的保留时间。
  信息衰减快的 channel 很快被遗忘，衰减慢的 channel 长期保留。
```

```
ASCII: State 矩阵的逐 channel 衰减

  State h [V, K]              gk_last [K]              衰减后 h [V, K]
  ┌───────────────────┐       ┌───────────┐           ┌───────────────────────┐
  │ ×  ×  ×  ...  ×  │       │ d1 d2 ... │           │ ×d1 ×d2 ...  ×dn      │
  │ ×  ×  ×  ...  ×  │  *    │ ...   dn  │           │ ×d1 ×d2 ...  ×dn      │
  │ ...               │       └───────────┘           │ ...                   │
  │ ×  ×  ×  ...  ×  │     (逐行广播)                 │ ×d1 ×d2 ...  ×dn      │
  └───────────────────┘                               └───────────────────────┘

  每个 V 行乘以相同的 gk_last [K] — 即每个 channel 的所有 V 元素以相同因子衰减
  (这是逐 channel 衰减，不是逐元素 — gk_last 的 shape 是 [K]，对 [V] 维度广播)
```

#### 步骤 4: 状态更新 `h += kg^T @ v_new` (第 266-288 行)

```
状态更新 (外积累加):
  h_new = h_decayed + K[t]^T @ v_new[t]

矩阵维度:
  - K[t]   → [K, BT]   (key 矩阵，行为 K，列为 BT tokens)
  - v_new  → [BT, BV]  (chunk t 的 v_new，V 维度的当前 tile)
  - h_new  → [BV, K]   (当前状态，V-tile x K-tile)

分块计算:
  使用 tl.dot(b_k, b_v) 然后转置:

  b_h1 += tl.trans(tl.dot(b_k1, b_v))   ← b_k1 [64, BT] @ b_v [BT, BV] → [64, BV] → trans → [BV, 64]
  b_h2 += tl.trans(tl.dot(b_k2, b_v))   ← b_k2 [64, BT] @ b_v [BT, BV]
  b_h3 += tl.trans(tl.dot(b_k3, b_v))
  b_h4 += tl.trans(tl.dot(b_k4, b_v))

  注意: k = kg = k * beta * exp2(gk_last - gk)
        (已由 Kernel 4 recompute_w_u_fwd 计算好)
```

```
ASCII: 状态更新的外积计算 (K^T @ v_new)

   K matrix [K, BT]               v_new [BT, BV]            State update
                                                                 [BV, K]
   ┌─────────────┐              ┌────────────┐          ┌───────────────────────┐
   │ k_tile_1    │              │            │          │ +k1^T@v  +k1^T@v  ...  │
   │ [64, BT]    │       @      │ v_new      │   →      │ ...                   │
   ├─────────────┤              │ [BT, BV]   │          ├───────────────────────┤
   │ k_tile_2    │              │            │          │ +k2^T@v  +k2^T@v  ...  │
   │ [64, BT]    │              └────────────┘          │ ...                   │
   ├─────────────┤                                      ├───────────────────────┤
   │ k_tile_3    │                                      │ +k3^T@v  +k3^T@v  ...  │
   │ [64, BT]    │                                      │ ...                   │
   ├─────────────┤                                      ├───────────────────────┤
   │ k_tile_4    │                                      │ +k4^T@v  +k4^T@v  ...  │
   └─────────────┘                                      └───────────────────────┘

   tl.dot 返回 [64, BV]，tl.trans 得到 [BV, 64]，累加到每个对应的 K-tile 块

   实际含义: h[i,j] += SUM_over_tokens_in_chunk( K[t, i] * v_new[t, j] )
   即 chunk 内所有 token 的 key-value 外积之和，加到状态矩阵对应位置
```

#### 步骤 5: Scalar Gate 补偿 (步骤 2 中的 b_v 变换)

```
注意: 步骤 2 中 b_v *= exp(b_g_last - b_g)[:, None] 实际上是在步骤 2 和步骤 3 之间执行的

时间线:
  ① 加载 u[t] → b_v
  ② b_v = u[t] - W @ h      → Delta Rule
  ③ b_v *= exp(b_g_last - b_g) ← Gate 补偿
  ④ (保存 v_new)             → 存到 v_new[t]，供 Kernel 6
  ⑤ h *= exp(b_g_last) / exp2(gk_last)  ← 状态衰减
  ⑥ h += K^T @ v_new         ← 状态更新
```

### 3.4 Epilogue: 最终状态写回 (第 291-307 行)

```
序列处理完所有 NT 个 chunk 后，最终状态 ht 写回 initial_state (in-place):

  initial_state[initial_state_indices[i_n], i_h, i_v*BV:(i_v+1)*BV, :] = final_h

使用 flat 1D store:
  p_ht = ht + i_v * BV * K + offset_k + tl.arange(0, BV * 64)
  tl.store(p_ht, b_h_flat.to(ht.dtype.element_ty))

目的: 为下一个 batch 提供正确的初始状态 (KV Cache 管理)。
```

### 3.5 为什么 state 需要衰减（遗忘机制）

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

Gate 的设计:
  - USE_G:   标量 gate exp(g_last)，所有 channel 相同衰减
  - USE_GK:  per-channel gate exp2(gk_last)，每个 channel 独立衰减
  - 本项目使用 USE_GK + exp2 (log2 空间)，数值更稳定
```

### 3.6 为什么需要保存 h 快照（供 Kernel 6 读取）

```
Kernel 6 (chunk_gla_fwd_kernel_o) 是 output kernel，计算:

  o[t] = q[t] * exp2(g[t]) @ h[t]^T + causal_local[t]

其中 h[t] 是第 t 个 chunk 的起始状态（即第 t-1 个 chunk 结束时的状态）。
Kernel 6 从 Kernel 5 写入的 h[B, NT, H, V, K] 中读取 h[t]：

  Kernel 5 循环中的保存时序:
    chunk 0: 保存 h[0] = initial_state,   → 计算 → h_new = updated
    chunk 1: 保存 h[1] = h_new,           → 计算 → h_new = updated
    chunk 2: 保存 h[2] = h_new,           → 计算 → h_new = updated
    ...

  Kernel 6 读取:
    o[chunk_t] 使用 h[chunk_t_index] (即 h[i_tg])
    其中 i_tg = i_b * NT + i_t  (非 varlen) 或直接 i_t (varlen)

  所以 h[t] 总是记录第 t 个 chunk 开始时的状态。
```

---

## 4. 关键代码对应

| 功能 | 行号 | 说明 |
|------|------|------|
| CTA 标识计算 | 81-82 | `i_v, i_nh = program_id(0), program_id(1)` |
| Varlen 序列边界 | 83-93 | 从 `cu_seqlens` 计算 `bos, eos, NT, boh` |
| State 寄存器分配 | 96-102 | `b_h1` ~ `b_h4`，每个 `[BV, 64]` float32 |
| Head/序列偏移计算 | 105-114 | `h`, `v`, `k`, `w`, `v_new` 基址 |
| Initial state 指针 | 116-122 | `h0` (读初始状态), `ht` (写最终状态) |
| 步骤 1: 加载初始状态 | 125-142 | `tl.make_block_ptr(h0, ...)` 分 4 个 K-tile |
| 步骤 2: 保存状态快照 | 149-167 | flat 1D store workaround |
| 步骤 3: Delta Rule (W @ h) | 169-191 | `b_v = tl.dot(...)` 累加，`b_v = u - b_v` |
| 步骤 4: 保存 v_new | 197-201 | `SAVE_NEW_VALUE` 控制是否保存 |
| 步骤 5a: 标量 gate | 204-218 | `b_v *= exp(b_g_last - b_g)`, `b_h *= exp(b_g_last)` |
| 步骤 5b: per-channel gate | 220-263 | `b_h *= exp2(b_gk_last)[None, :]` |
| 步骤 6: 状态更新 | 266-288 | `b_h += tl.trans(tl.dot(b_k, b_v))` |
| Epilogue: 写回 final state | 291-307 | flat 1D store 到 `ht` (initial_state in-place) |
| Python wrapper | 310-377 | 形状推断、grid 计算、参数分发 |
| NT_BUCKET 计算 | 374 | `(0 if NT <= 32 else (1 if NT <= 128 else 2))` |

### 4.1 Autotune 设计 (第 29-51 行)

```
configs=[
    triton.Config(
        {"BV": GDN_CHUNK_H_BV},         # 默认 32
        num_warps=GDN_CHUNK_H_NUM_WARPS, # 默认 4
        num_stages=GDN_CHUNK_H_NUM_STAGES, # 默认 2
    )
]

原因注释 (第 30-40 行):
  - 只有 1 个 config → autotune 不会多次运行 benchmark
  - 如果多个 config，autotune 在 benchmark 阶段多次执行 kernel
  - 每次执行都会 in-place 更新 initial_state，导致 state pool 损坏
  - 生产模型 (Kimi-Linear-48B) 上 clone state pool 会 OOM
  - NT_BUCKET 保留在 key 中，为未来 refactor 预留 (当有独立 output buffer 时可多 config)
```

---

## 5. 数据流图: 完整 Chunk 间状态流转

```
═══════════════════════════════════════════════════════════════════════════════════
                        Kernel 5: Delta Rule 跨 Chunk 递推
═══════════════════════════════════════════════════════════════════════════════════

    输入: k(=kg), w, u(=v), g, gk      输入: initial_state per sequence
    ┌──────────────────────────┐       ┌─────────────────────────────┐
    │ from Kernel 4            │       │ from KV Cache Pool          │
    │ (recompute_w_u_fwd)      │       │ [cache_size, H, V, K]       │
    └──────────────────────────┘       └─────────────────────────────┘
              │                                    │
              │                                    ▼
              │                       ┌─────────────────────────┐
              │                       │ Load to registers       │
              │                       │ b_h1~b_h4 [BV, 64]      │
              │                       │ = initial_state slice    │
              │                       └──────────┬──────────────┘
              │                                  │
              │              ╔═══════════════════╧═══════════════════╗
              │              ║     Chunk Loop: for i_t in 0..NT-1     ║
              │              ╠═══════════════════════════════════════╣
              │              ║                                       ║
              │              ║  ┌──────────────────────────────┐     ║
              │              ║  │ ① Save snapshot to h[NT,..]  │     ║
              │              ║  │   h[i_t, :, i_v*BV:..., :]   │     ║
              │              ║  │   = b_h1, b_h2, b_h3, b_h4   │     ║
              │              ║  └──────────────┬───────────────┘     ║
              │              ║                 │                     ║
              │  ┌───────────▼─────────────────▼─────────────────┐   ║
              │  │ ② Delta Rule: v_new = u - W @ h              │   ║
              │  │                                                 │   ║
              │  │   Load w[BT, K] in 4 tiles                     │   ║
              │  │   b_v = w1@h1^T + w2@h2^T + w3@h3^T + w4@h4^T │   ║
              │  │   Load u[t] [BT, V]                            │   ║
              │  │   b_v = u[t] - b_v       ← v_new (残差)       │   ║
              │  └──────────────┬────────────────────────────────┘   ║
              │                 │                                    ║
              │  ┌──────────────▼────────────────────────────────┐   ║
              │  │ ③ Optional: Save v_new to v_new[t]            │   ║
              │  │   (consumed by Kernel 6 output)               │   ║
              │  └──────────────┬────────────────────────────────┘   ║
              │                 │                                    ║
              │  ┌──────────────▼────────────────────────────────┐   ║
              │  │ ④ Gate Decay                                  │   ║
              │  │                                                │   ║
              │  │   IF USE_G:                                   │   ║
              │  │     b_v  *= exp(b_g_last - b_g)[:,None]       │   ║
              │  │     b_h  *= exp(b_g_last)    ← 标量衰减       │   ║
              │  │                                                │   ║
              │  │   IF USE_GK:                                  │   ║
              │  │     b_h  *= exp2(b_gk_last)[None,:]           │   ║
              │  │                     ← 逐 channel 衰减         │   ║
              │  └──────────────┬────────────────────────────────┘   ║
              │                 │                                    ║
              │  ┌──────────────▼────────────────────────────────┐   ║
              │  │ ⑤ State Update: h += kg^T @ v_new             │   ║
              │  │                                                │   ║
              │  │   Load kg[t] [K, BT] in 4 tiles               │   ║
              │  │   b_h1 += tl.trans( kg1@v_new )   [64,BV]→trans→[BV,64]
              │  │   b_h2 += tl.trans( kg2@v_new )               │   ║
              │  │   b_h3 += tl.trans( kg3@v_new )               │   ║
              │  │   b_h4 += tl.trans( kg4@v_new )               │   ║
              │  │                      ← 外积累加               │   ║
              │  └──────────────┬────────────────────────────────┘   ║
              │                 │                                    ║
              │  ╔══════════════╧════════════════════════════════╗   ║
              │  ║  Next iteration: updated b_h carries forward  ║   ║
              │  ╚═══════════════════════════════════════════════╝   ║
              │              ║                                       ║
              │              ╚═══════════════════════╤═══════════════╝
              │                                     │
              │                                     ▼
              │                       ┌─────────────────────────┐
              │                       │ Epilogue:               │
              │                       │ Write final state to    │
              │                       │ initial_state in-place  │
              │                       │ (for next batch reuse)  │
              │                       └─────────────────────────┘
              │
              ▼
    输出: h [B, NT, H, V, K]              输出: initial_state (updated)
    ┌──────────────────────────┐          ┌─────────────────────────────┐
    │ consumed by Kernel 6     │          │ consumed by next batch's    │
    │ (chunk_gla_fwd_o_gk)     │          │ inference as initial_state  │
    └──────────────────────────┘          └─────────────────────────────┘

    输出: v_new [B, T, H, V]
    ┌──────────────────────────┐
    │ consumed by Kernel 6     │
    │ (chunk_gla_fwd_o_gk)     │
    └──────────────────────────┘
```

### 5.1 Chunk 间状态传递详解

```
Timeline of state h across chunks for one (sequence, head, V-tile) CTA:

  Time ────────────────────────────────────────────────────────────────────►

  Chunk 0                  Chunk 1                  Chunk 2           ...
  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │                  │     │                  │     │                  │
  │ Load initial     │     │ h[0] from mem    │     │ h[1] from mem    │
  │ state from       │     │ = state after    │     │ = state after    │
  │ cache pool       │     │   chunk 0        │     │   chunk 1        │
  │        │         │     │        │         │     │        │         │
  │        ▼         │     │        ▼         │     │        ▼         │
  │ Save snapshot    │     │ Save snapshot    │     │ Save snapshot    │
  │ h[0] = state     │────►│ h[1] = state     │────►│ h[2] = state     │────► ...
  │        │         │     │        │         │     │        │         │
  │        ▼         │     │        ▼         │     │        ▼         │
  │ v_new₀ =         │     │ v_new₁ =         │     │ v_new₂ =         │
  │  u₀ - W₀ @ state │     │  u₁ - W₁ @ state │     │  u₂ - W₂ @ state │
  │        │         │     │        │         │     │        │         │
  │        ▼         │     │        ▼         │     │        ▼         │
  │ state *= decay₀  │     │ state *= decay₁  │     │ state *= decay₂  │
  │        │         │     │        │         │     │        │         │
  │        ▼         │     │        ▼         │     │        ▼         │
  │ state +=         │     │ state +=         │     │ state +=         │
  │  K₀^T @ v_new₀   │────►│  K₁^T @ v_new₁   │────►│  K₂^T @ v_new₂   │────► ...
  │                  │     │                  │     │                  │
  └─────────────────┘     └─────────────────┘     └─────────────────┘

  每个 chunk 内的顺序:
    1. SAVE  (保存上一个 chunk 结束时的 state)
    2. DELTA (计算 v_new = u - W @ state)
    3. DECAY (衰减 state)
    4. UPDATE (更新 state += K^T @ v_new)

  状态流转:
    初始值 → [Chunk0 更新] → h[1] → [Chunk1 更新] → h[2] → ... → h[NT-1]
                                                              ↓
                                                         写回 Cache Pool
```

### 5.2 序列间状态独立性

```
  Sequence 0:  h₀[0] → chunk 0 → h₀[1] → chunk 1 → h₀[2] → ... → final_h₀
  Sequence 1:  h₁[0] → chunk 0 → h₁[1] → chunk 1 → h₁[2] → ... → final_h₁
  ...
  Sequence N-1: hₙ₋₁[0] → chunk 0 → hₙ₋₁[1] → chunk 1 → ... → final_hₙ₋₁

  每个序列有独立的 initial_state (通过 initial_state_indices 索引)
  每个序列-头对由一个 CTA 处理，CTA 间完全独立，无同步
```

### 5.3 V 维度分块并行

```
  同一个 (sequence, head) 的 V 维度被 BV=32 分块:

  V-tile 0: CTA(i_v=0, i_nh) → 处理 h[0:32, :]
  V-tile 1: CTA(i_v=1, i_nh) → 处理 h[32:64, :]
  V-tile 2: CTA(i_v=2, i_nh) → 处理 h[64:96, :]
  ...

  各 V-tile 的状态矩阵行相互独立:
    h.row[i]  = h_prev.row[i] * decay + K[t]^T @ v_new[t, i]

  对每个 V-tile j:
    状态矩阵: h[:, K] 中行 [j*BV : (j+1)*BV]
    v_new 读取: v_new[t, :, j*BV : (j+1)*BV]
    K[t]^T @ v_new 的外积:
      h[j*BV : (j+1)*BV, :] += K[t]^T @ v_new[t, :, j*BV : (j+1)*BV]

  V-tile 之间无依赖，可安全并行。
```

### 5.4 Kernel 在整体 Pipeline 中的位置

```
KDA 前向传播 Pipeline:

  Input: q, k, v, g(gate), beta, scale, initial_state

  Step 1: Gate cumsum
    kda_gate_chunk_cumsum / chunk_local_cumsum
      g_raw → cumsum → gk (per-channel gate, log2 space)

  Step 2: Intra-chunk compute (Kernel 3 + Kernel 4 融合)
    chunk_kda_fwd_intra
      → w, u, Aqk, kg

  Step 3: ★ Kernel 5 — Delta Rule H (本 kernel) ★
    chunk_gated_delta_rule_fwd_h
      输入: kg, w, u, gk, initial_state
      输出: h [B, NT, H, V, K], v_new [B, T, H, V]
      更新: initial_state (in-place)

  Step 4: Output combine (Kernel 6)
    chunk_gla_fwd_o_gk
      输入: q, v_new, gk, h, Aqk
      输出: o [B, T, H, V]

  Output: o, final_state
```
