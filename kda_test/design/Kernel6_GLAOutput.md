# Kernel 6 核函数设计文档: GLA Output

## 1. 输入输出定义

Kernel `chunk_gla_fwd_kernel_o` (Python wrapper: `chunk_gla_fwd_o_gk`) 是 KDA 分块线性注意力的最后一个核函数，负责将跨块历史信息和块内精确注意力组合成最终输出。

### 输入张量

| 参数 | 形状 | 数据类型 | 含义 |
|------|------|----------|------|
| `q` | `[B, T, H, K]` | bf16/fp16 | Query 向量，按 `[batch, token, head, key_dim]` 布局 |
| `v` | `[B, T, H, V]` | bf16/fp16 | Value 向量（经过 Kernel 5 Delta Rule 修正后的 v_new），按 `[batch, token, head, val_dim]` 布局 |
| `g` | `[B, T, H, K]` | fp32 | 累积 gate（来自 Kernel 1 chunk-local cumsum），每个 key channel 独立的衰减量 |
| `h` | `[NT, H, V, K]` | fp32 | 每个 chunk 开始时的压缩状态快照（Kernel 5 输出），`:,:,v,k` 含义：key channel k 对 value 维度 v 的累积贡献 |
| `A` | `[B, T, H, BT]` | fp32/b16 | chunk 内因果注意力矩阵 Aqk，`A[t, j]` = token t 对 token j 的注意力权重（下三角） |
| `cu_seqlens` | `[B+1]` | int64 | 变长序列的累积偏移量（可选，`IS_VARLEN` 时使用） |
| `chunk_indices` | `[NT, 2]` | int32 | 每个 tile 对应的 `(seq_index, chunk_index)` 对 |
| `scale` | 标量 | float | 注意力缩放的 `1/sqrt(K)` 因子 |

### 输出张量

| 参数 | 形状 | 数据类型 | 含义 |
|------|------|----------|------|
| `o` | `[B, T, H, V]` | bf16/fp16 | 最终注意力输出，每个 token 的 V 维向量 |

### 常量参数

| 符号 | 典型值 | 含义 |
|------|--------|------|
| `T` | 序列总长度（变长模式）或每个 batch 的固定长度 | 每个 batch 的 token 数 |
| `H` | 64 | 注意力头数 |
| `K` | 64 | Key/Query 维度（= head dim） |
| `V` | 64 | Value 维度（= K = head dim） |
| `BT` | 64 | Chunk 大小（每个 chunk 内的 token 数） |
| `BK` | 32 | K 维度 tile 大小（autotune） |
| `BV` | 32 | V 维度 tile 大小（autotune） |
| `B` | 1~8 | Batch 大小 |
| `NT` | `cdiv(T, BT)` | Chunk 总数 |

---

## 2. 分核并行策略

### 3D Grid 布局

```
Grid 维度: ( GV,   NT,   B * H )
           │      │       │
           │      │       └── 所有 (batch, head) 对并行
           │      └── 所有 chunk 并行
           └── V 维度 tile 并行

其中:
  GV  = cdiv(V, BV)    ← V 维度切分成小块
  NT  = cdiv(T, BT)    ← 序列切分成 chunk 数
```

每个 CTA (thread block) 处理**一个 (chunk, head, V-tile)** 的交集，具体映射关系：

```
program_id(0) = i_v   → 选择 V 维度的第 i_v 个 tile: [i_v*BV, (i_v+1)*BV)
program_id(1) = i_t   → 选择第 i_t 个 chunk:       token [i_t*BT, (i_t+1)*BT)
program_id(2) = i_bh  → 选择 (batch, head) 对:     i_b = i_bh // H, i_h = i_bh % H
```

### 3D 并行可视化

```
                      CTAs = B * H 个 (batch,head) 对
                  ┌─────────────────────────────┐
                  │  head=0    head=1   head=63 │
              b=0 │  CTA 0     CTA 1    CTA 63  │
              b=1 │  CTA 64    CTA 65   CTA 127 │
                  └─────────────────────────────┘

  每个 (b,h) 对内部:

          V 维度 tile 方向 →
       BV=0    BV=1   ...  BV=GV-1
    ┌───────┬───────┬───────┐
    │       │       │       │  chunk 0   (token 0..63)
    │       │       │       │  chunk 1   (token 64..127)
    │  CTA  │  CTA  │  CTA  │    ...
    │       │       │       │  chunk NT-1
    └───────┴───────┴───────┘
    ↓
  chunk 方向

每一个小格子是一个 CTA，输出 o 矩阵的一个 [BT=64, BV=32] 子块。
```

### 为什么是 3D 而不是 4D

注意 K 维度**没有被 Grid 并行化**——它使用了 **sequential loop**。

```
for i_k in range(tl.cdiv(K, BK)):
    # 加载 q[BT, BK], g[BT, BK], h[BV, BK]
    # 累加到 b_o: b_o += dot(q_gated[BT, BK], h_t[BV, BK])
```

这是因为 K 维度被拆成 BK=32 的小块后，顺序循环加载 h、q、g 三个 block_ptr。每个 i_k 迭代中只需要 `[BT,BK] * [BV,BK]^T` 的矩阵乘，计算量相对轻量。将 K 维度留作 sequential loop 不显著增加总延迟。

---

## 3. 计算思路

Kernel 6 的核心计算可以概括为**两路相加**：

```
o[t] = o_cross[t] + o_intra[t]
```

- **跨块部分 `o_cross`**：通过 query 从压缩状态 h 中提取所有历史 chunk 的信息
- **块内部分 `o_intra`**：在当前 chunk 内部做精确的因果注意力

### 3.1 跨块部分：从压缩状态 h 读取历史信息

#### h 矩阵的结构和含义

`h` 的形状是 `[NT, H, V, K]`，对于当前 CTA 对应的 chunk `i_t`：

```
h[chunk=i_t, head=i_h] 是一个 [V=64, K=64] 矩阵

列方向 (K=64): 每个 key channel
行方向 (V=64): 每个 value 维度

h[v, k] = key channel k 对 value 维度 v 的累积贡献

                         K 方向 (key channels) →
                      k=0      k=1    ...    k=63
                 ┌──────────────────────────────┐
            v=0  │  ch0→v0   ch1→v0  ...  ch63→v0 │
    V 方向   v=1  │  ch0→v1   ch1→v1  ...  ch63→v1 │
    ↓        ...  │   ...      ...    ...    ...   │
            v=63  │  ch0→v63  ch1→v63 ... ch63→v63│
                 └──────────────────────────────┘

解读: 第 k 列 = "如果 query 对 key channel k 的关注度为 1，
        那么这个 channel 的历史累积对各 value 维度的贡献是多少"
```

**h 的物理含义**：h 是 Kernel 5（Delta Rule）输出的**压缩状态快照**，记录了该 chunk 开始时刻之前所有 token 的累积信息。它把过去 T_history 个 token 的信息压缩成一个固定大小的 `[V, K]` 矩阵，实现了 O(T_history) → O(K*V) 的信息压缩。

#### Gated Query: `q * exp2(g)`

在查询 h 矩阵之前，query 向量需要先经过 gate 调制：

```
q_gated[BT, BK] = q[BT, BK] * scale * exp2(g[BT, BK])
```

Gate `g` 控制了"遗忘"程度：
- `g` 越负（衰减越大）→ `exp2(g)` 越小 → 这个 key channel 在历史信息中衰减得越多
- `g` 接近 0（几乎不衰减）→ `exp2(g)` 接近 1 → 完整保留关键信息

`scale = 1/sqrt(K)` 是标准注意力缩放因子，防止 q@h 的内积值随 K 增大而膨胀。

#### 分块矩阵乘法

跨块部分的核心计算：

```
o_cross[BT, V] = q_gated[BT, K] @ h[V, K]^T
```

由于 K 维度较大（64）且需要处理 tile 边界，实际采用分块乘法策略：

```
b_o = zeros([BT, BV])                    ← 累加器, fp32 精度

for i_k in range(0, K, step=BK):         ← K 维度循环, BK=32
    ┌────────────────────────────────────────────────────┐
    │  加载 q_gated [BT, BK]        从 [chunk_start, i_k*BK)│
    │  加载 h       [BV, BK]        从 [i_v*BV, i_k*BK)   │
    │                                                    │
    │  b_o += tl.dot(q_gated[BT,BK], h_t[BV,BK])        │
    │         └──── [BT, BK] @ [BK, BV] = [BT, BV] ────┘ │
    └────────────────────────────────────────────────────┘
```

每次迭代的矩阵乘：
- `q_gated` 形状 `[BT=64, BK=32]` — 当前 chunk 的一组 token 对 32 个 key channel 的 gated query
- `h^T` 形状 `[BK=32, BV=32]` — 这 32 个 key channel 对当前 BV=32 个 value 维度的贡献
- 结果累加到 `b_o[BT, BV]`

分块矩阵乘法示意图：

```
   o_cross [BT, BV]
  ┌──────────────────────┐
  │  .   .   .   .   .   │     q_gated [BT, K]        h^T [K, BV]
  │  .   .   .   .   .   │    ┌──────────────┐     ┌──────────────┐
  │  .   .   .   .   .   │    │  chunk tok0  │     │ k0  k1 ...   │
  │  .   .   .   .   .   │  = │  chunk tok1  │  X  │  ↓   ↓       │
  │  .   .   .   .   .   │    │    ...       │     │ v0  v1 ...   │
  └──────────────────────┘    │  chunk tok63 │     │ v0  v1 ...   │
                              └──────────────┘     │ ... ... ...  │
      [BT=64, BV=32]                               └──────────────┘
      K 维度按 BK=32
      分两次迭代累加                                  [K=64, BV=32]
```

### 3.2 块内部分：精确 chunk 内注意力

#### Aqk 因果矩阵结构

`A` (即 Aqk) 是一个**下三角矩阵**，表示 chunk 内 token 间的精确注意力关系：

```
Aqk [BT, BT] 结构（以 chunk_size=4 为例）:

         j=0    j=1    j=2    j=3
    ┌──────────────────────────┐
 i=0│  A00     0      0      0  │     A[i,j] = scale * q[i] . k[j] * exp2(g[i] - g[j])
 i=1│  A10    A11     0      0  │
 i=2│  A20    A21    A22     0  │     下三角: token i 只能看到 j <= i 的 token (因果约束)
 i=3│  A30    A31    A32    A33 │
    └──────────────────────────┘

m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    = [[True,  False, False, False],
       [True,  True,  False, False],
       [True,  True,  True,  False],
       [True,  True,  True,  True ]]
```

#### Causal mask 的应用

```python
b_A = tl.where(m_s, b_A, 0.0)  # 将上三角部分置零
```

实际计算中，`b_A` 是从 Kernel 3/4 预计算好的 Aqk 矩阵中加载的。虽然是下三角矩阵，但存储了整个 `[BT, BT]` 块（上三角也计算了但被 mask 掉）。通过因果 mask 确保 token i 不会"看到"未来的 token j > i。

#### 矩阵乘法

```
o_intra[BT, BV] = Aqk_masked[BT, BT] @ v_new[BT, BV]
```

```
   Aqk (因果 masked)          v_new                 o_intra
  ┌─────────────────┐    ┌───────────────┐    ┌───────────────┐
  │ A00   0    0   0│    │ v[0, 0..63]  │    │ o0,0 ... o0,63│
  │ A10  A11   0   0│  X │ v[1, 0..63]  │  = │ o1,0 ... o1,63│
  │ A20  A21  A22  0│    │ v[2, 0..63]  │    │ o2,0 ... o2,63│
  │ A30  A31  A32 A33│   │ v[3, 0..63]  │    │ o3,0 ... o3,63│
  └─────────────────┘    └───────────────┘    └───────────────┘
    [BT=64, BT=64]        [BT=64, V=64]         [BT=64, V=64]
                                                  只输出 BV tile
```

对于 token i（行 i）：
- `o_intra[i, :] = sum_j Aqk[i, j] * v_new[j, :]` (仅 j <= i)
- 这是标准的因果注意力：当前 token 对 chunk 内所有过去 token 的 value 加权求和

### 3.3 两部分相加得到最终输出

```
o[BT, BV] = o_cross[BT, BV] + o_intra[BT, BV]
```

两路计算完成后，结果写入输出 tensor：

```python
tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
```

存储时将 fp32 累加器转回 bf16/fp16。

### 完整两路计算结构图

```
                         ┌─────────────────────────────────┐
                         │         输入数据准备              │
                         └─────────────┬───────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                                     │
                    ▼                                     ▼
     ┌──────────────────────────┐          ┌──────────────────────────┐
     │   跨块路径 (o_cross)      │          │   块内路径 (o_intra)      │
     │                          │          │                          │
     │ q[BT,K] ──┬── g[BT,K]    │          │ Aqk[BT,BT]               │
     │           │               │          │   │                      │
     │           ▼               │          │   ▼ causal mask          │
     │  q_gated = q*scale*       │          │ Aqk_masked =             │
     │            exp2(g)        │          │   where(m_s, Aqk, 0)     │
     │           │               │          │   │                      │
     │           ▼               │          │   ▼                      │
     │  h[V,K] ──┤               │          │ dot(Aqk_masked, v_new)   │
     │           │               │          │   │                      │
     │  ┌────────┴────────┐      │          │   ▼                      │
     │  │ K-loop: i_k=0.. │      │          │ o_intra[BT,BV]           │
     │  │  dot(q_gated    │      │          │                          │
     │  │   [BT,BK],      │      │          └──────────┬───────────────┘
     │  │   h_t[BK,BV])   │      │                     │
     │  │  acc += result  │      │                     │
     │  └────────┬────────┘      │                     │
     │           │               │                     │
     │           ▼               │                     │
     │  o_cross[BT,BV]           │                     │
     │          │                │                     │
     └──────────┼────────────────┘                     │
                │                                      │
                └──────────────┬───────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ o = o_cross + o_intra│
                    │  (fp32累加)          │
                    │         │            │
                    │         ▼            │
                    │ store o[BT,BV]       │
                    │ (cast to bf16/fp16)  │
                    └─────────────────────┘
```

---

## 4. 关键代码对应

### 4.1 Python Wrapper: `chunk_gla_fwd_o_gk`

```python
def chunk_gla_fwd_o_gk(
    q, v, g, A, h, o, scale,
    cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    # 变长序列: 准备 chunk_indices 映射表
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    def grid(meta):
        return (cdiv(V, meta["BV"]), NT, B * H)
        #       V-tiles             NT   B*H

    chunk_gla_fwd_kernel_o[grid](...)
    return o
```

**关键设计**：
- `grid` 函数读取 autotune 确定的 `BV` 来动态调整 V 维度的切分
- `BT` 由用户指定（默认 64），不在 autotune 范围内
- `NT` 处理定长/变长两种模式
- `scale` 传入 `1/sqrt(K)`，统一在跨块部分使用

### 4.2 Kernel: `chunk_gla_fwd_kernel_o`

```python
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps)
        for BK in [16, 32]
        for BV in [32, 64]
        for num_warps in [1]
        for num_stages in [1]
    ],
    key=["BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(q, v, g, h, o, A, cu_seqlens,
                            chunk_indices, scale, T, H, K, V, BT, BK, BV,
                            IS_VARLEN):
```

**Autotune 配置**：
- `BK in [16, 32]`：K 维度 tile 大小
- `BV in [32, 64]`：V 维度 tile 大小
- `num_warps in [1]`：使用 1 个 warp（32 线程）
- `num_stages in [1]`：pipeline 1 个 stage
- Key 为 `["BT", "IS_VARLEN"]`：cache 条件为 chunk_size 和是否变长

### 4.3 CTA 绑定与索引计算

```python
i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
i_b, i_h = i_bh // H, i_bh % H
```

**变长模式**：
```python
if IS_VARLEN:
    i_tg = i_t  # 保存全局 tile index
    # chunk_indices[i_t] = (seq_index, chunk_index)
    i_n, i_t = tl.load(chunk_indices + i_t * 2), tl.load(chunk_indices + i_t * 2 + 1)
    bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
    T = eos - bos
    NT = tl.cdiv(T, BT)
```

**定长模式**：
```python
else:
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos, eos = i_b * T, i_b * T + T
```

`i_tg` (global tile index) 用于访问 `h` 矩阵的第 `i_tg` 个 chunk 状态。

### 4.4 因果 Mask 构造

```python
m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
# 生成 [BT, BT] 的下三角 bool 矩阵
```

### 4.5 K 维度循环（跨块计算）

```python
b_o = tl.zeros([BT, BV], dtype=tl.float32)  # fp32 累加器

for i_k in range(tl.cdiv(K, BK)):
    # 加载 q 和 g: [BT, BK] tile
    p_q = tl.make_block_ptr(
        q + (bos * H + i_h) * K,
        (T, K), (H * K, 1),          # shape & strides
        (i_t * BT, i_k * BK), (BT, BK), (1, 0)  # offsets & block shape
    )
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * K,
        (T, K), (H * K, 1),
        (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_qg = (b_q * exp2(b_g)).to(b_q.dtype)  # gated query

    # 加载 h: [BV, BK] tile
    # h 布局: [NT, H, V, K] → stride: (H*V*K, V*K, K, 1)
    p_h = tl.make_block_ptr(
        h + (i_tg * H + i_h) * V * K,
        (V, K), (K, 1),              # h[chunk] 是 [V, K] 矩阵
        (i_v * BV, i_k * BK), (BV, BK), (1, 0)
    )
    b_h = tl.load(p_h, boundary_check=(0, 1))

    # 矩阵乘: [BT, BK] @ [BK, BV] = [BT, BV]
    b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
```

**关键细节**：
- `h` 按 `[V, K]` 布局（行优先），stride 为 `(K, 1)`
- 需要 `tl.trans` 转置为 `[BK, BV]` 才能与 `q_gated[BT, BK]` 相乘
- `tl.trans(b_h).to(b_qg.dtype)` 先将 h 转置并类型转换为与 q_gated 一致，然后执行 `tl.dot`
- 注释 "works but dkw, owing to divine benevolence" 说明这个 trans + dot 的写法在 Triton 底层有特殊优化路径

### 4.6 块内计算与最终输出

```python
# 加载 v_new: [BT, BV] tile
p_v = tl.make_block_ptr(
    v + (bos * H + i_h) * V,
    (T, V), (H * V, 1),
    (i_t * BT, i_v * BV), (BT, BV), (1, 0)
)
b_v = tl.load(p_v, boundary_check=(0, 1))

# 加载 Aqk: [BT, BT] tile（整个下三角）
p_A = tl.make_block_ptr(
    A + (bos * H + i_h) * BT,
    (T, BT), (H * BT, 1),
    (i_t * BT, 0), (BT, BT), (1, 0)
)
b_A = tl.load(p_A, boundary_check=(0, 1))
b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)  # 施加因果 mask

# 矩阵乘: [BT, BT] @ [BT, BV] = [BT, BV]
b_o += tl.dot(b_A, b_v)

# 写出结果
tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
```

**关键细节**：
- Aqk 加载整个 `[BT, BT]` 块（不是只加载下三角部分），在 Triton 内通过 `tl.where` 施加 mask
- `b_o` 是 fp32 累加器，先累加 o_cross (K-loop)，再累加 o_intra (A @ v)
- 写出时转回存储类型的精度

---

## 5. 数据流图

### 5.1 端到端流水线中的位置

```
                              chunk_kda_fwd() 总流水线
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  raw_gate ──→ [Kernel 1] ──→ g (cumsum) ──┐                     │
  │                                            │                     │
  │  q,k,v,beta ──→ [Kernel 2] ──→ 对角线Aqk/Akk                    │
  │                     │                       │                    │
  │                     ▼                       │                    │
  │               [Kernel 3] ──→ Akk_inv        │                    │
  │                     │                       │                    │
  │                     ▼                       │                    │
  │               [Kernel 4] ──→ w,u,kg,Aqk ────┤                    │
  │                                            │                     │
  │  initial_state ──→ [Kernel 5] ──→ h, v_new │                    │
  │                     │                       │                    │
  │                     │         ┌─────────────┘                    │
  │                     │         ▼                                  │
  │                     │   ┌─────────────────────┐                  │
  │                     │   │  Kernel 6 (本核)     │                  │
  │                     │   │  chunk_gla_fwd_o_gk │                  │
  │                     └───┤  输入:              │                  │
  │                         │    q, v_new, g,     │                  │
  │                         │    Aqk, h           │                  │
  │                         │  输出: o [B,T,H,V]  │                  │
  │                         └─────────────────────┘                  │
  └──────────────────────────────────────────────────────────────────┘
```

### 5.2 单 CTA 内部数据流

```
  输入(由 CTA 坐标选定):
  ┌─────────────────────────────────────────────────────────┐
  │  chunk = i_t       (token窗口 [bos + i_t*BT, bos + (i_t+1)*BT))│
  │  v_tile = i_v      (V维窗口 [i_v*BV, (i_v+1)*BV))       │
  │  (batch, head) = (i_bh//H, i_bh%H)                      │
  └─────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                                       │
          ▼                                       ▼
  ┌───────────────────┐                 ┌───────────────────┐
  │ 跨块路径 (K-loop)  │                 │ 块内路径            │
  │                   │                 │                   │
  │ for i_k in K/BK:  │                 │ b_v = load(       │
  │   b_qg = q*scale* │                 │   v[chunk, V-tile]│
  │          exp2(g)  │                 │   )               │
  │          [BT,BK]  │                 │                   │
  │   b_h  = load(    │                 │ b_A = load(       │
  │   h[tile,         │                 │   Aqk[chunk, :BT] │
  │      V-tile,      │                 │   )               │
  │      K-tile]      │                 │ b_A = mask(b_A)   │
  │   ) [BV,BK]       │                 │                   │
  │                   │                 │ o_intra =         │
  │   b_o += dot(     │                 │   dot(A_masked,   │
  │     b_qg,         │                 │        b_v)       │
  │     b_h^T)        │                 │   [BT,BV]         │
  │                   │                 │                   │
  └────────┬──────────┘                 └────────┬──────────┘
           │                                     │
           │  b_o (fp32累加器)                     │
           └──────────────┬──────────────────────┘
                          │
                          ▼
                ┌───────────────────┐
                │  b_o = b_o +      │
                │   dot(b_A, b_v)   │
                │                   │
                │  store(o, b_o)    │
                │  (→ bf16/fp16)    │
                └───────────────────┘
```

### 5.3 跨块部分的 block_ptr 内存访问模式

```
q 和 g 的内存布局: [B, T, H, K], stride = (T*H*K, H*K, K, 1)

  对于 CTA (i_t=2, i_h=0):
                                        q [batch, token_range, head, K]
    block_ptr 参数:
      base  = q + (bos * H + i_h) * K  ← 定位到 batch 起始 + head
      shape = (T, K)                    ← 将此2D空间视为 [T, K] 矩阵
      stride = (H * K, 1)              ← T维步长 H*K, K维步长 1
      offsets = (i_t * BT, i_k * BK)   ← 当前chunk + 当前K-tile
      block = (BT, BK)                 ← 加载 BT×BK 的块
      order = (1, 0)                   ← K维连续(最内层), T维跳跃

h 的内存布局: [NT, H, V, K], stride = (H*V*K, V*K, K, 1)

  对于 CTA (i_tg=5, i_h=0, i_v=2):
    block_ptr 参数:
      base  = h + (i_tg * H + i_h) * V * K  ← 定位到 chunk + head
      shape = (V, K)                         ← 将此2D空间视为 [V, K] 矩阵
      stride = (K, 1)                        ← V维步长 K, K维步长 1
      offsets = (i_v * BV, i_k * BK)         ← 当前V-tile + 当前K-tile
      block = (BV, BK)                       ← 加载 BV×BK 的块
      order = (1, 0)                         ← K维连续, V维跳跃
```

### 5.4 两路计算语义对比

```
┌─────────────────────────────────────────────────────────────────┐
│                         Kernel 6 双路径语义对比                   │
├──────────────────────┬──────────────────────┬───────────────────┤
│       特性            │     跨块 (Cross)      │    块内 (Intra)    │
├──────────────────────┼──────────────────────┼───────────────────┤
│ 输入                 │ q, g, h              │ Aqk, v_new        │
│ 计算                 │ (q*exp2(g)) @ h^T    │ Aqk_masked @ v    │
│ 矩阵乘形状            │ [BT,BK] @ [BK,BV]    │ [BT,BT] @ [BT,BV] │
│ 覆盖范围              │ 所有历史 chunk       │ 当前 chunk 内      │
│ 时间复杂度            │ O(BT * K * V)        │ O(BT^2 * V)       │
│ K-loop               │ 有 (K/BK 次迭代)     │ 无                 │
│ 依赖                 │ q, g, h (Kernel 1,5) │ Aqk (Kernel 3)    │
│ 精度                 │ 近似 (压缩状态)       │ 精确 (因果矩阵)    │
└──────────────────────┴──────────────────────┴───────────────────┘
```

### 5.5 完整计算流程图（符号化）

```
输入:
  q [B, T, H, K]      ─── query 向量
  v_new [B, T, H, V]  ─── 修正后的 value (from Kernel 5)
  g [B, T, H, K]      ─── 累积 gate (from Kernel 1)
  h [NT, H, V, K]     ─── 压缩状态快照 (from Kernel 5)
  Aqk [B, T, H, BT]   ─── chunk 内因果注意力权重 (from Kernel 3)

变量:
  scale = 1/sqrt(K)
  m_s   = 因果 mask [BT, BT] (下三角 True)

流程:
  1. 初始化累加器: b_o = zeros([BT, BV], fp32)

  2. 跨块计算 (K 循环):
     for i_k in 0 .. K/BK - 1:
       q_tile  = q[chunk, i_k*BK : (i_k+1)*BK]    # [BT, BK]
       g_tile  = g[chunk, i_k*BK : (i_k+1)*BK]    # [BT, BK]
       h_tile  = h[chunk_id, BV_tile, K_tile]     # [BV, BK]
       q_gated = q_tile * scale * exp2(g_tile)    # [BT, BK]
       b_o    += q_gated @ h_tile^T               # [BT, BK] @ [BK, BV]

  3. 块内计算:
     v_tile = v_new[chunk, BV_tile]               # [BT, BV]
     A_tile = Aqk[chunk, :, :]                    # [BT, BT]
     A_mask = mask(A_tile, m_s)                   # 保留下三角
     b_o   += A_mask @ v_tile                     # [BT, BT] @ [BT, BV]

  4. 写出:
     o[chunk, BV_tile] = b_o                     # fp32 → bf16/fp16

输出:
  o [B, T, H, V]  ─── 最终注意力输出
```
