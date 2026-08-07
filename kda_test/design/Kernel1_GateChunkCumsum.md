# Kernel 1 核函数设计文档: Gate Chunk Cumsum

## 1. 输入输出定义

### 1.1 输入张量

| 参数名       | Shape               | Dtype          | 含义                                                   |
|-------------|---------------------|----------------|--------------------------------------------------------|
| `s` (g_org)  | `[B, T, H, K]`      | fp32/bf16      | 原始门控值（未经激活的 raw gate），`S = K` 即每个 head 的通道维 |
| `A_log`      | `[H]` (numel=H)     | fp32           | 每个 head 的对数尺度参数，控制衰减强度                        |
| `dt_bias`    | `[H * K]` (可选)    | fp32           | 每个 head 每个通道的偏置，形状平铺为 `[H*K]`                  |
| `scale`      | 标量 (可选)          | fp32           | 输出缩放因子，实际调用时固定为 `RCP_LN2 = 1.442695...`        |
| `cu_seqlens` | `[N+1]` (可选)      | int64          | 变长序列的累积长度（VARLEN 模式），`N` 为序列数                 |
| `chunk_indices` | `[NT, 2]` (可选) | int64          | 预计算的 chunk 索引表，每行 `[seq_id, chunk_start_time]`      |
| `lower_bound` | 标量 (可选)         | fp32           | 安全门控模式的下界，非 None 时启用 safe gate 替代 standard gate |
| `T`          | 标量               | int (meta)     | 每 batch 的时间步数（固定长度模式）                            |
| `H`          | 标量               | int (constexpr)| Head 数                                                  |
| `S`          | 标量               | int (constexpr)| 每 head 的通道数 (=K)                                      |
| `BT`         | 标量               | int (constexpr)| Chunk 大小 (=64)，必须为 2 的幂                              |
| `BS`         | 标量               | int (constexpr)| S 维度的 tile 大小 (=32，来自 BS_LIST=[32])                 |

### 1.2 输出张量

| 参数名 | Shape               | Dtype          | 含义                                                    |
|-------|---------------------|----------------|---------------------------------------------------------|
| `o` (g) | `[B, T, H, K]`    | fp32 (output_dtype) | 门控激活 + chunk 内累积求和的结果，已转换为 log2 空间          |

## 2. 分核并行策略

### 2.1 Grid 拓扑

```
Grid = (cdiv(S, BS), NT, B * H)
       ~~~~~~~~~~~  ~~  ~~~~~
           |         |     |
        S维分块     chunk   所有 (batch, head) 对
```

其中:
- `S = K`（通道维大小）
- `BS = 32`（编译期常量）
- `BT = 64`（chunk 大小）
- `NT`：chunk 总数
  - 固定长度模式: `NT = cdiv(T, BT)`
  - VARLEN 模式: `NT = len(chunk_indices)`（即所有序列的 chunk 总数）

### 2.2 各 CTA 职责

```
                     cdiv(S, BS) 个 CTA
              ┌──────────────────────────────┐
              │ CTA(0,0)  CTA(1,0)  ...      │  ← 处理 chunk 0, (B=0,H=0)
   NT 个      │ CTA(0,1)  CTA(1,1)  ...      │  ← 处理 chunk 1, (B=0,H=0)
   chunk      │   ...       ...     ...       │
              │ CTA(0,NT-1) CTA(1,NT-1) ...  │  ← 处理 chunk NT-1, (B=0,H=0)
              ├──────────────────────────────┤
              │ ...     (同样的 S 维切分)      │  ← 处理 (B=0,H=1) 的所有 chunk
              └──────────────────────────────┘
                          ...
              ┌──────────────────────────────┐
              │ ...     (同样的 S 维切分)      │  ← 处理 (B=B-1,H=H-1) 的所有 chunk
              └──────────────────────────────┘
                       B * H 个平面
```

每个 CTA 处理一个 `(chunk, head)` 对:
- 加载 `[BT, BS]` 大小的 tile，即 chunk 内 BT 个时间步、BS 个通道
- 对 tile 内的每个通道独立做门控激活 + chunk 内累加
- 输出同形状 tile

### 2.3 程序 ID 映射

```
i_s   = program_id(0)     → S 维分块索引 (0 .. cdiv(S,BS)-1)
i_t   = program_id(1)     → chunk 索引 (0 .. NT-1)
i_bh  = program_id(2)     → (batch, head) 联合索引 (0 .. B*H-1)
i_b   = i_bh // H         → batch 索引
i_h   = i_bh %  H         → head 索引
```

### 2.4 VARLEN 模式下的 Chunk 映射

VARLEN 模式下，`chunk_indices` 表将全局 chunk 索引 `i_t` 映射到具体序列和偏移:

```
i_n = chunk_indices[i_t, 0]    → 序列 ID
i_t_local = chunk_indices[i_t, 1]  → 该序列内的 chunk 偏移
bos = cu_seqlens[i_n]          → 该序列起始位置
eos = cu_seqlens[i_n+1]        → 该序列结束位置
T = eos - bos                  → 该序列的实际长度
```

## 3. 计算思路

### 3.1 总体数据流

```
raw_gate [B,T,H,K]  +  A_log [H]  +  dt_bias [H*K] (可选)
            |               |              |
            v               v              v
       ┌───────────────────────────────────────┐
       │  Step 1: 门控激活 (Gate Activation)     │
       │  tile: [BT, BS] per CTA               │
       └───────────────────────────────────────┘
                         |
                         v
       ┌───────────────────────────────────────┐
       │  Step 2: Chunk 内累积求和 (Cumsum)      │
       │  tl.cumsum(b_gate, axis=0)            │
       │  每个 chunk 独立，跨 chunk 边界不传递     │
       └───────────────────────────────────────┘
                         |
                         v
       ┌───────────────────────────────────────┐
       │  Step 3: 缩放 + log2 转换              │
       │  b_o *= RCP_LN2  (= 1.442695...)      │
       └───────────────────────────────────────┘
                         |
                         v
              output [B,T,H,K] dtype=fp32
```

### 3.2 数据划分示意（固定长度模式）

```
原始张量: [B, T, H, K]
           |   |  |  |
           |   |  |  +-- S 维 (通道)，按 BS=32 切分为 cdiv(S,BS) 块
           |   |  +----- H 维 (head)，每个 CTA 处理 1 个 head
           |   +-------- T 维 (时间)，按 BT=64 切分为 NT 个 chunk
           +------------ B 维 (batch)，每个 CTA 处理 1 个 batch

一个 CTA 处理的 tile:
  时间轴 (axis=0, BT=64)
  ┌─────────────────────┐
  │ t0    [k0..k31]     │  ← BS=32 通道
  │ t1    [k0..k31]     │
  │ ...                 │
  │ t63   [k0..k31]     │
  └─────────────────────┘
    ↑ 沿此轴做 cumulative sum
```

### 3.3 计算步骤详解

#### Step A: 门控激活

加载数据后，首先将 raw gate 与 dt_bias 相加（如果提供了 bias），然后执行门控激活。

**Standard Gate (USE_LOWER_BOUND = False)**:
```
gate = -exp(A_log) * softplus(raw_gate + dt_bias)

其中 softplus(x) = log(1 + exp(x))，x >= 20 时用 x 近似
```

公式展开:
```
gate(t, k) = -e^{A_log[h]} * log(1 + e^{raw_gate[t, k] + dt_bias[h, k]})
```

直观含义:
- `A_log[h]` 控制该 head 的整体衰减强度（负值越大，衰减越快）
- `softplus` 确保 gate 始终为正值
- 负号使 gate 为负值，后续 exp2(gate) 产生 (0,1] 范围的衰减因子
- 这是 KDA (Kronecker Delta Attention) 中 delta 规则的 chunk-wise 衰减门控

**Safe Gate (USE_LOWER_BOUND = True)**:
```
gate = lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))

其中 sigmoid(x) = 1 / (1 + exp(-x))
```

公式展开:
```
gate(t, k) = lower_bound * 1 / (1 + exp(-e^{A_log[h]} * (raw_gate[t, k] + dt_bias[h, k])))
```

直观含义:
- `exp(A_log) * raw_gate` 先将原始门控按 head 缩放
- `sigmoid` 保证输出在 (0, 1) 范围
- `lower_bound` 提供最小值保证，避免 gate 过小导致数值问题

#### Step B: Chunk 内累积求和

```python
b_o = tl.cumsum(b_gate, axis=0)
```

在每个 chunk 内部沿时间轴做前缀和 (prefix sum):

```
时间轴 (axis=0)
BT=4 示意:
  输入 b_gate:        输出 b_o:
  ┌───────────┐      ┌───────────┐
  │ g0 g0 g0  │      │ g0 g0 g0  │  ← b_o[0] = b_gate[0]
  │ g1 g1 g1  │      │g0+g1 ...  │  ← b_o[1] = b_o[0] + b_gate[1]
  │ g2 g2 g2  │      │g0+..+g2   │  ← b_o[2] = b_o[1] + b_gate[2]
  │ g3 g3 g3  │      │g0+..+g3   │  ← b_o[3] = b_o[2] + b_gate[3]
  └───────────┘      └───────────┘

每个 chunk 独立:
  chunk 0: cumsum over [t0  .. t63]
  chunk 1: cumsum over [t64 .. t127]  (从 0 重新开始)
  chunk 2: cumsum over [t128.. t191]  (从 0 重新开始)
```

关键点: cumsum 跨 chunk 边界**不传递**。chunk N 的首个元素始终等于 gate[N*BT]，而不是前一个 chunk 末尾的累积值。这使得后续的 chunk-wise attention 可以独立计算每个 chunk 内的衰减。

#### Step C: Log2 空间转换

```python
if HAS_SCALE:
    b_o *= scale  # scale = RCP_LN2 = 1.4426950216293335
```

乘以 `RCP_LN2` 将结果从 ln 空间转换到 log2 空间:
```
log2(x) = ln(x) / ln(2) = ln(x) * RCP_LN2
```

这样设计是因为后续的 chunk 内核使用 `exp2` 而非 `exp`，可以在硬件上更高效地计算。

### 3.4 VARLEN 模式下的内存访问示意

```
cu_seqlens = [0, 5, 12]  即 seq0 长度=5, seq1 长度=7

原始数据（平铺在 B=1 的维度中）:
  T 轴: [s0_t0, s0_t1, s0_t2, s0_t3, s0_t4, s1_t0, s1_t1, ..., s1_t6]
  chunk 0: [s0_t0 ... s0_t4] (实际只有 5 个时间步)
  chunk 1: [s1_t0 ... s1_t6] (实际只有 7 个时间步，但 BT=64，大部分 padding)

每次 CTA 启动时:
  i_n   = chunk_indices[i_t, 0]   → 确定属于哪个序列
  bos   = cu_seqlens[i_n]         → 该序列在平铺空间的起始偏移
  T_seq = cu_seqlens[i_n+1] - bos → 该序列长度
  p_s base = s + (bos * H + i_h) * S   → 从正确序列的起点开始取数据
```

## 4. 关键代码对应

### 4.1 核函数签名 (行 874-892)

```python
def kda_gate_chunk_cumsum_vector_kernel(
    s, A_log, dt_bias, o, scale, cu_seqlens, chunk_indices,
    lower_bound, T,
    H: tl.constexpr, S: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_LOWER_BOUND: tl.constexpr,
):
```

参数说明:
- `s`: 原始门控输入
- `A_log`: 每 head 对数尺度
- `dt_bias`: 可选偏置
- `o`: 输出张量
- `scale`: 可选缩放因子
- `cu_seqlens`: VARLEN 模式下的累积序列长度
- `chunk_indices`: 预计算的 chunk 索引表
- `lower_bound`: 安全门控下界
- `H`, `S`, `BT`, `BS`: 编译期常量，用于优化循环展开
- `HAS_BIAS`, `HAS_SCALE`, `IS_VARLEN`, `USE_LOWER_BOUND`: 编译期条件，通过 `@triton.heuristics` 注入

### 4.2 程序 ID 与索引计算 (行 893-906)

```python
i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
i_b, i_h = i_bh // H, i_bh % H

# VARLEN 模式：从 chunk_indices 表中解析序列 ID 和 chunk 偏移
if IS_VARLEN:
    i_n, i_t = (
        tl.load(chunk_indices + i_t * 2).to(tl.int32),
        tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
    )
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int32),
        tl.load(cu_seqlens + i_n + 1).to(tl.int32),
    )
    T = eos - bos
else:
    bos, eos = i_b * T, i_b * T + T
```

### 4.3 Tile 加载 (行 908-925)

使用 `tl.make_block_ptr` 创建 2D 块指针，实现高效的对齐内存访问:

```python
# 输入块指针: shape=(T,S), strides=(H*S, 1), offset=(i_t*BT, i_s*BS)
p_s = tl.make_block_ptr(
    s + (bos * H + i_h) * S,  # base = s[bos, i_h, 0]
    (T, S),                    # shape
    (H * S, 1),                # strides: 跨时间步跳过 H*S, 跨通道步长为 1
    (i_t * BT, i_s * BS),      # block offset: chunk 起始 + S 维偏移
    (BT, BS),                  # block shape
    (1, 0),                    # order: 时间轴连续（行优先）
)

# 加载 [BT, BS] tile，转为 fp32
b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
```

### 4.4 Bias 加载与加法 (行 927-937)

```python
if HAS_BIAS:
    p_b = tl.make_block_ptr(
        dt_bias + i_h * S,  # base = dt_bias[i_h, :]
        (S,),                # 1D shape: 通道维
        (1,),                # stride
        (i_s * BS,),         # offset: S 维分块偏移
        (BS,),               # block shape
        (0,),                # order
    )
    b_bias = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
    b_s = b_s + b_bias[None, :]  # 广播加法: [BT,BS] + [1,BS] → [BT,BS]
```

### 4.5 门控激活 (行 939-945)

```python
b_A = tl.load(A_log + i_h).to(tl.float32)  # 加载该 head 的 A_log

if not USE_LOWER_BOUND:
    # Standard gate: -exp(A_log) * softplus(g + bias)
    b_gate = -exp(b_A) * softplus_fwd(b_s)
else:
    # Safe gate: lower_bound * sigmoid(exp(A_log) * (g + bias))
    b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)
```

其中 `softplus_fwd` 定义在行 851-854:
```python
@triton.jit
def softplus_fwd(x):
    """Standard softplus: log(1 + exp(x)), with linear approx for large x."""
    return tl.where(x < 20.0, log(1.0 + exp(x)), x)
```

### 4.6 Chunk 内累积求和 (行 948)

```python
b_o = tl.cumsum(b_gate, axis=0)
```

Triton 内置的 `tl.cumsum` 沿 axis=0 (时间轴) 做前缀和。每个 chunk 独立，因为每个 CTA 只加载一个 chunk 的数据，不与其他 chunk 交互。

### 4.7 缩放与输出 (行 950-952)

```python
if HAS_SCALE:
    b_o *= scale  # scale = RCP_LN2, 转换为 log2 空间
tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
```

### 4.8 Python 包装函数 (行 955-1020)

```python
def kda_gate_chunk_cumsum(
    g, A_log, chunk_size, scale=None, dt_bias=None,
    cu_seqlens=None, output_dtype=torch.float,
    chunk_indices=None, lower_bound=None,
) -> torch.Tensor:
```

关键逻辑:
1. 输入 shape 验证: `B, T, H, S = g.shape`
2. Chunk 数量计算: `NT = cdiv(T, BT)` 或 VARLEN 模式 `len(chunk_indices)`
3. Chunk 大小必须为 2 的幂: 行 997-999
4. 输出分配: `torch.empty_like(g, dtype=output_dtype)` (行 1001)
5. Grid 定义: `(cdiv(S, BS), NT, B * H)` (行 1003-1004)

### 4.9 上层调用点 (行 1047-1059)

在 `chunk_kda_fwd` 中调用:
```python
g = kda_gate_chunk_cumsum(
    g,                  # raw gate [B,T,H,K]
    A_log=A_log,        # [H]
    chunk_size=64,      # BT = 64
    scale=RCP_LN2,      # 1.442695...
    dt_bias=dt_bias,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    lower_bound=lower_bound,
)
```

Scale 固定为 `RCP_LN2`，实现从 ln 空间到 log2 空间的转换，供后续 `exp2` 内核使用。

## 5. 数据流图

```
                        输入
                         |
        ┌────────────────┼────────────────┐
        |                |                |
        v                v                v
   raw_gate          A_log [H]       dt_bias [H*K]
  [B,T,H,K]              |            (可选)
        |                |                |
        |   ┌────────────┘                |
        |   |   ┌─────────────────────────┘
        |   |   |
        v   v   v
  ┌─────────────────────────────────────────────────┐
  │          Grid: (cdiv(S,BS), NT, B*H)            │
  │                                                 │
  │  CTA(i_s, i_t, i_bh) 处理:                      │
  │    - 时间维度: chunk i_t, BT=64 个时间步         │
  │    - 通道维度: BS=32 个通道, S 维偏移 i_s*32      │
  │    - batch: i_b = i_bh // H                    │
  │    - head:  i_h = i_bh %  H                     │
  │                                                 │
  │  ┌─────────────────────────────────────────┐   │
  │  │ 1. 加载 tile [BT, BS]                   │   │
  │  │    b_s = load(p_s)  ← raw_gate          │   │
  │  │    + bias if HAS_BIAS                    │   │
  │  └─────────────────────────────────────────┘   │
  │                    |                            │
  │                    v                            │
  │  ┌─────────────────────────────────────────┐   │
  │  │ 2. 门控激活                              │   │
  │  │    b_A = load(A_log[i_h])                │   │
  │  │    standard: -exp(b_A) * softplus(b_s)   │   │
  │  │    safe:     lb * sigmoid(exp(b_A)*b_s)  │   │
  │  └─────────────────────────────────────────┘   │
  │                    |                            │
  │                    v                            │
  │  ┌─────────────────────────────────────────┐   │
  │  │ 3. Chunk 内 cumulative sum              │   │
  │  │    b_o = tl.cumsum(b_gate, axis=0)      │   │
  │  │    (仅 chunk 内部，跨 chunk 边界归零)      │   │
  │  └─────────────────────────────────────────┘   │
  │                    |                            │
  │                    v                            │
  │  ┌─────────────────────────────────────────┐   │
  │  │ 4. 缩放 (log2 转换)                     │   │
  │  │    b_o *= RCP_LN2  if HAS_SCALE         │   │
  │  │    store(p_o, b_o)                      │   │
  │  └─────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────┘
                         |
                         v
                  output [B,T,H,K]
                  dtype = fp32
                  (已激活 + 已 cumsum + 已转换到 log2 空间)
```

### 跨 Chunk 边界示意

```
时间轴 T (假设 T=256, BT=64, NT=4):
  ┌────────┬────────┬────────┬────────┐
  │Chunk 0 │Chunk 1 │Chunk 2 │Chunk 3 │
  │t0..t63 │t64..127│t128..191│t192..255│
  └────────┴────────┴────────┴────────┘
       ↑        ↑        ↑        ↑
       |        |        |        |
   各 chunk 内部独立 cumsum，互不依赖

   Chunk 0 输出: cumsum(gate[0:64])     ← 起始于 gate[0]
   Chunk 1 输出: cumsum(gate[64:128])   ← 起始于 gate[64] (非 gate[63]的累加值)
   Chunk 2 输出: cumsum(gate[128:192])  ← 起始于 gate[128]
   Chunk 3 输出: cumsum(gate[192:256])  ← 起始于 gate[192]
```

### 两种门控模式对比

```
Standard Gate:
  raw_gate ──┬── softplus ── *(-exp(A_log)) ──→ gate (负值, 衰减)
  dt_bias ───┘

  公式: gate = -exp(A_log) * softplus(raw_gate + dt_bias)
  输出范围: (-inf, 0)
  后续: exp2(gate) → (0, 1] 作为衰减因子

Safe Gate:
  raw_gate ──┬── *exp(A_log) ── sigmoid ── *lower_bound ──→ gate (有下界)
  dt_bias ───┘

  公式: gate = lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))
  输出范围: (0, lower_bound) 或 (-lower_bound, 0) 取决于 lower_bound 符号
  用途: 避免 gate 绝对值过大导致 exp2 溢出或下溢
```

### 编译期特化路径

```
                    ┌──────────────┐
                    │   Kernel     │
                    │  Dispatcher  │
                    └──────┬───────┘
                           |
          ┌────────────────┼────────────────┐
          |                |                |
    HAS_BIAS?         HAS_SCALE?       IS_VARLEN?
    (T/F)              (T/F)            (T/F)
          |                |                |
    分支选择:         分支选择:         分支选择:
    - 加载 bias       - 不缩放           - 解析 cu_seqlens
    - 不加载 bias     - 乘以 RCP_LN2     - 固定 T
                      (实际总是 True)    - 解析 chunk_indices

                    USE_LOWER_BOUND?
                       (T/F)
                         |
                    - Standard gate
                    - Safe gate
```

所有编译期特化路径共 `2*2*2*2 = 16` 种组合，Triton 的 `@triton.heuristics` 将它们编译为特化版本，运行时零开销选择。

---

## 附录: 关键常量速查

| 常量          | 值                  | 用途                              |
|--------------|--------------------|-----------------------------------|
| `BS`         | 32                 | S 维 tile 大小                    |
| `BT`         | 64                 | Chunk 大小 / 时间维 tile 大小      |
| `RCP_LN2`    | 1.4426950216293335 | ln(2) 倒数, ln→log2 转换          |
| `num_warps`  | 1                  | 每个 CTA 使用 1 个 warp (32 threads) |
| `SOFTPLUS_THRESHOLD` | 20.0        | softplus 中线性近似的阈值            |
