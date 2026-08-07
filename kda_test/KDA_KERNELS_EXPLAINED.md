# KDA Chunkwise Linear Attention: Kernel-by-Kernel 实现详解

## 1. 背景：为什么需要 Chunkwise Linear Attention

### 1.1 问题：标准 Attention 在处理长文本时太慢了

标准 Softmax Attention 的核心操作是：**每个 token 都要和所有 token 算一遍相似度**。

```
对于位置 t 的 token:
  output[t] = softmax(q[t] · k[0], q[t] · k[1], ..., q[t] · k[T-1]) @ v
```

如果序列有 T 个 token，每个 token 要和其他 T 个 token 做比较，总计算量是 **O(T²)**。当 T=128K 时，T² ≈ 160 亿，这在一个 GPU/NPU 上根本算不动。

### 1.2 朴素方法：逐 token 递归（KDA 想要逼近的目标）

**Prefill 阶段的输入**：用户输入一个 prompt，比如 "你好"，模型将其编码为若干个 token。假设只有 4 个 token（K=2, V=2, H=1），chunk_kda_fwd 收到的输入是：

```
模型参数（每个 head 一份，推理时固定）:
  A_log  = [-0.5]        ← 控制整体遗忘倾向
  dt_bias = [-0.3, 0.1]  ← 控制每个 channel 的遗忘速度

模型当前层输出（每个 token 不同）:
  token 0: q=[1, 0],  k=[1, 0],  v=[2, 1],  raw_gate=[1.0, 0.5],  beta=1.0
  token 1: q=[0, 1],  k=[0, 1],  v=[1, 3],  raw_gate=[0.8, 0.3],  beta=1.0
  token 2: q=[1, 1],  k=[1, 0],  v=[0, 2],  raw_gate=[2.0, 0.1],  beta=1.0
  token 3: q=[0, 0],  k=[-1, 1], v=[3, 0],  raw_gate=[1.5, 0.7],  beta=1.0

initial_state = [[0, 0], [0, 0]]   ← [K, V] 矩阵，prefill 首轮没有历史状态
scale = 1/sqrt(2) ≈ 0.707
```

**KDA 想要计算的数学目标**是一个递推公式（逐 token 递归）：

第一步：把 raw_gate 激活为 gate（Kernel 1 做的事情，稍后展开）：

```
gate[t] = -exp(A_log) * softplus(raw_gate[t] + dt_bias)   ← 保证 gate < 0
```

第二步：用 gate 做逐 token 递归：

```
state 是一个 [K, V] 矩阵（K=2 行, V=2 列），每行是一个 key channel 对各 value 维度的贡献
state = initial_state   ← [K, V]，prefill 首轮通常是零矩阵
for each token t:
  state   = state * exp(gate[t])                          (1) 遗忘: gate[K] 以列向量广播到 [K,V]
  predict = k[t] @ state                                  (2) [1,K] @ [K,V] → [1,V]，用 key 查 state
  delta   = v[t] - predict                                (3) 只保留"新信息"
  state   = state + k[t]^T @ delta * beta[t]              (4) 写入: outer product k^T[K,1] @ delta[1,V] → [K,V]
  o[t]    = q[t] @ state * scale                          (5) [1,K] @ [K,V] → [1,V]，用 query 查 state
```

**为什么用 [K, V] 布局？**

看所有矩阵乘法的维度：`q [1,K] @ state [K,V] = [1,V]`，`k [1,K] @ state [K,V] = [1,V]`。q、k、v、o 都是 1×V 的行向量，state 是 K×V，所有乘法方向一致，不需要转置。第 (4) 步的外积 `k^T @ delta` = `[K,1] @ [1,V]` = `[K,V]` 也自然正确。

> 实际 kernel 代码中 state 按 `[V, K]` 存储（`h = [NT, H, V, K]`），因为 K==V（都是 64），`[K,V]` 和 `[V,K]` 是方阵转置，乘法等价。

**用上面的真实输入走一遍**（先算 gate，再跑递归）：

```
===== Gate 激活（对所有 token 并行）=====
gate[t] = -exp(-0.5) * softplus(raw_gate[t] + [-0.3, 0.1])

token 0: gate[0] = -0.607 * softplus([1.0-0.3, 0.5+0.1])
                 = -0.607 * softplus([0.7, 0.6])
                 = -0.607 * [0.803, 0.737] = [-0.49, -0.45]

token 1: gate[1] = -0.607 * softplus([0.8-0.3, 0.3+0.1])
                 = -0.607 * [0.695, 0.613] = [-0.42, -0.37]

token 2: gate[2] = -0.607 * softplus([2.0-0.3, 0.1+0.1])
                 = -0.607 * [1.495, 0.398] = [-0.91, -0.24]

token 3: gate[3] = -0.607 * softplus([1.5-0.3, 0.7+0.1])
                 = -0.607 * [1.053, 0.703] = [-0.64, -0.43]
```

现在 gate 都有了，跑逐 token 递归：

```
初始: state = [[0, 0],    ← K=2 行 (key channel 0, channel 1)
               [0, 0]]    ← V=2 列 (value 维度 0, 维度 1)

===== Token 0 =====
(1) 遗忘: state = [[0,0],[0,0]] * exp([[-0.49], [-0.45]]) = [[0,0],[0,0]]
                ↑ gate 作为 [K,1] 列向量广播，每行（key channel）独立衰减

(2) 预测: predict = k[0] @ state = [1,0] @ [[0,0],[0,0]] = [0, 0]

(3) 求差: delta = [2, 1] - [0, 0] = [2, 1]

(4) 写入: state += k[0]^T @ delta = [[1],[0]] @ [2,1] = [[2, 1], [0, 0]]

(5) 输出: o[0] = q[0] @ state * 0.707 = [1,0] @ [[2,1],[0,0]] * 0.707 = [1.414, 0.707]
  ↑ [1,K] @ [K,V] → [1,V]，自然得到行向量

===== Token 1 =====
(1) 遗忘: state = [[2,1],[0,0]] * exp([[-0.42], [-0.37]])
                ≈ [[2*0.657, 1*0.657], [0*0.691, 0*0.691]]
                ≈ [[1.314, 0.657], [0, 0]]
      ↑ 第0行（key channel 0）衰减 34%，第1行（key channel 1）衰减 31%

(2) 预测: predict = k[1] @ state = [0,1] @ [[1.314,0.657],[0,0]] = [0, 0]
      ↑ k[1] 的第一维是 0，看不到 state 第0行的信息 → 预测为 0

(3) 求差: delta = [1, 3] - [0, 0] = [1, 3]

(4) 写入: state += k[1]^T @ delta = [[0],[1]] @ [1,3] = [[0,0],[1,3]]
        state = [[1.314, 0.657], [1, 3]]

(5) 输出: o[1] = q[1] @ state * 0.707 = [0,1] @ [[1.314,0.657],[1,3]] * 0.707 = [0.707, 2.121]

===== Token 2 =====
(1) 遗忘: state = [[1.314,0.657],[1,3]] * exp([[-0.91], [-0.24]])
                ≈ [[1.314*0.403, 0.657*0.403], [1*0.787, 3*0.787]]
                ≈ [[0.529, 0.265], [0.787, 2.361]]
      ↑ 第0行（key channel 0，gate=-0.91）遗忘了 60%，第1行（channel 1，gate=-0.24）只遗忘 21%

(2) 预测: predict = k[2] @ state = [1,0] @ [[0.529,0.265],[0.787,2.361]] = [0.529, 0.265]
      ↑ k[2] 第一维是 1，读到了 state 第0行的信息 → 预测成功！

(3) 求差: delta = [0, 2] - [0.529, 0.265] = [-0.529, 1.735]
      ↑ v[2] 第0维=0，state 预测了 0.529 → 向下修正
      ↑ v[2] 第1维=2，state 只预测了 0.265 → 补充 1.735
      ↑ 这就是"Delta"——只写入修正量

(4) 写入: state += k[2]^T @ delta = [[1],[0]] @ [-0.529,1.735] = [[-0.529, 1.735],[0, 0]]
        state = [[0.529-0.529, 0.265+1.735], [0.787+0, 2.361+0]]
              = [[0, 2], [0.787, 2.361]]

(5) 输出: o[2] = q[2] @ state * 0.707 = [1,1] @ [[0,2],[0.787,2.361]] * 0.707 = [1.783, 1.857]

===== Token 3 =====
(1) 遗忘: state = [[0,2],[0.787,2.361]] * exp([[-0.64], [-0.43]])
                ≈ [[0*0.527, 2*0.527], [0.787*0.651, 2.361*0.651]]
                ≈ [[0, 1.054], [0.512, 1.536]]

(2) 预测: predict = k[3] @ state = [-1,1] @ [[0,1.054],[0.512,1.536]] = [0.512, 0.482]

(3) 求差: delta = [3, 0] - [0.512, 0.482] = [2.488, -0.482]

(4) 写入: state += k[3]^T @ delta = [[-1],[1]] @ [2.488,-0.482] = [[-2.488, 0.482],[2.488, -0.482]]
        state = [[0-2.488, 1.054+0.482], [0.512+2.488, 1.536-0.482]]
              = [[-2.488, 1.536], [3, 1.054]]

(5) 输出: o[3] = q[3] @ state * 0.707 = [0,0] @ [[-2.488,1.536],[3,1.054]] * 0.707 = [0, 0]
```

**最终输出**：`o = [[1.414, 0.707], [0.707, 2.121], [1.783, 1.857], [0, 0]]`

这 4 个 token 的 o 就是 `chunk_kda_fwd` 应该产出的结果。这就是 Level 4 正确性测试的 ground truth——naive 逐 token 递归。

### 1.3 问题：逐 token 递归太慢，无法并行

上面的递归是 O(T·K·V) 但**严格串行**——token 0 算完 state 变了，才能算 token 1。在 GPU/NPU 上，99% 的计算单元在闲置。

而且 `state *= exp(gate[t])` 每一步都乘以 < 1 的数，128K 步后最早的信息会下溢到 0。

### 1.4 解决方案：分块计算（chunk_kda_fwd 的实际实现）

KDA 的方法：**把 prompt 的 T 个 token 切成 chunk，块内并行、块间串行**，输出和逐 token 递归一致。

设定 `chunk_size=2`（真实是 64），4 个 token 切成 2 个 chunk：

```
chunk 0 = [token0, token1]    chunk 1 = [token2, token3]
```

`chunk_kda_fwd` 用 6 个 kernel 来实现这段计算，从 prefill 的输入直接产出 `o [T, V]`：

---

**Step A — Kernel 1 (`kda_gate_chunk_cumsum`)**：激活 gate + chunk 内 cumsum

```
输入: raw_gate, A_log, dt_bias
输出: g_cumsum [T, K]  (chunk_kda_fwd 后续步骤全部用这个)

每个 chunk 内的 gate 独立做前缀和（chunk 间不累积）:

chunk 0:
  gate_raw[0] = -exp(-0.5) * softplus(raw_gate[0] + dt_bias) = [-0.49, -0.45]
  gate_raw[1] = -exp(-0.5) * softplus(raw_gate[1] + dt_bias) = [-0.42, -0.37]
  
  g_cumsum[0] = gate_raw[0]               = [-0.49, -0.45]
  g_cumsum[1] = gate_raw[0] + gate_raw[1] = [-0.91, -0.82]

chunk 1:
  gate_raw[2] = [-0.91, -0.24]
  gate_raw[3] = [-0.64, -0.43]
  
  g_cumsum[2] = gate_raw[2]               = [-0.91, -0.24]    ← 重新从 0 开始！
  g_cumsum[3] = gate_raw[2] + gate_raw[3] = [-1.55, -0.67]
```

---

**Step B — Kernel 2+3+4 (`chunk_kda_fwd_intra`)**：chunk 内的并行计算

chunk 0 和 chunk 1 **各自并行**，互不依赖。以 chunk 0 (token 0,1) 为例。

#### 为什么需要 w, u, kg

回顾 Step C（Delta Rule）需要做的事情——对每个 token i：

```
v_new[i] = u[i] - w[i] @ h          ← h 是 chunk 的初始状态（来自前面 chunk）
state    += kg[i]^T @ v_new[i]      ← 用 kg 更新状态
```

这里的关键问题是：**chunk 内的 token 之间有因果依赖**。Token 1 的 v_new 会受 token 0 的影响（因为 token 0 已经修改了 state），所以不能各自独立算 v_new。

我们需要找到一组"解耦后"的 w、u、kg，使得**每个 token 的 `u[i] - w[i] @ h` 等于逐 token 递归算出的 v_new[i]**，同时 chunk 内 token 间的因果依赖已经被吸收进 w 和 u 内部。

#### 推导：从逐 token 递归到线性系统

逐 token 递归公式（1.2 节）可以展开。对于 chunk 内的 token i（初始状态为 h）：

```
token 0:
  v_new[0] = v[0]*β[0] - k[0]*β[0]*exp2(gk[0]) @ h

token 1:
  state_0 = h * exp2(gk[0]) + k[0]^T @ v_new[0]
  v_new[1] = v[1]*β[1] - k[1]*β[1] @ state_0
           = v[1]*β[1] - k[1]*β[1]*exp2(gk[0]) @ h
             - (k[1]*β[1] · k[0]) * v_new[0]

token 2 (如果有):
  v_new[2] = v[2]*β[2] - k[2]*β[2]*exp2(gk[0]+gk[1]) @ h
             - (k[2]*β[2] · k[0])*exp2(gk[1]) * v_new[0]
             - (k[2]*β[2] · k[1]) * v_new[1]
```

观察规律：**v_new[i] 依赖于所有 v_new[j] (j < i)**，系数是 `(k[i]*β[i] · k[j]) * exp2(gk[i-1] - gk[j])`。如果定义：

```
Akk[i,j] = (k[i]*β[i]) · k[j] * exp2(gk[i] - gk[j])   (j <= i)
Akk[i,i] = (k[i]*β[i]) · k[i]                           (对角线)
```

那整个 chunk 的 v_new 满足一个**下三角线性系统**：

```
Akk @ v_new = u_raw - w_raw @ h

其中:
  u_raw[i] = v[i] * β[i]                           ← 耦合的 value
  w_raw[i] = k[i] * β[i] * exp2(gk[i])             ← 耦合的 key
  h        = chunk 的初始状态                         ← 来自前面 chunk
```

这个方程的含义：**Token i 的"原始 value" u_raw[i] 等于**：所有前面 token 的"新信息" v_new[j] 对 i 的贡献之和（通过 Akk[i,j] 加权），**再加上**历史状态 h 对 i 的贡献（w_raw[i] @ h）。

#### Akk_inv 的作用：解耦

两边同时左乘 Akk_inv：

```
v_new = Akk_inv @ u_raw - Akk_inv @ w_raw @ h
      =       u        -       w        @ h
```

定义解耦后的量：
```
w = Akk_inv @ (k * β * exp2(gk))     ← 解耦后的等效 key
u = Akk_inv @ (v * β)               ← 解耦后的等效 value
```

**核心效果**：
- 乘 Akk_inv 之前：v_new[i] 和 v_new[j] (j < i) 互相耦合，必须串行求解
- 乘 Akk_inv 之后：每个 token 的 v_new[i] = u[i] - w[i] @ h，**完全独立，可并行计算**

Akk_inv 把 chunk 内 64 个 token 之间的因果依赖关系"吸收"进了 w 和 u 里。这是一个纯代数变换，不丢失任何信息。

#### kg：用于更新 state

kg 负责在 chunk 结束时把 v_new 写入 state：

```
kg[i] = k[i] * exp2(gk_last - gk[i])
```

`exp2(gk_last - gk[i])` 把 token i 的 key 对齐到 chunk 最后一个 token 的时间点。这样 `kg^T @ v_new` 加到 state 后，state 的时间戳就是 chunk 结束时刻，下一个 chunk 可以直接用。

---

#### 用例子演算

理解了"为什么"之后，下面是具体数值（和之前一样）：

*Kernel 2 (token_parallel)*：计算 Akk/Aqk 的对角线块（sub-chunk 内）
```
Aqk[0,0] = q[0] · k[0]*exp2(g[0]-g[0]) * scale = [1,0]·[1,0]*exp2(0)*0.707 = 0.707
Aqk[1,0] = q[1] · k[0]*exp2(g[1]-g[0]) * scale = [0,1]·[1,0]*exp2([-0.42,-0.37])*0.707 = 0
Aqk[1,1] = q[1] · k[1]*exp2(g[1]-g[1]) * scale = [0,1]·[0,1]*0.707 = 0.707
Akk[0,0] = k[0]*β[0] · k[0]*exp2(0) = [1,0]·[1,0] = 1
Akk[1,0] = k[1]*β[1] · k[0]*exp2(g[1]-g[0]) = [0,1]·[0.747, 0] = 0
Akk[1,1] = k[1]*β[1] · k[1]*exp2(0) = [0,1]·[0,1] = 1
```

*Kernel 3 (inter_solve)*：求 Akk 矩阵的逆（解耦 chunk 内因果依赖）
```
Akk = [[1, 0],     Akk_inv = [[1, 0],
       [0, 1]]               [0, 1]]
```
这个例子中 Akk 恰好是对角阵（因为 k[0]=[1,0] 和 k[1]=[0,1] 正交），所以 Akk_inv = I。正交意味着 token 0 和 token 1 **没有因果耦合**——token 1 的 key 看不到 token 0 的 key channel，v_new[1] 不受 v_new[0] 影响。如果 k[0] 和 k[1] 不正交，Akk 的 off-diagonal 非零，Akk_inv 就会把这种耦合解掉。

*Kernel 4 (recompute_w_u)*：用逆矩阵算解耦后的 w, u, kg
```
w[0] = Akk_inv[0,0] * k[0]*β[0]*exp2(g[0]) = 1 * [1,0]*exp2([-0.49,-0.45]) = [0.712, 0]
w[1] = Akk_inv[1,0] * ... + Akk_inv[1,1] * k[1]*β[1]*exp2(g[1]) = [0, 0.566]

u[0] = Akk_inv[0,0] * v[0]*β[0] = 1 * [2,1] = [2, 1]
u[1] = Akk_inv[1,1] * v[1]*β[1] = 1 * [1,3] = [1, 3]

kg[0] = k[0] * exp2(g[1] - g[0]) = [1,0] * exp2([-0.42,-0.37]) = [0.747, 0]
kg[1] = k[1] * exp2(g[1] - g[1]) = [0,1] * exp2(0) = [0, 1]
```

有了 w 和 u，Step C 中对每个 token 只需算 `u[i] - w[i] @ h` 就得到 v_new[i]，不再需要处理 chunk 内 token 间依赖（已被 Akk_inv 解耦）。

同样，chunk 1 并行算出 w[2],w[3], u[2],u[3], kg[2],kg[3], Aqk（步骤完全一样）。

#### Aqk、Akk、Akk_inv 三者关系总结

这三个矩阵名字相似、同时计算，但服务于**不同的下游步骤**：

```
                    ┌──────────────────────────────────────┐
                    │  Kernel 2 (token_parallel)            │
                    │  输入: q, k, g, β                     │
                    │                                       │
                    │  共用计算: gated_k[j] = k[j] * exp2(g[i] - g[j])  │
                    │                                       │
                    │  Aqk[i,j] = scale * (q[i] · gated_k[j])    │
                    │  Akk[i,j] = (k[i]*β[i]) · gated_k[j]      │
                    └──────────────┬───────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
    ┌──────────────────┐                    ┌──────────────────┐
    │ Kernel 3/4       │                    │ Kernel 6 (Step D) │
    │ (Step B→C 的桥梁)│                    │ 最终输出          │
    │                  │                    │                  │
    │ Akk_inv = Akk⁻¹  │                    │ o[i] = Σ Aqk[i,j] │
    │      ↓           │                    │        * v_new[j] │
    │ w = Akk_inv @ w_raw  │                │ + q[i] @ h * scale│
    │ u = Akk_inv @ u_raw  │                └──────────────────┘
    │      ↓           │
    │   传给 Step C     │
    └──────────────────┘
```

| 矩阵 | 公式 | 形状 | 服务于 | 作用 |
|------|------|------|--------|------|
| **Aqk** | `scale * q[i] · k[j] * exp2(g[i]-g[j])` | [T, T] 下三角 | Step D (输出) | chunk 内 token i 对 token j 的注意力权重 |
| **Akk** | `(k[i]*β[i]) · k[j] * exp2(gk[i]-gk[j])` | [T, T] 下三角 | Step B (解耦) | chunk 内 token j 通过 state 更新对 token i 的因果影响 |
| **Akk_inv** | Akk 的逆矩阵 (forward substitution 求解) | [T, T] 下三角 | Step B→C | 把耦合的 k/v 变换为解耦的 w/u |

**一句话分清**：

- **Aqk** 是 Q-K 注意力矩阵 → 用于最终**输出**（"token i 应该关注 token j 多少"）
- **Akk** 是 K-K 因果依赖矩阵 → 描述**state 传递**中的 token 间耦合（"token j 的写入对 token i 的预测有多大影响"）
- **Akk_inv** 是 Akk 的逆 → **消除**这种耦合，使得 64 个 token 的 `v_new[i] = u[i] - w[i] @ h` 可以**并行**计算

Aqk 和 Akk 在 Kernel 2 中**同时计算**，因为它们用到了相同的 `k[j] * exp2(g[i]-g[j])`（gated key），只是分别与 q[i] 和 k[i]*β[i] 做点积。共享中间结果减少了对 HBM 的读取次数。

---

**Step C — Kernel 5 (`chunk_gated_delta_rule_fwd_h`)**：跨 chunk 传状态

这是**唯一串行**的部分——chunk 1 必须等 chunk 0 结束，因为要依赖 state。但这只跑 NT 步（chunk 的数量，如 2048），而不是 T 步（128K）。

```
state = initial_state = [[0, 0], [0, 0]]

===== Chunk 0 =====
h[0] = state = [[0, 0], [0, 0]]                             ← 保存快照

v_new[0] = u[0] - w[0] @ state
         = [2,1] - [0.712,0] @ [[0,0],[0,0]] = [2, 1]

v_new[1] = u[1] - w[1] @ state
         = [1,3] - [0,0.566] @ [[0,0],[0,0]] = [1, 3]
         ↑ state 是零，v_new 直接等于 u

state *= exp2(g_cumsum[1]) = [[0,0],[0,0]] * exp2([-0.91,-0.82]) = [[0,0],[0,0]]

state += kg^T @ v_new
       = [[0.747,0],[0,1]]^T @ [[2,1],[1,3]]   = [[1.494, 0.747],
                                                   [1,     3    ]]

===== Chunk 1 =====
h[1] = state = [[1.494, 0.747], [1, 3]]   ← 包含了 chunk 0 的信息！

v_new[2] = u[2] - w[2] @ state    ← 减去 state（含 chunk0 信息）能预测的部分
v_new[3] = u[3] - w[3] @ state

state *= exp2(g_cumsum[3])
state += kg[chunk1]^T @ v_new[chunk1]
```

---

**Step D — Kernel 6 (`chunk_gla_fwd_o_gk`)**：最终输出

```
chunk 0, token 0:
  跨块: q[0]*exp2(g_cumsum[0]) @ h[0] * scale
       = [1,0]*exp2([-0.49,-0.45]) @ [[0,0],[0,0]] * 0.707 = 0
       ↑ h[0] 是零（prefill 首轮没历史），跨块贡献为 0

  块内: Aqk[0,0] * v_new[0] = 0.707 * [2,1] = [1.414, 0.707]

  o[0] = 跨块 + 块内 = [1.414, 0.707]     ← 和逐 token 递归的 o[0] 一致！

chunk 1, token 2:
  跨块: q[2]*exp2(g_cumsum[2]) @ h[1] * scale
       = [1,1]*exp2([-0.91,-0.24]) @ [[1.494,0.747],[1,3]] * 0.707
       = [1.783, 1.857] 中的大部分来自跨块
       ↑ h[1] 包含了 chunk 0 的信息，token 2 通过 h[1] "看到"了 token 0 和 token 1

  块内: Aqk[2,2] * v_new[2] + (如果 token 2 前面还有 chunk 内的 token)

  o[2] = [1.783, 1.857]    ← 和逐 token 递归的 o[2] 一致！
```

**最终输出**：`chunk_kda_fwd` 返回 `o = [[1.414, 0.707], [0.707, 2.121], [1.783, 1.857], [0, 0]]`，和逐 token 递归完全一致。

---

#### 逐 token vs 分块的核心区别（用 token 2 对比）

| | 逐 token 方式 | 分块方式（chunk_kda_fwd） |
|---|---|---|
| token 2 怎么看到 token 0 和 1 | 通过 state2（串行累积了 0→1→2） | 通过 h[1]（chunk 0 结束时的 state，Kernel 5 传递） |
| token 2 怎么看到 chunk 内的 token 3 | 递归到 3 后 churn 看不到（因果） | 通过 Aqk 因果矩阵（Kernel 2/3 算出） |
| 并行度 | 每步 1 token，其余闲置 | 每个 chunk 64 token 同时算 |
| 代价 | 无额外开销 | 需多算 Aqk 和 Akk_inv（Kernel 2/3/4，都是矩阵乘法，GPU/NPU 擅长） |
| 输出 | o [T, V] | o [T, V]（**完全一致**） |

每个 chunk 内部又分成 4 个 sub-chunk（每个 16 token），进一步利用寄存器级别的矩阵乘法。这将在第 5 节（Inter Solve 内核）详细展开。

### 1.5 从头到尾走一遍（用最小例子）

下面用一个极小的例子（T=4 个 token, chunk_size=2, K=V=2, H=1）完整走一遍 KDA 的计算过程。虽然真实场景是 128K token 和 64 的 chunk_size，但核心逻辑完全一样。

**初始数据**：

```
token 0:  q=[1, 0],  k=[1, 0],  v=[2, 1],  gate=[-0.5, -0.3]
token 1:  q=[0, 1],  k=[0, 1],  v=[1, 3],  gate=[-0.4, -0.6]
token 2:  q=[1, 1],  k=[1, 1],  v=[0, 2],  gate=[-0.3, -0.2]
token 3:  q=[0, 0],  k=[-1, 1], v=[3, 0],  gate=[-0.1, -0.7]

chunk_size=2, 所以 chunk 0 = [token0, token1], chunk 1 = [token2, token3]
initial_state = [[0, 0], [0, 0]]  (2×2 零矩阵)
beta = [1, 1, 1, 1]  (简化，不做额外门控)
scale = 1/sqrt(2) ≈ 0.707
```

---

#### Step A: Gate Cumsum → 为每个 chunk 独立做前缀和

**输入**: `raw_gate [4,2]`
**输出**: `g_cumsum [4,2]`
**Kernel**: `kda_gate_chunk_cumsum` (Kernel 1)

对 chunk 0（token 0,1）和 chunk 1（token 2,3）各自独立做 cumsum：

```
chunk 0:
  g_cumsum[0] = gate[0]       = [-0.5,  -0.3]
  g_cumsum[1] = gate[0]+gate[1] = [-0.9,  -0.9]

chunk 1:
  g_cumsum[2] = gate[2]       = [-0.3,  -0.2]
  g_cumsum[3] = gate[2]+gate[3] = [-0.4,  -0.9]
```

**关键**：chunk 1 的 cumsum 从 0 开始，不包含 chunk 0 的累积。这保证了数值稳定性——gate 累积不会跨 chunk 无限增长。

```
g_cumsum 结果:
token 0: [-0.5, -0.3]
token 1: [-0.9, -0.9]
token 2: [-0.3, -0.2]
token 3: [-0.4, -0.9]
```

---

#### Step B: chunk_kda_fwd_intra → 块内求解

**输入**: `q, k, g_cumsum, beta`
**输出**: `w, u, kg, Aqk`
**Kernel**: Kernel 2 (token_parallel) + Kernel 3 (inter_solve) + Kernel 4 (recompute_w_u)

##### B1 — Token Parallel (Kernel 2)：计算 chunk 内 token 两两关系

对 chunk 0（token 0,1），token 1 需要知道它和 token 0 的关系：

```
计算 Aqk[1,0]: query token 1 和 key token 0 的 gated dot product

  exp2(g_cumsum[1] - g_cumsum[0]) = exp2([-0.9-(-0.5), -0.9-(-0.3)])
                                   = exp2([-0.4, -0.6])
                                   ≈ [0.758, 0.660]

  gated_k0 = k[0] * exp2(g1 - g0) = [1,0] * [0.758, 0.660] = [0.758, 0]

  Aqk[1,0] = (q[1] · gated_k0) * scale
           = ([0,1] · [0.758, 0]) * 0.707
           = 0 * 0.707 = 0

计算 Akk[1,0]: key token 1 和 key token 0 的 gated 关系

  Akk[1,0] = (k[1]*beta[1]) · gated_k0
           = [0,1] · [0.758, 0] = 0
```

（数值被刻意简化了，真实场景中这些值通常非零。重点是理解计算流程。）

##### B2 — Inter Solve (Kernel 3)：求 Akk 矩阵的逆

chunk 0 的 Akk 矩阵（2×2，chunk_size=2）：

```
Akk = [[Akk[0,0],     0    ],    = [[(k0·k0),  0  ],
       [Akk[1,0], Akk[1,1]]]      [   0   , (k1·k1)]]

     = [[1, 0],    ← k[0]·k[0] = 1²+0² = 1
        [0, 1]]    ← k[1]·k[1] = 0²+1² = 1
```

Akk 求逆很简单（对角矩阵）：

```
Akk_inv = [[1, 0],
           [0, 1]]
```

##### B3 — Recompute W/U (Kernel 4)：用逆矩阵重算 w, u, kg

```
chunk 0:
  w = Akk_inv @ (k * beta * exp2(g_cumsum))
    = [[1,0],[0,1]] @ [[1*exp2(-0.5), 0*exp2(-0.3)],
                        [0*exp2(-0.9), 1*exp2(-0.9)]]
    ≈ [[1,0],[0,1]] @ [[0.707, 0], [0, 0.536]]
    = [[0.707, 0], [0, 0.536]]

  u = Akk_inv @ (v * beta)
    = [[1,0],[0,1]] @ [[2,1], [1,3]]
    = [[2,1], [1,3]]

  kg = k * exp2(gk_last - g_cumsum)
    gk_last = g_cumsum[1] = [-0.9, -0.9]
    kg[0] = [1,0] * exp2([-0.9-(-0.5), -0.9-(-0.3)]) = [0.758, 0]
    kg[1] = [0,1] * exp2([-0.9-(-0.9), -0.9-(-0.9)]) = [0, 1]    ← 最后一个 token 的 kg 就是原始 k
```

chunk 1 同理（独立计算，不需要 chunk 0 的信息）。

**此时 Step B 输出**：

```
w[0]: [0.707, 0]    w[1]: [0, 0.536]    w[2]: [...]  w[3]: [...]
u[0]: [2, 1]        u[1]: [1, 3]        u[2]: [...]  u[3]: [...]
kg[0]: [0.758, 0]   kg[1]: [0, 1]       kg[2]: [...]  kg[3]: [...]
```

---

#### Step C: Delta Rule (Kernel 5) → 跨 chunk 传递状态

**输入**: `kg, w, u, g_cumsum, initial_state`
**输出**: `h (每个 chunk 的状态快照), v_new`
**Kernel**: `chunk_gated_delta_rule_fwd_h` (Kernel 5)

```
state = [[0,0],[0,0]]    ← initial_state

===== Chunk 0 =====
h[0] = state = [[0,0],[0,0]]

v_new[0] = u[0] - w[0] @ state = [2,1] - [0.707,0] @ [[0,0],[0,0]] = [2,1]
v_new[1] = u[1] - w[1] @ state = [1,3] - [0,0.536] @ [[0,0],[0,0]] = [1,3]
  ↑ 因为 state 是零矩阵，v_new 就等于 u（历史为空，没有可减的）

state *= exp2(gk_last) = [[0,0],[0,0]] * exp2([-0.9,-0.9]) = [[0,0],[0,0]]
  ↑ state 本来就是零，衰减后还是零

state += kg^T @ v_new = [[0.758,0],[0,1]]^T @ [[2,1],[1,3]]
       = [[0.758, 0],   @ [[2,1],
          [0,    1]]      [1,3]]

     = [[0.758*2 + 0*1, 0.758*1 + 0*3],    = [[1.516, 0.758],
        [0*2 + 1*1,     0*1 + 1*3]]         [1,     3    ]]
  ↑ state 现在包含了 chunk 0 的信息

===== Chunk 1 =====
h[1] = state = [[1.516, 0.758], [1, 3]]   ← 此时 state 带有 chunk 0 的信息！

v_new[2] = u[2] - w[2] @ state    ← 从 u[2] 中减去 state 能预测的部分
v_new[3] = u[3] - w[3] @ state    ← 这就是 Delta Rule 的 "Delta"

  ↑ 如果 chunk 0 的信息已经能预测 chunk 1 的内容，v_new 就会很小
  ↑ 这就是"只保留新信息"的关键机制

state *= exp2(gk_last_of_chunk1)    ← 再次衰减
state += kg[chunk1]^T @ v_new[chunk1]    ← 加入 chunk 1 的新信息
```

**此时 Step C 输出**：

```
h[0] = [[0, 0], [0, 0]]          ← chunk 0 开始时 state 是零
h[1] = [[1.516, 0.758], [1, 3]]  ← chunk 1 开始时 state 包含 chunk 0 的信息

v_new[0] = [2, 1]
v_new[1] = [1, 3]
v_new[2] = [...]    ← u[2] 减去 state 预测后的剩余
v_new[3] = [...]    ← u[3] 减去 state 预测后的剩余
```

---

#### Step D: GLA Output (Kernel 6) → 最终输出

**输入**: `q, v_new, g_cumsum, Aqk, h`
**输出**: `o [4,2]` (每个 token 的注意力输出)
**Kernel**: `chunk_gla_fwd_o_gk` (Kernel 6)

```
Chunk 0, token 0:
  跨块: q[0]*exp2(g[0]) @ h[0] * scale = [1,0]*exp2([-0.5,-0.3]) @ [[0,0],[0,0]] * 0.707
       = 0 (h[0] 是零矩阵)

  块内: Aqk[0,0] @ v_new[0] = Aqk[0,0] * [2,1]
        (Aqk[0,0] 是 q[0] 和 k[0] 自身的注意力权重)

  o[0] = 跨块 + 块内

Chunk 0, token 1:
  跨块: q[1]*exp2(g[1]) @ h[0] * scale → 0 (h[0] 是零)

  块内: Aqk[1,0] * v_new[0] + Aqk[1,1] * v_new[1]
        ↑ token 1 能看到 token 0 和 token 1

  o[1] = 跨块 + 块内

Chunk 1, token 2:
  跨块: q[2]*exp2(g[2]) @ h[1] * scale
       = [1,1]*exp2([-0.3,-0.2]) @ [[1.516, 0.758], [1, 3]] * 0.707
       ↑ 非零！因为 h[1] 包含了 chunk 0 的信息
       ↑ token 2 虽然没有直接看 token 0 和 token 1，但通过 h[1] 间接获取了它们的信息

  块内: Aqk[2,2] * v_new[2]

  o[2] = 跨块 + 块内
```

**关键洞察**：token 2 的跨块注意力能看到 token 0 和 token 1，不是直接看的（它们不在同一个 chunk），而是通过 `h[1]` 这个压缩状态间接看的。`h[1]` 把 2 个历史 token 的信息压缩成了一个 2×2 的矩阵。

---

#### 总结：6 个 Kernel 的输入输出关系

```
raw_gate ──→ [Kernel 1: Gate Cumsum] ──→ g_cumsum ──┐
                                                      │
q, k, beta ──────────────────────────────────────┬────┤
                                                  │    │
            ┌─────────────────────────────────────┘    │
            ▼                                          ▼
     [Kernel 2: Token Parallel]              g_cumsum 传入
      输出: Aqk (对角线), Akk (对角线)
            │
            ▼
     [Kernel 3: Inter Solve]
      输入: q, k, g_cumsum, beta, Akk(对角线), Aqk(对角线)
      输出: Akk_inv (完整逆矩阵)
            │
            ▼
     [Kernel 4: Recompute W/U]
      输入: k, v, beta, Akk_inv, g_cumsum
      输出: w, u, kg
            │
            ▼
     [Kernel 5: Delta Rule H]  ←── initial_state (上一个 batch 的记忆)
      输入: kg, w, u, g_cumsum, initial_state
      输出: h (每个 chunk 的状态快照), v_new
            │
            ▼
     [Kernel 6: GLA Output]
      输入: q, v_new, g_cumsum, Aqk, h
      输出: o (最终注意力输出)
```

注意 Kernel 2 和 Kernel 3/4 之间有一个**信息扩大**的过程：

- Kernel 2 只算了对角线的 Aqk/Akk（chunk 内同一 sub-chunk 的 token 关系）
- Kernel 3 算出了 off-diagonal 的 Aqk/Akk（chunk 内不同 sub-chunk 的 token 关系），并求出了完整 Akk 矩阵的逆
- Kernel 4 用这个完整逆矩阵，一次性算出所有 token 的 w/u/kg

这就是分块计算的精髓：**先用 Kernel 2 算小块内部关系（便宜），再用 Kernel 3 把小块拼成大块的解（利用块三对角结构，也比直接算便宜），最后 Kernel 4 用大块解一次性产出所有 token 的结果**。

### 1.6 真实推理：Prefill 和 Decode 分别走什么路径

上面 1.5 的例子走完了 `chunk_kda()` 的 6 个 kernel。但在真实推理（比如 Kimi K3）中，这 6 个 kernel 只在 **Prefill 阶段**使用。Decode 阶段走的是另一条更简单的路径。

#### Prefill 阶段：6 个 kernel 全走

用户输入一个 prompt，比如 4096 个 token。模型一次性收到全部 token：

```
用户输入: "请帮我写一篇关于人工智能的文章，从历史发展讲起..."
         └────────── 4096 个 token ──────────┘

模型:
  initial_state = 零矩阵（上一轮对话已结束，没有历史状态）
  调用 chunk_kda(q, k, v, g, beta, initial_state=零)
    └── 走 6 个 kernel 的完整 chunkwise 流水线
    └── 4096 token ÷ 64 chunk_size = 64 个 chunk
    └── 输出: o [4096, V]  ← 每个 token 的注意力输出
            state_final     ← 最后一个 chunk 结束时的状态
```

Prefill 的特点：**T 很大（几千到十几万），initial_state 是零，全部 token 一次性并行处理**。

#### Decode 阶段：不走 chunkwise，走 recurrent

Prefill 完成后，模型开始逐 token 生成回复：

```
生成 token "人":
  输入: 只有 1 个新 token（q,k,v,g,beta 全部 T=1）
  initial_state = prefill 产出的 state_final（包含了前 4096 个 token 的所有信息）
  调用 fused_recurrent_kda(q, k, v, g, beta, initial_state)
    └── 这是一个单独的轻量 kernel，只做一次 Delta Rule 递归
    └── T=1，不需要分 chunk，不需要任何矩阵求逆
    └── 输出: o [1, V], state_new（原地更新 initial_state）

生成 token "工":
  initial_state = 上一步更新后的 state（现在包含 4097 个 token 的信息）
  同样调用 fused_recurrent_kda → o, state_new

生成 token "智":
  initial_state = 包含 4098 个 token 信息的 state
  同样调用 fused_recurrent_kda → ...

... 每个新 token 都只做一次简单的 recurrent 更新
```

Decode 的特点：**T=1（每次只有 1 个 token），initial_state 来自上一步（不为零），不需要 chunkwise 分块**。

#### 对比

| | Prefill | Decode |
|---|---|---|
| 每次处理的 token 数 | 全部 prompt（几千到十几万） | 1 个 |
| 调用的函数 | `chunk_kda()` | `fused_recurrent_kda()` |
| 用了哪些 kernel | 全部 6 个 | 1 个（fused_recurrent 融合了遗忘+预测+求差+写入+输出） |
| initial_state | 通常为零矩阵 | 上一个 batch 的 final_state |
| 为什么走不同的路径 | T 大 → 需要分块并行加速 | T=1 → 分块毫无意义，直接递归更快 |
| 核心操作 | 矩阵乘法和求逆（GPU/NPU 擅长） | 向量-matrix 乘法（轻量） |

#### 为什么 initial_state 在 decode 中这么重要

用 1.5 的例子说明。假设 T=4 的 prompt 处理完后，state 变成了：

```
state_final = [[0,    2.10],    ← 包含了 token0~3 的所有 key-value 信息
               [0.45, 1.65]]
```

现在 decode 阶段来了第 5 个 token：

```
新 token: q=[0.5, 0.8], k=[0.3, -0.1], v=[1.5, 0.2], gate=[-0.2, -0.4]

fused_recurrent_kda 做的事情（T=1，极简）:

(1) 遗忘: state *= exp(gate)  (K=2 个 channel 独立衰减)
    state = [[0,    0.45],   * [[exp(-0.2)],    ≈ [[0,    0.37],
             [2.10, 1.65]]     [exp(-0.4)]]       [1.41, 1.11]]
      ↑ gate 为 [K,1] 列向量，每行独立衰减

(2) 预测: predicted = k @ state = [0.3, -0.1] @ [[0, 0.37],[1.41, 1.11]]
           ≈ [-0.141, 0]

(3) 求差: residual = [1.5, 0.2] - [-0.141, 0] = [1.641, 0.2]

(4) 写入: state += k^T @ residual = [[0.3],[-0.1]] @ [1.641, 0.2]
         ≈ [[0.49, 0.06], [-0.16, -0.02]]
         state ≈ [[0.49, 0.43], [1.25, 1.09]]

(5) 输出: output = q @ state = [0.5, 0.8] @ [[0.49, 0.43],[1.25, 1.09]] ≈ [1.25, 1.09]
```

这个新 token **不需要和其他 token 做 chunkwise 分块**（T=1 没有 chunk 可言），**也不需要重新处理前面 4 个 token**——它们的信息已经全在 `state` 里了。一次简单的 recurrent 更新就搞定。

这就是 KDA 适合长文本推理的根本原因：Prefill 用 chunkwise 并行高效处理长 prompt，Decode 用 state 继承历史信息、每次只算 1 个 token，全程不需要 O(T²) 的注意力计算。

---

## 2. 调用链概览

```
chunk_kda_fwd()  (kda.py:1023)
│
├── Step A: kda_gate_chunk_cumsum()          # Gate 激活 + chunk-local cumsum
│
├── Step B: chunk_kda_fwd_intra()            # 块内矩阵计算 + 块间求解
│     ├── chunk_kda_fwd_intra_token_parallel # 对角线 Aqk/Akk 块
│     ├── chunk_kda_fwd_kernel_inter_solve_fused  # 块间 forward substitution
│     └── recompute_w_u_fwd()                # w/u/kg 重计算
│
├── Step C: chunk_gated_delta_rule_fwd_h()  # Delta Rule 跨块循环
│
└── Step D: chunk_gla_fwd_o_gk()             # 最终输出
```

每个 kernel 的输出是下一个 kernel 的输入，形成一条严格的流水线。

---

## 3. Kernel 1: `kda_gate_chunk_cumsum` — Gate 激活 + Chunk-local Cumsum

### 3.1 计算了什么

**输入**: `raw_gate [B,T,H,K]`, `A_log [H]`, `dt_bias [H*K]`

**输出**: `g_cumsum [B,T,H,K]` (fp32)

**数学公式**:

```
gate[t,h,k] = -exp(A_log[h]) * softplus(raw_gate[t,h,k] + dt_bias[h,k])
g_cumsum[t] = cumsum(gate[t])  within each chunk, scaled by RCP_LN2
```

### 3.2 为什么需要这个计算

1. **Gate 激活**：`raw_gate` 是模型原始输出（未激活），`A_log` 和 `dt_bias` 是模型参数。`-exp(A_log) * softplus(x)` 保证 gate 始终为负数，使 `exp(gate) < 1`，状态只会衰减不会爆炸。

2. **Chunk-local cumsum**：后续 kernel 需要的是 **chunk 内相对位置** 的 gate 累积和，而不是全局累积。例如 chunk 内第 i 个 token 的 gate 累计是 `sum_{j=0}^{i} gate[j]`，与 chunk 外的历史无关。这种局部性避免了长序列的全局累积误差。

3. **RCP_LN2 缩放**：将 natural-log 空间转换为 log2 空间，后续 kernel 使用 `exp2()` 而非 `exp()`，在硬件上更高效。

### 3.3 核心实现思路

- **Gate 激活**：向量化计算 `-exp(A_log) * softplus(g + dt_bias)`，所有 token 并行
- **Chunk-local cumsum**：按 chunk 分组，每个 chunk 内独立做 prefix sum。使用 `log2` 空间的并行 prefix sum 算法，避免逐 token 串行
- **输出 fp32**：gate 需要高精度，后续 kernel 的 `exp2()` 操作对精度敏感

---

## 4. Kernel 2: `chunk_kda_fwd_intra_token_parallel` — 对角线 Aqk/Akk 块

### 4.1 计算了什么

**输入**: `q,k [B,T,H,K]`, `g [B,T,H,K]`, `beta [B,T,H]`

**输出**: `Aqk [B,T,H,BT]` (对角线 QK 块), `Akk [B,T,H,BC]` (对角线 KK 块)

**数学公式**（每个 sub-chunk 内，token 按 token 并行）:

```
对每个 token t (在 sub-chunk s 内):
  Aqk[t, j] = scale * (q[t] · (k[j] * exp2(g[t] - g[j])))     for j <= t in same sub-chunk
  Akk[t, j] = (k[t]*beta[t]) · (k[j] * exp2(g[t] - g[j]))     for j < t  in same sub-chunk
```

### 4.2 为什么需要这个计算

这是 chunkwise 方法的核心：**chunk 内部的 token 交互用精确矩阵乘法计算**。

- `exp2(g[t] - g[j])` 是 gate 衰减因子：当 j 在 t 之前时，gate 累积差 `g[t] - g[j]` 为正，`exp2()` > 1，放大历史影响；当 j 在 t 之后时，需要 mask 掉（因果注意力）
- `Aqk` 用于后续计算 chunk 内注意力
- `Akk` 是一个 **块三对角矩阵**，用于 forward substitution 解耦 chunk 间依赖

### 4.3 核心实现思路

- **Token-parallel**: 每个 token 一个 CTA block，grid = `(B*T, H)`。每个 block 独立计算该 token 与同一 sub-chunk 内所有 key 的交互
- **Sub-chunk 分块**: 对角线块只在同一个 sub-chunk (BC=16) 内计算，大幅减少计算量
- **Gated dot product**: 先计算 `k[j] * exp2(g[t] - g[j])`，再与 `q[t]` 做点积，将 gate 衰减融入注意力权重

---

## 5. Kernel 3: `chunk_kda_fwd_kernel_inter_solve_fused` — 块间 Forward Substitution

### 5.1 这个 Kernel 在解决什么问题（用大白话讲）

回顾一下问题：chunk 内有 64 个 token，它们之间有因果依赖关系——token 30 能看到 token 0-29，但看不到 token 31-63。

在逐 token 递归中，这天然成立（算到 token 30 时，state 已经包含了 0-29 的信息）。但在 chunkwise 并行计算中，我们同时算 64 个 token，**必须显式地建模这种依赖关系**。

这个 kernel 做的是：**计算一个 64×64 的"依赖矩阵"的逆**，这个矩阵描述了 chunk 内 token 之间的相互影响。有了这个逆矩阵，我们就可以用矩阵乘法（而不是逐 token 串行）来同时计算所有 64 个 token 的等效 key/value。

### 5.2 具体来说，Akk 矩阵是什么

Akk 是一个 64×64 的矩阵，精确描述了 chunk 内每对 token 的关系：

```
Akk[i, j] = (k[i] * beta[i]) · (k[j] * exp2(g[i] - g[j]))
```

- 当 i=j 时，对角线元素，表示 token 自己的贡献
- 当 i>j 时，下三角元素，表示 token j 对 token i 的因果影响
- 当 i<j 时，上三角为 0（因果掩码）

Akk 的实际结构是"块三对角"的——因为 chunk 被分成了 4 个 sub-chunk（每个 16 token）：

```
Akk (64×64):
┌──────────┬──────────┬──────────┬──────────┐
│ Akk_00   │    0     │    0     │    0     │  ← sub-chunk 0 (token 0-15)
│  (16×16) │          │          │          │
├──────────┼──────────┼──────────┼──────────┤
│ Akk_10   │ Akk_11   │    0     │    0     │  ← sub-chunk 1 (token 16-31)
│  (16×16) │  (16×16) │          │          │
├──────────┼──────────┼──────────┼──────────┤
│ Akk_20   │ Akk_21   │ Akk_22   │    0     │  ← sub-chunk 2 (token 32-47)
│  (16×16) │  (16×16) │  (16×16) │          │
├──────────┼──────────┼──────────┼──────────┤
│ Akk_30   │ Akk_31   │ Akk_32   │ Akk_33   │  ← sub-chunk 3 (token 48-63)
│  (16×16) │  (16×16) │  (16×16) │  (16×16) │
└──────────┴──────────┴──────────┴──────────┘
```

### 5.3 为什么需要求 Akk 的逆

回到 Step C（Delta Rule），我们需要计算 `w = Akk^{-1} @ (k * beta * exp2(gk))`。这个公式的含义是：

> 给定 chunk 内 64 个 token 的原始 key 和它们之间的依赖关系 Akk，求出 64 个"等效 key" w，使得 `w @ state` 能精确计算 state 对每个 token 的贡献，而不用逐 token 串行。

换句话说，**Akk^{-1} 把 chunk 内的依赖关系"解耦"了**。有了它，我们就可以用一次矩阵乘法代替 64 步串行计算。

### 5.4 如何高效求逆（Forward Substitution + 块三对角）

直接求一个 64×64 矩阵的逆是 O(64³) = 262K 次运算。但利用块三对角结构，可以大幅简化：

**Step 1 — 计算 off-diagonal 块**（Akk_10, Akk_20, Akk_21, Akk_30, Akk_31, Akk_32）：
```
Akk_10 = (k_1 ⊙ exp2(g_1 - g_0_norm)) @ (k_0 ⊙ exp2(g_0_norm - g_1))^T  [16×16]
Akk_20 = (k_2 ⊙ exp2(g_2 - g_0_norm)) @ (k_0 ⊙ exp2(g_0_norm - g_2))^T  [16×16]
...
```

**Step 2 — Forward Substitution**（对 4 个对角线块分别求逆）：
```
A_00^{-1} = solve_tril(Akk_00)   ← 只需求一个 16×16 下三角矩阵的逆
A_11^{-1} = solve_tril(Akk_11)
A_22^{-1} = solve_tril(Akk_22)
A_33^{-1} = solve_tril(Akk_33)
```

**Step 3 — 链式合并**（用已求出的逆计算 off-diagonal 逆）：
```
A_10^{-1} = -A_11^{-1} @ Akk_10 @ A_00^{-1}
A_20^{-1} = -A_22^{-1} @ (Akk_20 @ A_00^{-1} + Akk_21 @ A_10^{-1})
A_30^{-1} = -A_33^{-1} @ (Akk_30 @ A_00^{-1} + Akk_31 @ A_10^{-1} + Akk_32 @ A_20^{-1})
...
```

这利用了一个数学性质：**块三角矩阵的逆，可以先求对角线逆，再通过链式矩阵乘法传播到非对角线位置**。计算量从 O(64³) 降到 O(4 × 16³ + 链式乘法) ≈ O(20K)，约 13x 加速。

### 5.5 核心实现思路

- **Grid 结构**: `(NT, B*H)` — 每个 (chunk, head) 对是一个 CTA，独立计算
- **寄存器驻留**: 所有 16×16 的 off-diagonal 块保存在寄存器中（18+ 个 float32 块），避免 HBM 读写
- **Forward substitution**: 对每个 sub-chunk 的对角线块，按行依次求解，每次只加载一行
- **链式矩阵乘法**: 用 `tl.dot` 链式计算 `A_30^{-1} = -A_33^{-1} @ (...)`
- **FUSE_RECOMPUTE 路径**: 当 grid 较小时，将 w/u/kg 重计算也融合进此 kernel，减少 kernel launch 次数

---

## 6. Kernel 4: `recompute_w_u_fwd` — w/u/kg 重计算

### 6.1 计算了什么

**输入**: `k,v [B,T,H,K/V]`, `beta [B,T,H]`, `Akk_inv [B,T,H,BT]`, `gk [B,T,H,K]`

**输出**: `w [B,T,H,K]`, `u [B,T,H,V]`, `kg [B,T,H,K]`

**数学公式**（每个 chunk 独立）:

```
w[chunk] = Akk_inv @ (k[chunk] * beta[chunk] * exp2(gk[chunk]))
u[chunk] = Akk_inv @ (v[chunk] * beta[chunk])
kg[chunk] = k[chunk] * exp2(gk_last - gk[chunk])
```

### 6.2 为什么需要这个计算

这三个量是 Delta Rule (Step C) 的输入，各有不同的物理意义：

- **w**: "等效 key" — 已经用 Akk_inv 解耦了 chunk 内 token 相互影响后的 key。后续 `w @ h` 能精确计算隐藏状态对当前 chunk 的贡献。
- **u**: "等效 value" — 同理，解耦后的 value。`u - w @ h` 就是 v_new。
- **kg**: "gate 校准后的 key" — 用于更新隐藏状态 `h += kg^T @ v_new`。`exp2(gk_last - gk)` 将每个 token 的 gate 相对于 chunk 最后一个 token 对齐。

### 6.3 核心实现思路

- **分块矩阵乘法**: 每个 chunk 独立做 `Akk_inv[BC×BC] @ kb[BC×K]`，grid = `(cdiv(K,BK), cdiv(V,BV), NT, B*H)`
- **输入精度**: Akk_inv 是 fp32（来自 inter_solve），k/v 是 bf16，dot product 用 tf32 保持精度
- **kg 计算简单**: 只需逐元素 `exp2(gk_last - gk)` 和乘法，无矩阵运算

---

## 7. Kernel 5: `chunk_gated_delta_rule_fwd_h` — Delta Rule 跨块循环

### 7.1 这个 Kernel 在做什么（用大白话讲）

这是整个 KDA 算法最核心的一步。它做了一件事：**用一个"记忆矩阵" state 在 chunk 之间传递信息**。

想象你在读一本书，每读完 64 页（一个 chunk），你更新一次你的"理解状态"：

```
读完 chunk 0 后的理解状态 = state_0
读完 chunk 1 后的理解状态 = state_1 = state_0 * 遗忘 + chunk_1 的新信息
读完 chunk 2 后的理解状态 = state_2 = state_1 * 遗忘 + chunk_2 的新信息
...
```

这个 `state` 是一个 `[K, V]` 矩阵（比如 64×64），它把之前所有 chunk 的 key-value 信息压缩在里面。每一行是一个 key channel 对各 value 维度的贡献，每一列是一个 value 维度。它不是显式地存储每个 token，而是存储"key 和 value 之间的关系"——类似于一个协方差矩阵。

### 7.2 每一步具体在做什么

```
state = initial_state                     # 初始状态（上一个 batch 留下的记忆）

===== 处理 chunk 0 =====
h[0] = state                              # ① 保存当前状态（后面 Step D 输出时要读）
v_new[0] = u[0] - w[0] @ state              # ② 从当前 value 中减去历史已经能预测的部分
v_new[0] *= exp(gk_last - gk)             # ③ 补偿 chunk 内的 gate 不均匀
state *= exp2(gk_last)                    # ④ 遗忘：旧信息随时间衰减
state += kg[0]^T @ v_new[0]               # ⑤ 写入：把 chunk 0 的新信息加进 state

===== 处理 chunk 1 =====
h[1] = state                              # ① 保存状态（此时 state 已包含 chunk 0 的信息）
v_new[1] = u[1] - w[1] @ state              # ② 减去 state 能预测的，留下"新信息"
...（重复 ③④⑤）

===== 处理 chunk 2, 3, ... =====
...（重复）
```

### 7.3 为什么第②步要"减去历史预测"

这是 Delta Rule 的 Delta 所在。类比：

- 你读第 10 章时，如果第 10 章的内容你已经从前面 9 章猜到了，那第 10 章就没有"新信息"，不需要更新你的理解
- 只有你猜不到的部分，才是值得记住的"增量"

数学上：`w[0] @ state` 用当前的 state 去"预测"chunk 0 的 value。如果 state 已经包含了某个 token 的信息，预测值就会接近真实值，`v_new = u - predicted` 就很小。这样一来，**state 不会重复存储已知信息，节省了"记忆空间"**。

### 7.4 为什么第④步要"遗忘"

state 的容量是有限的（K×V 个数字）。如果不遗忘，旧信息会永远占据 state 的空间，新信息加不进去。

`exp2(gk_last)` 是一个小于 1 的衰减因子。`gk_last` 是 chunk 最后一个 token 的 gate 累积值——gate 越负，`exp2(gate)` 越小，遗忘越快。这意味着：

- 如果模型认为当前 chunk 的信息"保质期短"，就会给 gate 一个很负的值，让 state 快速遗忘
- 如果模型认为当前 chunk 的信息"很重要"，就会给 gate 一个接近 0 的值，让 state 保留更久

### 7.5 输入输出的物理意义

| 变量 | 形状 | 物理意义 |
|------|------|---------|
| `initial_state` | `[B,H,K,V]` | 上一个 batch 留下的"记忆"（推理时复用，类似 KV cache） |
| `w` | `[B,T,H,K]` | 等效 key——已用 Akk_inv 解耦了 chunk 内 token 互相影响 |
| `u` | `[B,T,H,V]` | 等效 value——同上 |
| `kg` | `[B,T,H,K]` | gate 校准后的 key——用于更新 state |
| `gk` | `[B,T,H,K]` | gate 累积值——控制遗忘速度 |
| `h`（输出） | `[B,NT,H,V,K]` | 每个 chunk 开始时的 state 快照，供 Step D 读 |
| `v_new`（输出） | `[B,T,H,V]` | 修正后的 value——只包含"新信息" |

### 7.6 核心实现思路

- **串行 chunk 循环**：chunk 之间必须串行（state 依赖前一个 chunk），但 chunk 内并行
- **h 存储布局**：`[NT, H, V, K]` — 每个 chunk 保存一份 state 快照，供 Step D 的 GLA 输出使用
- **Tensor 收缩**：`w @ state` 是 `[BT, K] @ [K, V] → [BT, V]` 的矩阵乘法。`kg^T @ v_new` 是 `[K, BT] @ [BT, V] → [K, V]` 的矩阵乘法，结果直接累加到 `[K, V]` 的 state 中
- **环境变量控制**: `SGLANG_GDN_CHUNK_H_BV`, `SGLANG_GDN_CHUNK_H_NUM_WARPS` 等控制 tile 大小和并行度

---

## 8. Kernel 6: `chunk_gla_fwd_o_gk` — 最终输出

### 8.1 这个 Kernel 在做什么（用大白话讲）

终于到了最后一步。现在我们有两样东西：

1. **一个"记忆矩阵" h[c]**：包含了 chunk c 之前所有历史信息的压缩状态（K×V 矩阵，K 行 V 列）
2. **chunk 内的精确交互 Aqk**：描述了当前 chunk 内 64 个 token 之间的直接关系（64×64 因果矩阵）

这个 kernel 把这两者结合起来，产生最终的注意力输出。

### 8.2 分两部分理解

```
对于 chunk c 中的每个 token t:
  output[t] = (跨块部分) + (块内部分)

  跨块部分 = (q[t] * exp2(g[t])) · h[c] 的每一列  * scale
           = 用 query 从"记忆矩阵"中读取所有历史信息

  块内部分 = (Aqk[t, :t+1]) · v_new[0:t+1]
           = 用因果注意力矩阵计算当前 chunk 内的精确交互
```

**类比**：

- **跨块部分** 就像你凭"印象"回答问题——你不需要翻前面的每一页，你的理解状态 h 已经压缩了所有关键信息。`q @ h` 就是用当前问题 q 从理解状态中提取相关答案。

- **块内部分** 就像你仔细读当前页面——因为这一页就在眼前，你可以精确地看每一个字。Aqk 保存了 64 个 token 之间的精确关系，不需要压缩。

两者相加，既有了全局视角（跨块），又有了局部精度（块内）。

### 8.3 直观理解 h 的作用

`h[c]` 是一个 `[K, V]` 矩阵（K 行 V 列，和 state 相同布局）。

当计算 `q[t] @ h[c]` 时：
- h[c] = [K, V]，K 行 V 列
- 每一行对应一个 key channel，值是"这个 channel 对各 value 维度的贡献"
- `q[t]` 是 K 维向量，"我想关注哪些 key channel"
- `q @ h` 是一个加权求和：用 q 的系数把 K 个"key→value 映射"线性组合，得到一个 V 维输出
- `q @ h` 的结果是一个 V 维向量，就是"根据历史信息回答的问题"

这个过程和标准 Attention 的 `softmax(q@k^T) @ v` 本质相同，但复杂度从 O(T) 降到了 O(V)，因为 h 已经把 T 个历史 token 压缩了。

### 8.4 核心实现思路

- **两部分独立计算后相加**：跨块和块内没有依赖，可以并行
- **Causal mask**: 块内注意力用三角矩阵 mask，保证 token 只看过去不看未来
- **Scale**: 跨块部分乘以 `1/sqrt(K)`，与标准 Attention 的缩放一致
- **Grid**: `(cdiv(K,BK), cdiv(V,BV), NT, B*H)`，4 维 grid 最大化并行

---

## 9. 完整流水线：`chunk_kda_fwd`

### 9.1 端到端调用

```python
def chunk_kda_fwd(q, k, v, g, beta, scale, initial_state, ...):
    # Step A: Gate 激活 + Cumsum
    g = kda_gate_chunk_cumsum(g, A_log, ...)  # 或 chunk_local_cumsum

    # Step B: 块内计算 + 块间求解
    w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(q, k, v, gk=g, beta, ...)

    # Step C: Delta Rule 跨块循环
    h, v_new = chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g, ...)

    # Step D: 最终输出
    o = chunk_gla_fwd_o_gk(q=q, v=v_new, g=g, A=Aqk, h=h, ...)

    return o
```

### 9.2 数据流图

```
输入: q,k,v,raw_gate [B,T,H,K]  beta [B,T,H]  initial_state [B,H,K,V]
       │
       ▼
  ┌─────────────────────────────────────────────────┐
  │ Step A: Gate Cumsum                             │  ← Kernel 1
  │  gate = -exp(A_log) * softplus(raw_gate + dt_bias)│
  │  g_cumsum = chunk-local cumsum(gate) * RCP_LN2  │
  │  Output: g [B,T,H,K] fp32                       │
  └────────────────────┬────────────────────────────┘
                       │ g
                       ▼
  ┌─────────────────────────────────────────────────┐
  │ Step B: Intra-chunk + Inter-solve               │  ← Kernels 2,3,4
  │  ├── token_parallel: 对角线 Aqk/Akk 块          │
  │  ├── inter_solve: 块间 forward substitution     │
  │  └── recompute_w_u: w/u/kg 重计算               │
  │  Output: w [B,T,H,K], u [B,T,H,V],              │
  │          kg [B,T,H,K], Aqk [B,T,H,BT]           │
  └────────────────────┬────────────────────────────┘
                       │ w, u, kg, Aqk
                       ▼
  ┌─────────────────────────────────────────────────┐
  │ Step C: Delta Rule H                            │  ← Kernel 5
  │  for each chunk:                                │
  │    v_new = u - w @ state      (subtract history)  │
  │    state *= exp2(gk_last)   (state decay)       │
  │    state += kg^T @ v_new    (state update)      │
  │  Output: h [NT,H,V,K], v_new [B,T,H,V]          │
  └────────────────────┬────────────────────────────┘
                       │ h, v_new
                       ▼
  ┌─────────────────────────────────────────────────┐
  │ Step D: GLA Output                              │  ← Kernel 6
  │  o = (q*exp2(g)) @ h * scale    (跨块)           │
  │    + (Aqk ⊙ causal) @ v_new      (块内)         │
  │  Output: o [B,T,H,V]                            │
  └────────────────────┬────────────────────────────┘
                       │
                       ▼
              输出: o [B,T,H,V]
```

### 9.3 两种 Gate 路径

`chunk_kda_fwd` 支持两种 gate 输入方式：

| 路径 | 条件 | 使用的 kernel | 说明 |
|------|------|-------------|------|
| A_log 路径 | `A_log is not None` | `kda_gate_chunk_cumsum` | 融合 gate+cumsum，性能更好 |
| Pre-activated 路径 | `A_log is None` | `chunk_local_cumsum` | gate 已由调用方激活，只做 cumsum |

### 9.4 两种 Intra 路径

| 路径 | 条件 | 说明 |
|------|------|------|
| `_small_grid=True` | `B*NT*H <= 256` (非 NPU) | `fuse_diagonal=True, fuse_recompute=True`，3 个 kernel 融合为 1 个 |
| `_small_grid=False` | 大 grid 或 NPU | 拆成 3 个独立 kernel (token_parallel + inter_solve + recompute_w_u) |

NPU 上强制走 `_small_grid=False`，因为融合 kernel 对 aicore 太重。

---

## 10. 关键设计决策

### 10.1 为什么用 `exp2` 而非 `exp`

硬件上 `exp2` 比 `exp` 更快（可以用位操作近似）。配合 `RCP_LN2 = 1/ln(2) ≈ 1.44` 将 natural-log 空间转换为 log2 空间：

```
exp(gate) = 2^{gate / ln(2)} = 2^{gate * RCP_LN2} = exp2(gate * RCP_LN2)
```

### 10.2 为什么 chunk_size=64, sub_chunk(BC)=16

- **chunk_size=64**：平衡了块内并行度（越大越好）和块间串行开销（越小越好）。64 是经验值，在 910B 系列 NPU 上刚好能装进共享内存
- **BC=16**：16×16 的矩阵在寄存器中刚好合适，4 个 sub-chunk 覆盖 64 token。Akk 变成 64×64 块三对角矩阵，forward substitution 高效求解

### 10.3 为什么需要 `beta` — 另一个门控

`beta` 是一个额外的标量 `[B,T,H]`，控制每个 token 被写入 state 的强度：

```
state += (v - state@k^T) ⊗ k * beta
```

- 如果 beta=0，这个 token 完全不写入 state（模型认为它不重要）
- 如果 beta=1，这个 token 正常写入
- 这类似于 LSTM 的 input gate——模型可以学习"跳过"噪音 token

### 10.4 为什么 Gate 是逐 Channel 的（K 维向量而非标量）

每个 channel 有独立的遗忘速率。这允许模型在不同的特征维度上有不同的记忆模式：

- 某些 channel 可能需要记住很久之前的信息（如"主语是谁"）
- 某些 channel 只需要短期记忆（如"当前词的词性"）

代价是 gate 的存储和计算都是 O(K)，对于 K=128 这点开销可以忽略。

### 10.5 一句话总结每个 Kernel

| Kernel | 一句话 |
|--------|--------|
| Gate Cumsum | 激活 gate 并做 chunk 内累积，控制"遗忘速度" |
| Token Parallel | 计算 chunk 内 token 两两之间的精确关系（对角线块） |
| Inter Solve | 求解依赖矩阵的逆，把 64 步串行变成一次矩阵乘法 |
| Recompute W/U | 用逆矩阵重算等效 key/value，为 Delta Rule 做准备 |
| Delta Rule H | 跨 chunk 传递"记忆矩阵"，减去可预测的，只保留新信息 |
| GLA Output | 记忆矩阵（全局）+ 精确交互（局部）→ 最终输出 |

---

## 11. 测试覆盖

`test_level2_kernel_precision.py` 为每个 kernel 提供了 10 条参数化测试用例，覆盖：

| 场景 | 配置 | 覆盖内容 |
|------|------|---------|
| 完整 chunk | T=128, H=2 | 2 个完整 chunk，正常路径 |
| 不完整 chunk | T=63, T=65, T=127 | 边界条件：最后一个 chunk 不满 |
| 单 head | H=1 | 最小并行度 |
| 奇数 head | H=3 | 非 2 的幂 |
| 极短序列 | T=1, T=2 | 单个/两个 token |
| 非 2 的幂 | T=96, T=100 | 非对齐 token 数 |

每个 kernel 的 NPU 输出与 CPU 参考实现对比 RMSE，所有 kernel 精度 < 0.0025。

---

## 12. 源码文件索引

| 文件 | 内容 |
|------|------|
| `kda.py` | `chunk_kda_fwd()` 主函数, `kda_gate_chunk_cumsum`, `recompute_w_u_fwd`, `chunk_gla_fwd_o_gk` |
| `chunk_intra.py` | `chunk_kda_fwd_intra()` 调度, `chunk_kda_fwd_kernel_inter_solve_fused` kernel |
| `chunk_intra_token_parallel.py` | `chunk_kda_fwd_intra_token_parallel` kernel |
| `chunk_delta_h.py` | `chunk_gated_delta_rule_fwd_h` kernel |
| `cumsum.py` | `chunk_local_cumsum` kernel (pre-activated gate 路径) |
| `kda_test/test_level2_kernel_precision.py` | 逐 kernel 精度验证 (61 条用例) |