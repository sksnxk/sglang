# Kernel 3 设计文档: 独立 Inter-Solve Fused 算子

> 本文档是 `kda_test/design/Kernel3_InterSolve.md` 的**独立子集重构**。
> 与原始文档的差异:
> - 只保留 `B,T,H,K` 固定长度的最小闭环, 不涉及 VARLEN / safe-gate /
>   FUSE_RECOMPUTE / FUSE_DIAGONAL 等扩展路径;
> - 对角线 Akk 块由 Kernel-2（token_parallel）写好, 本 kernel 读 `Akkd` 直接做
>   前向替换 + 链式求逆;
> - 行号改为引用本目录的 `src/inter_solve_kernel.py`;
> - 增加「精度测试策略」和「性能测试思路」两节, 配套本目录测试驱动。

---

## 1. 输入输出定义

### 1.1 输入张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `q` | `[B, T, H, K]` | fp32 | Query 张量 |
| `k` | `[B, T, H, K]` | fp32 | Key 张量 |
| `g` (gate) | `[B, T, H, K]` | fp32 | Kernel-1 的 chunk 局部 gate cumsum 输出（log2 空间） |
| `beta` | `[B, T, H]` | fp32 | Per-token per-head 的标量权重 |
| `Akkd` | `[B, T, H, BC]` | fp32 | Kernel-2 输出的对角线 Akk 块（严格下三角, j<i 同 sub-chunk 内 gated dot） |
| `scale` | 标量 | fp32 | Attention scale，通常为 `K^{-0.5}` |

编译期常量：`H`、`K`、`BT`（chunk 大小=64）、`BC`（sub-chunk 大小=16）、
`BK`（= `next_power_of_2(K)`）。

### 1.2 输出张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `Aqk` | `[B, T, H, BT]` | fp32 | 非对角线 Aqk 块, 列 = j 在 chunk 内位置 (`j % BT`) |
| `Akk_inv` | `[B, T, H, BT]` | fp32 | 合并的下三角逆（10 个 16×16 子块）, 上三角为 0 |

## 2. 分核并行策略（Grid 拓扑）

```
Grid = (NT, B * H)
        ~~~   ~~~~~
         |      |
       chunk数 所有 (batch, head) 对
```

- `NT = cdiv(T, BT)`：时间维 chunk 总数；
- `B * H`：每个 (batch, head) 组合一个平面。

每个 CTA 处理一个 `(时间 chunk i_t, (b,h))`，1 个 warp（num_warps=1），
在该 chunk 内完成 Phase 1/2/3 三阶段计算。

### 程序 ID 映射

```
i_t  = program_id(0)   chunk 索引 (0 .. NT-1)
i_hg = program_id(1)   (batch, head) 联合索引 (0 .. B*H-1)
i_b  = i_hg // H       batch 索引
i_h  = i_hg %  H       head 索引
bos  = i_b * T          batch 内起始 token 偏移
```

## 3. 计算思路

### 3.1 块三对角矩阵结构

每个 chunk（BT=64 token）内的 Akk 矩阵是一个 64×64 的下三角矩阵（因果掩码）。
它被划分为 4×4 个 16×16 的子块（BC=16）：

```
Akk (64×64) = 4×4 子块, 每个子块 16×16

          col0    col1    col2    col3
         (0-15)  (16-31) (32-47) (48-63)
    ┌─────────────────────────────────────┐
r0  │  D0      │        │        │        │   对角块: Kernel-2 写入 Akkd
    │ [16×16]  │   0    │   0    │   0    │
    ├──────────┼────────┼────────┼────────┤
r1  │  Akk_10  │  D1    │        │        │   非对角块: 本 kernel 计算
    │ [16×16]  │[16×16] │   0    │   0    │
    ├──────────┼────────┼────────┼────────┤
r2  │  Akk_20  │ Akk_21 │  D2    │        │
    │ [16×16]  │[16×16] │[16×16] │   0    │
    ├──────────┼────────┼────────┼────────┤
r3  │  Akk_30  │ Akk_31 │ Akk_32 │  D3    │
    │ [16×16]  │[16×16] │[16×16] │[16×16] │
    └─────────────────────────────────────┘

图例:
  D0..D3   — 对角线子块 (16×16 严格下三角, 来自 Akkd)
  Akk_ij   — 非对角线下三角子块 (16×16 全矩阵, 本 kernel 计算)
  0         — 零矩阵 (因果掩码, 上三角区域)
```

### 3.2 Phase 1: 非对角线块

每个子块公式：

```
Akk_ij = (K_i * exp2(G_i - G_i[last])) @ (K_j * exp2(G_i[last] - G_j))^T * beta_j
Aqk_ij = (Q_i * exp2(G_i - G_i[last])) @ (K_j * exp2(G_i[last] - G_j))^T * scale

其中:
  - i, j 为子块索引 (0-3), i > j
  - G_i[last]: 子块 i 的末尾 token 的 gate (上游用 i_tc_i 的第一行 g)
```

Phase 1 在一个 K 维度的循环中完成（BK = next_power_of_2(K)）：

```
for i_k in range(cdiv(K, BK)):
    load k0, g0, q1, k1, g1, q2, k2, g2, q3, k3, g3
    b_gn1 = g[i_tc1 的第一行, :]  (子块1的起始 = 子块0的末尾)
    b_gqn = exp2(g1 - gn1)
    b_kgt = (k0 * exp2(gn1 - g0))^T

    Aqk_10 += (q1 * gqn) @ b_kgt
    Akk_10 += (k1 * gqn) @ b_kgt
    ...（共 6 对）
```

K 循环结束后：
- Aqk 块写入全局内存（带 scale）；
- Akk 块乘以 beta 留在寄存器。

```
Akk_ij = Akk_ij * beta_j[:, None]    (行广播 beta_j)
Aqk_ij 存储到全局内存 (带 scale)
```

### 3.3 Phase 2: 对角线块的前向替换

从 Akkd 加载 4 个对角 16×16 严格下三角块，做逐行前向替换求逆：

```
算法: 前向替换求下三角矩阵的逆

输入:  D = strict_tril(Akkd)  — 16×16 严格下三角
输出:  D_inv — (I - D) 的逆

第 1 步: 初始化
  A = -strict_tril(D)         # 取负号, 严格下三角部分
  for i in 2..15:              # 从第2行开始
    a_row = -D[i]
    a_row += sum(a_row[:, None] * A, axis=0)   # 矩阵行乘累加
    A[i, :] = a_row            # 更新第 i 行

第 2 步: 加回单位矩阵
  D_inv = A + I
```

4 个对角块 (D0, D1, D2, D3) 各自独立做前向替换，得到 Ai_00, Ai_11, Ai_22, Ai_33。

### 3.4 Phase 3: 链式矩阵乘法合并逆

已知 4 个对角块的逆 `Ai_00, Ai_11, Ai_22, Ai_33` 和 6 个非对角线 Akk 块，
通过链式矩阵乘法计算合并后的下三角逆矩阵：

```
公式推导 (块三对角矩阵求逆):

对于 4×4 块下三角矩阵 A:
  第 0 行: A00 @ Ai_00 = I          → Ai_00 = A00^{-1}
  第 1 行: A10@Ai_00 + A11@Ai_10 = 0 → Ai_10 = -Ai_11 @ A10 @ Ai_00
  第 2 行: A20@Ai_00 + A21@Ai_10 + A22@Ai_20 = 0
           → Ai_20 = -Ai_22 @ (A20@Ai_00 + A21@Ai_10)
  第 3 行: 同理...
```

计算顺序（按依赖关系）:

```
第1层: Ai_10 = -Ai_11 @ Akk_10 @ Ai_00
       Ai_21 = -Ai_22 @ Akk_21 @ Ai_11
       Ai_32 = -Ai_33 @ Akk_32 @ Ai_22

第2层: Ai_20 = -Ai_22 @ (Akk_20@Ai_00 + Akk_21@Ai_10)
       Ai_31 = -Ai_33 @ (Akk_31@Ai_11 + Akk_32@Ai_21)

第3层: Ai_30 = -Ai_33 @ (Akk_30@Ai_00 + Akk_31@Ai_10 + Akk_32@Ai_20)
```

寄存器占用: 10 个 [16,16] float32 块 (10 × 256 × 4 = 10KB)
  Ai_00, Ai_11, Ai_22, Ai_33  (4 个对角)
  Ai_10, Ai_20, Ai_21, Ai_30, Ai_31, Ai_32 (6 个非对角)

### 3.5 尾 chunk 处理

- T 不是 BT 倍数时，最后一个 chunk 的某些子块可能越界；
- 用 `m_tc1 = (i_tc1 + o_i) < T` 等掩码处理：越界位置不计算、不存储；
- 上游用 `if i_tc1 < T:` 等条件包裹子块 1/2/3 的计算，本目录实现一致。

## 4. 关键代码对应（`src/inter_solve_kernel.py`）

| 设计逻辑 | 本目录实现 |
|---------|-----------|
| Grid 定义: (NT, B*H) | `inter_solve_triton` 中 `grid = (NT, B * H)` |
| 全局 token 索引解析 | `_inter_solve_kernel`: `i_b = i_hg // H; i_h = i_hg % H; bos = i_b * T` |
| 寄存器初始化 12 个 [BC,BC] | `b_Aqk10..b_Akk32 = tl.zeros([BC, BC], tl.float32)` |
| K 维循环加载子块 | `for i_k in range(tl.cdiv(K, BK)): tl.make_block_ptr + tl.load` |
| 非对角块 dot 累加 | `b_Aqk10 += tl.dot(b_qg1, b_kgt)` 等 |
| Aqk 存 global (带 scale) | `tl.store(p_Aqk10, (b_Aqk10 * scale).to(...))` |
| Akk 乘 beta 留寄存器 | `b_Akk10 = b_Akk10 * b_b1[:, None]` |
| 对角块从 Akkd 加载 | `b_Ai00 = tl.load(p_Akk00, boundary_check=(0,1))` |
| 前向替换逐行累加 | `for i in range(2, BC): b_a += tl.sum(b_a[:,None] * b_Ai, 0)` |
| 链式乘法第1层 | `b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_Akk10), b_Ai00)` |
| 链式乘法第2层 | `b_Ai20 = -tl.dot(b_Ai22, tl.dot(b_Akk20, b_Ai00) + tl.dot(b_Akk21, b_Ai10))` |
| 链式乘法第3层 | `b_Ai30 = -tl.dot(b_Ai33, ...)` |
| 写回 10 个子块 | `tl.store(p_Akk00..p_Akk33, b_Ai00..b_Ai33)` |

## 5. torch 元算子实现（`inter_solve_torch`，性能/精度基准）

- 按 chunk 批量化：对每 chunk 的 4 个 sub-chunk 位置 `a`，提取
  `qi/ki/gi/bi = [B, NT, BC, H, K]` 切片；
- 参考点 `gni[i] = G[:,:,i*BC+BC-1,:,:].unsqueeze(2)`  —— 5D 张量切片；
- 非对角块用 `torch.einsum('bndhk,bndjk->bndhj', qi[i]*gq, bk_t)` 批量计算；
- 乘 beta_j 行广播后写回 `off[(i,j)]`；
- 对角线前向替换用 `_batch_forward_solve` 批量化（[P, BC, BC] 逐行）；
- 链式合并用 `torch.matmul` (`@`) 批量化。

它和 CPU 参考在 fp32 上逐元素一致（实测 max-diff ~1e-5），作为性能基准时
是整个 kda 算子图（einsum + matmul + 前向替换循环 + 链式合并的中间张量往返）。

## 6. 精度 & 性能对比测试策略（配套 `run.py` / `testcases.csv`）

- 每个 case 固定 seed，输入分布：`q/k = randn`，`g = randn*0.5`（log2 累积
  gate 量级较小），`beta = randn*0.1 + 1.0`（恒正、在对角线附近抖动）；
  `Akkd` 由内联 token_parallel 数学计算（逐 token 循环, 严格下三角）；
- 对每个 case 依次跑三个实现：
  1. **torch_npu 元算子** `inter_solve_torch` —— 精度基本准 + 性能基本准；
  2. **triton kernel** `inter_solve_triton` —— 被测对象；
  3. **CPU 参考** `inter_solve_ref`（逐 chunk 循环，ground truth）。
- 精度指标（两个都满足才 PASS）：
  - `max|triton - torch_npu| < 1e-2`（Aqk / Akk_inv 分开算）；
  - `max|torch_npu - ref| < 1e-2` 且 `max|triton - ref| < 1e-2`。
- 性能指标：预热 `--warmup`(默认 5) 次、各跑 `--repeats`(默认 30) 次，用
  `torch.npu.synchronize()` 包裹计时，输出每个 case 的
  **加速比 = torch_npu_time / triton_time**。
- CSV 中共 15 个 case，覆盖：完整 chunk / 尾 chunk 不满（`T=63/65/96/100/127/193/2562`）、
  单/多 head（`H=1/2/3`）、单/多 batch（`B=1/2`）、`K=32/64/128`。

## 7. 性能测试思路

- torch_npu 元算子：einsum + matmul + 前向替换串行循环 + 链式合并的算子图，
  多次 kernel 启动与中间张量读写，是性能基准的下界参考；
- triton kernel：每 chunk/head 一个 CTA，单 kernel 完成 Phase 1/2/3，
  无中间张量；K 维循环内复用同 chunk 的 k/g，寄存器内完成 dot 积；
- 每 case 预热 5 次、计时 30 次（`torch.npu.synchronize()` 包裹），报告
  `torch_ms / triton_ms / speedup`。

加速比主要来自: 单 kernel 复用与寄存器密集型计算（Phase 1 12 块 + Phase 3 10 块
[16,16] fp32 寄存器），以及避免中间张量（Aqk/Akk 块）的全局内存往返。

## 附录 A: 关键常量速查

| 常量 | 值 | 用途 |
|------|-----|------|
| `BT` | 64 | Chunk 大小 / 时间维 tile 大小 |
| `BC` | 16 | Sub-chunk 大小（NC = BT/BC = 4） |
| `BK` | `next_power_of_2(K)` | K 维 block/pad |
| `num_warps` | 1 | 每 CTA warp 数（与真实 kernel 一致） |

## 附录 B: 与真实 kernel 的差异对照

| 项 | `python/.../fla/chunk_intra.py` | 本目录独立实现 |
|----|---------------------------------------------------|----------------|
| VARLEN / `cu_seqlens` | 支持 | 只做固定长度 `bos = i_b * T` |
| `FUSE_DIAGONAL` | 对角块内联计算 | 不支持（Akkd 来自 Kernel-2） |
| `FUSE_RECOMPUTE` | 直接计算 w/u/kg | 不支持（只输出 Akk_inv） |
| `USE_SAFE_GATE` | 对角块预求逆时跳过 Phase 2 | 不支持（始终做前向替换） |
| `tl.autotune` (BK/num_warps) | 有 | 无（固定 BK=next_pow2(K), num_warps=1） |
| 无 NPU 环境 | 无法运行 | `inter_solve_ref` 退化为纯 CPU 参考 |
| 依赖 | sglang 包 | 仅 torch + torch_npu + triton |
