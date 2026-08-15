#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-3（Inter-Solve Fused：非对角线 Aqk/Akk + 对角块前向替换 + 链式求逆）独立实现。

本目录不依赖 sglang 包，只用 ``torch`` / ``torch_npu`` / ``triton``。数学与上游
``python/sglang/kernels/ops/attention/fla/chunk_intra.py``
（``chunk_kda_fwd_kernel_inter_solve_fused``）一致，但做如下**独立子集化**：

  * 只做固定长度（B,T,H,K），不做 VARLEN / safe-gate / FUSE_RECOMPUTE / FUSE_DIAGONAL；
  * 输入的对角线 Akk 块已由 Kernel-2（token_parallel）写好，本 kernel 读 ``Akkd``
    直接做前向替换 + 链式求逆；
  * 输出 Akk_inv = [B, T, H, BT]（10 个 16×16 子块的合并下三角逆）。

计算内容:

    Phase 1（非对角线块, 每 CTA 一个 (chunk, head)）:
        Akk_ij = (K_i * exp2(G_i - G_i[last])) @ (K_j * exp2(G_i[last] - G_j))^T * beta_j
        Aqk_ij = (Q_i * exp2(G_i - G_i[last])) @ (K_j * exp2(G_i[last] - G_j))^T * scale
          (i, j = 0..3, i > j;  G_i[last] 是子块 i 的**末尾** token 的 g 参考点)

    Phase 2（对 4 个对角 16×16 下三角子块做逐行前向替换求逆）:
        D_inv = (I - tril(D))^{-1}   (逐行累加, fp32)

    Phase 3（链式矩阵乘合并逆）:
        Ai_10 = -Ai_11 @ Akk_10 @ Ai_00
        Ai_21 = -Ai_22 @ Akk_21 @ Ai_11
        Ai_32 = -Ai_33 @ Akk_32 @ Ai_22
        Ai_20 = -Ai_22 @ (Akk_20 @ Ai_00 + Akk_21 @ Ai_10)
        Ai_31 = -Ai_33 @ (Akk_31 @ Ai_11 + Akk_32 @ Ai_21)
        Ai_30 = -Ai_33 @ (Akk_30 @ Ai_00 + Akk_31 @ Ai_10 + Akk_32 @ Ai_20)

模块提供:
  * ``inter_solve_ref``     —— 纯 torch CPU 参考（逐 chunk 循环, ground truth）,
                                返回 (Aqk, Akk_inv)
  * ``inter_solve_torch``   —— torch 元算子版本（**精度/性能基准**）,数学与 ref 一致
  * ``inter_solve_triton``  —— triton kernel 版（1 CTA / chunk / head）
"""

import torch
import torch_npu  # noqa: F401  (必须在创建任何 npu 张量之前 import)
import triton
import triton.language as tl

_BT = 64   # chunk 大小
_BC = 16   # sub-chunk 大小


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


# ═══════════════════════════════════════════════════════════════════════════
# torch CPU 参考（ground truth）
# ═══════════════════════════════════════════════════════════════════════════

def diag_solve_forward_ref(D):
    """对一个 [n, n] 下三角矩阵做前向替换求逆（含对角线）。

    采用与上游一致的行累积算法:
        A = -strict_tril(D)
        for i in 2..n-1:
            A[i] = -D[i] + sum_k A[i,k] * A[k]     (仅 i 行, 逐行)
        D_inv = A + I

    返回 [n, n] fp32。n 任意（用于大块验算）。
    """
    n = D.shape[-1]
    A = -torch.tril(D, diagonal=-1).float()  # [n,n]
    for i in range(2, n):
        row = A[i]  # [n]
        row = row + (row[:, None] * A).sum(dim=0)  # 行乘整行累加
        A[i] = row
    return A + torch.eye(n, dtype=torch.float32, device=D.device)


def inter_solve_ref(
    q, k, g, beta, Akkd, scale,
    chunk_size=_BT, sub_chunk_size=_BC,
):
    """逐 chunk 循环的 CPU 参考。

    输入:
      q/k/g:  [B, T, H, K] fp32
      beta:   [B, T, H] fp32
      Akkd:   [B, T, H, BC] fp32  （Kernel-2 输出的对角线 Akk 块；j<i 非零，j>=i 为 0）

    输出:
      Aqk:     [B, T, H, BT]  非对角线 Aqk 子块（行=token, 列=chunk 内 j 位置）
      Akk_inv: [B, T, H, BT]  10 个 16×16 子块合并的下三角逆
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    NC = BT // BC
    NT = _cdiv(T, BT)
    qf, kf, gf = q.float(), k.float(), g.float()
    beta_f = beta.float()
    Akkd_f = Akkd.float()
    # 尾 chunk 不满 BT 时, 补 0 到完整 chunk 边界, 避免 sub-chunk 切片不等长
    pad = NT * BT - T
    if pad:
        zp = torch.zeros(B, pad, H, K, dtype=torch.float32)
        qf = torch.cat([qf, zp], dim=1)
        kf = torch.cat([kf, zp.clone()], dim=1)
        gf = torch.cat([gf, zp.clone()], dim=1)
        zA = torch.zeros(B, pad, H, BC, dtype=torch.float32)
        Akkd_f = torch.cat([Akkd_f, zA], dim=1)
        zb = torch.zeros(B, pad, H, dtype=torch.float32)
        beta_f = torch.cat([beta_f, zb], dim=1)
    Aqk = torch.zeros(B, NT * BT, H, BT, dtype=torch.float32)
    Akk_inv = torch.zeros(B, NT * BT, H, BT, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            for i_c in range(NT):
                base = i_c * BT
                # —— Phase 1: 非对角线块 ——
                off = {}
                for i in range(NC):
                    r = base + i * BC
                    q_i = qf[b, r:r + BC, h]          # [BC, K]
                    k_i = kf[b, r:r + BC, h]
                    g_i = gf[b, r:r + BC, h]
                    # 参考点 = 子块末尾 token 的 gate
                    gni = gf[b, r + BC - 1, h]  # [K]
                    gqn = torch.exp2(g_i - gni.unsqueeze(0))  # [BC,K]
                    for j in range(i):
                        c = base + j * BC
                        k_j = kf[b, c:c + BC, h]
                        g_j = gf[b, c:c + BC, h]
                        # 行乘 beta_i (子块 i 的 beta, 与上游 / triton kernel 一致)
                        beta_i = beta_f[b, r:r + BC, h]        # [BC]
                        # [K, BC] = (k_j * exp2(gni - g_j))^T
                        kgt = (k_j * torch.exp2(gni.unsqueeze(0) - g_j)).t()
                        bj = (q_i * gqn) @ kgt
                        Aqk[b, r:r + BC, h, c - base:c - base + BC] = bj * scale
                        bk = (k_i * gqn) @ kgt
                        off[(i, j)] = bk * beta_i[:, None]
                # —— Phase 2: 对角块前向替换 ——
                di = {}
                for i in range(NC):
                    r = base + i * BC
                    D = Akkd_f[b, r:r + BC, h].clone()  # [BC,BC]
                    di[i] = diag_solve_forward_ref(D)
                # —— Phase 3: 链式合并 ——
                Ai = {}
                for i in range(NC):
                    Ai[(i, i)] = di[i]
                for (i, j) in [(1, 0), (2, 1), (3, 2)]:
                    Ai[(i, j)] = -di[i] @ off[(i, j)] @ di[j]
                Ai[(2, 0)] = -di[2] @ (off[(2, 0)] @ di[0] + off[(2, 1)] @ Ai[(1, 0)])
                Ai[(3, 1)] = -di[3] @ (off[(3, 1)] @ di[1] + off[(3, 2)] @ Ai[(2, 1)])
                Ai[(3, 0)] = -di[3] @ (off[(3, 0)] @ di[0]
                                       + off[(3, 1)] @ Ai[(1, 0)]
                                       + off[(3, 2)] @ Ai[(2, 0)])
                # 写回 Akk_inv
                for (i, j) in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
                               (3, 0), (3, 1), (3, 2), (3, 3)]:
                    Akk_inv[b, base + i * BC:base + i * BC + BC, h,
                            j * BC:j * BC + BC] = Ai[(i, j)]
    Aqk = Aqk[:, :T].contiguous()
    Akk_inv = Akk_inv[:, :T].contiguous()
    return Aqk, Akk_inv

# ═══════════════════════════════════════════════════════════════════════════
# torch 元算子版本（性能/精度基准）——按 chunk 批量化, 避免 Python 逐 chunk 循环
# ═══════════════════════════════════════════════════════════════════════════

def _batch_forward_solve(Dsub):
    """对一批 [P, BC, BC] 下三角矩阵做前向替换求逆（逐行向量化）。

    A = -strict_tril(D);  for i=2..BC-1: A[:,i] += A[:,i][:,None]*A 的总和行
    """
    P = Dsub.shape[0]
    A = -torch.tril(Dsub, diagonal=-1)          # [P,BC,BC]
    for i in range(2, Dsub.shape[1]):
        # A[:, i] = -D[:, i] + sum_k A[:, i, k] * A[:, k, :]
        row = A[:, i]                            # [P,BC]
        contrib = (row[:, :, None] * A).sum(1)   # [P,BC]
        A[:, i] = row + contrib
    I = torch.eye(Dsub.shape[1], dtype=Dsub.dtype, device=Dsub.device)
    return A + I


def inter_solve_torch(
    q, k, g, beta, Akkd, scale,
    chunk_size=_BT, sub_chunk_size=_BC,
):
    """torch 元算子版本: 与 ``inter_solve_ref`` 数学一致, 但按 chunk 批量化。

    返回值与 ref 相同: (Aqk[B,T,H,BT], Akk_inv[B,T,H,BT])。

    说明: 前向替换阶段与 Kernel-3 一样是串行的 (BC 行), 批量化收益主要来自
    Phase 1 的 [BC,BC] 块聚合与 Phase 3 的链式 matmul。
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    NC = BT // BC
    NT = _cdiv(T, BT)
    dev = q.device
    pad = NT * BT - T
    qf, kf, gf = q.float(), k.float(), g.float()
    beta_f = beta.float()
    Akkd_ = Akkd.float()
    if pad:
        z = torch.zeros(B, pad, H, K, dtype=torch.float32, device=dev)
        qf = torch.cat([qf, z], dim=1); kf = torch.cat([kf, z], dim=1)
        gf = torch.cat([gf, z.clone()], dim=1)
        zk = torch.zeros(B, pad, H, BC, dtype=torch.float32, device=dev)
        Akkd_ = torch.cat([Akkd_, zk], dim=1)
        zb = torch.zeros(B, pad, H, dtype=torch.float32, device=dev)
        beta_f = torch.cat([beta_f, zb], dim=1)
    # reshape to [B, NT, BT, H, K] 按 chunk 排列, 再取子块
    Q = qf.reshape(B, NT, BT, H, K)
    Ks = kf.reshape(B, NT, BT, H, K)
    G = gf.reshape(B, NT, BT, H, K)
    Bt = beta_f.reshape(B, NT, BT, H)
    D = Akkd_.reshape(B, NT, BT, H, BC)
    qi = [Q[:, :, i * BC:(i + 1) * BC, :, :] for i in range(NC)]
    ki = [Ks[:, :, i * BC:(i + 1) * BC, :, :] for i in range(NC)]
    gi = [G[:, :, i * BC:(i + 1) * BC, :, :] for i in range(NC)]
    bi = [Bt[:, :, i * BC:(i + 1) * BC, :] for i in range(NC)]
    # 参考点: 子块 i 的末尾 token 的 gate（在 chunk 内的位置 i*BC+BC-1）
    gni = [G[:, :, i * BC + BC - 1, :, :].unsqueeze(2) for i in range(NC)]  # [B,NT,1,H,K]
    # 非对角线块
    Aqk = torch.zeros(B, NT * BT, H, BT, dtype=torch.float32, device=dev)
    Akk_inv = torch.zeros(B, NT * BT, H, BT, dtype=torch.float32, device=dev)
    off = {}
    for i in range(1, NC):
        gq = torch.exp2(gi[i] - gni[i])                     # [B,NT,BC,H,K]
        for j in range(i):
            # bk_t = k_j * exp2(gni[i] - g_j)  [B,NT,BC,H,K]
            bk_t = (ki[j] * torch.exp2(gni[i] - gi[j]))
            # Aqk_ij = (q_i * gq) · bk_t^T  消 K 维, 保留两个不同的 BC 轴
            #   qi[i]*gq: [B,NT,BC_i,H,K];  bk_t: [B,NT,BC_j,H,K]
            #   einsum: 'bnihk,bnjhk->bnihj'  (i=rows, j=cols, 不可合并 d)
            aqk_m = torch.einsum('bnihk,bnjhk->bnihj', qi[i] * gq, bk_t) * scale
            akk_m = torch.einsum('bnihk,bnjhk->bnihj', ki[i] * gq, bk_t)
            # 乘 beta_i (行广播, 子块 i 的 beta, 与上游 triton kernel 一致)
            beta_i = bi[i]                          # [B,NT,BC,H]
            akk_m = akk_m * beta_i.unsqueeze(-1)
            # 写回 Aqk[b, nt*BT + i*BC + p, h, j*BC + q]
            a_i, a_j = i * BC, j * BC
            # aqk_m / akk_m: [B,NT,BC_i,H,BC_j] -> [B,NT,BC_i,H,BC_j]
            Aqk.view(B, NT, BT, H, BT)[:, :, a_i:a_i + BC, :, a_j:a_j + BC] = aqk_m
            # off[(i,j)] 统一存成 [B,NT,H,BC_i,BC_j] 以便后续链式 matmul
            off[(i, j)] = akk_m.permute(0, 1, 3, 2, 4).contiguous()  # [B,NT,H,BC,BC]
    # 对角线前向替换
    di = {}
    for i in range(NC):
        Di = D[:, :, i * BC:(i + 1) * BC, :, :]  # [B,NT,BC,H,BC]
        Di = Di.permute(0, 1, 3, 2, 4).reshape(B * NT * H, BC, BC)  # [P,BC,BC]
        di[i] = _batch_forward_solve(Di).reshape(B, NT, H, BC, BC)
    # 链式合并
    Ai = {}
    Ai[(0, 0)] = di[0]
    Ai[(1, 1)] = di[1]
    Ai[(2, 2)] = di[2]
    Ai[(3, 3)] = di[3]
    Ai[(1, 0)] = -di[1] @ off[(1, 0)] @ di[0]
    Ai[(2, 1)] = -di[2] @ off[(2, 1)] @ di[1]
    Ai[(3, 2)] = -di[3] @ off[(3, 2)] @ di[2]
    Ai[(2, 0)] = -di[2] @ (off[(2, 0)] @ di[0] + off[(2, 1)] @ Ai[(1, 0)])
    Ai[(3, 1)] = -di[3] @ (off[(3, 1)] @ di[1] + off[(3, 2)] @ Ai[(2, 1)])
    Ai[(3, 0)] = -di[3] @ (off[(3, 0)] @ di[0] + off[(3, 1)] @ Ai[(1, 0)]
                           + off[(3, 2)] @ Ai[(2, 0)])
    # 写回: Ai[(i,j)] 形状 [B,NT,H,BC,BC] -> 需要 [B,NT,BC,H,BC] 放到 [B,NT,BT,H,BT] 切片
    for (i, j) in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
                   (3, 0), (3, 1), (3, 2), (3, 3)]:
        a_i, a_j = i * BC, j * BC
        # Ai[(i,j)]: [B,NT,H,BC,BC] -> [B,NT,BC,H,BC]
        blk = Ai[(i, j)].permute(0, 1, 3, 2, 4)  # [B,NT,BC,H,BC]
        Akk_inv.view(B, NT, BT, H, BT)[:, :, a_i:a_i + BC, :, a_j:a_j + BC] = blk
    return Aqk[:, :T].contiguous(), Akk_inv[:, :T].contiguous()


# ═══════════════════════════════════════════════════════════════════════════
# triton kernel：1 CTA / chunk / head（与上游 inter_solve_fused 策略一致）
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit(do_not_specialize=["T"])
def _inter_solve_kernel(
    q, k, g, beta, Akkd, Aqk, Akk_out,
    scale,
    T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
):
    """1 个 CTA 处理 1 个 (chunk, head)。

    Phase 1: 计算 6 个非对角线 Aqk/Akk 子块 (i>j);
    Phase 2: 4 个对角子块前向替换求逆;
    Phase 3: 链式矩阵乘合并下三角逆, 写回 Akk_out [B,T,H,BT]。
    """
    i_tc, i_hg = tl.program_id(0), tl.program_id(1)
    i_b = i_hg // H
    i_h = i_hg % H
    bos = i_b * T
    if i_tc * BT >= T:
        return

    i_tc0 = i_tc * BT
    i_tc1 = i_tc0 + BC
    i_tc2 = i_tc0 + 2 * BC
    i_tc3 = i_tc0 + 3 * BC

    # 指针偏移到本 (batch, head)
    q   += (bos * H + i_h) * K
    k   += (bos * H + i_h) * K
    g   += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk_out += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC
    beta += bos * H + i_h

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    # ── Phase 1: 寄存器初始化 12 个 [BC,BC] fp32 块 ──
    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    # ── Phase 1: K 维循环累加非对角块 ──
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        # 子块 0
        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        # 子块 1
        p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
        b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
        b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_gqn = tl.where(m_tc1[:, None], tl.math.exp2(b_g1 - b_gn1[None, :]), 0.0)
        b_kgt = tl.trans(b_k0 * tl.math.exp2(b_gn1[None, :] - b_g0)).to(tl.bfloat16)
        b_qg1 = (b_q1 * b_gqn).to(tl.bfloat16)
        b_kg1 = (b_k1 * b_gqn).to(tl.bfloat16)
        b_Aqk10 += tl.dot(b_qg1, b_kgt)
        b_Akk10 += tl.dot(b_kg1, b_kgt)

        # 子块 2
        p_q2 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
        p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
        p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
        b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
        b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
        b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_gqn2 = tl.where(m_tc2[:, None], tl.math.exp2(b_g2 - b_gn2[None, :]), 0.0)
        b_qg2 = (b_q2 * b_gqn2).to(tl.bfloat16)
        b_kg2 = (b_k2 * b_gqn2).to(tl.bfloat16)
        # (2, 0)
        b_kgt = tl.trans(b_k0 * tl.math.exp2(b_gn2[None, :] - b_g0)).to(tl.bfloat16)
        b_Aqk20 += tl.dot(b_qg2, b_kgt)
        b_Akk20 += tl.dot(b_kg2, b_kgt)
        # (2, 1)
        b_kgt = tl.trans(b_k1 * tl.math.exp2(b_gn2[None, :] - b_g1)).to(tl.bfloat16)
        b_Aqk21 += tl.dot(b_qg2, b_kgt)
        b_Akk21 += tl.dot(b_kg2, b_kgt)

        # 子块 3
        p_q3 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
        p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
        p_g3 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
        b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
        b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
        b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
        b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_gqn3 = tl.where(m_tc3[:, None], tl.math.exp2(b_g3 - b_gn3[None, :]), 0.0)
        b_qg3 = (b_q3 * b_gqn3).to(tl.bfloat16)
        b_kg3 = (b_k3 * b_gqn3).to(tl.bfloat16)
        # (3, 0)
        b_kgt = tl.trans(b_k0 * tl.math.exp2(b_gn3[None, :] - b_g0)).to(tl.bfloat16)
        b_Aqk30 += tl.dot(b_qg3, b_kgt)
        b_Akk30 += tl.dot(b_kg3, b_kgt)
        # (3, 1)
        b_kgt = tl.trans(b_k1 * tl.math.exp2(b_gn3[None, :] - b_g1)).to(tl.bfloat16)
        b_Aqk31 += tl.dot(b_qg3, b_kgt)
        b_Akk31 += tl.dot(b_kg3, b_kgt)
        # (3, 2)
        b_kgt = tl.trans(b_k2 * tl.math.exp2(b_gn3[None, :] - b_g2)).to(tl.bfloat16)
        b_Aqk32 += tl.dot(b_qg3, b_kgt)
        b_Akk32 += tl.dot(b_kg3, b_kgt)

    # ── Phase 1 尾部: 存 Aqk 非对角块 (带 scale), Akk 乘 beta ──
    # 条件已移除: boundary_check 对越界位置返回 0, 乘 beta=0 也为 0, 安全
    p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    p_b1 = tl.make_block_ptr(beta, (T,), (H,), (i_tc1,), (BC,), (0,))
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_Akk10 = b_Akk10 * b_b1[:, None]

    p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    p_b2 = tl.make_block_ptr(beta, (T,), (H,), (i_tc2,), (BC,), (0,))
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_Akk20 = b_Akk20 * b_b2[:, None]
    b_Akk21 = b_Akk21 * b_b2[:, None]

    p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    p_b3 = tl.make_block_ptr(beta, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
    b_Akk30 = b_Akk30 * b_b3[:, None]
    b_Akk31 = b_Akk31 * b_b3[:, None]
    b_Akk32 = b_Akk32 * b_b3[:, None]

    # ── Phase 2: 加载对角块 + 前向替换求逆 ──
    # 对角块来自 Akkd (Kernel-2 已写好, 严格下三角, 需要前向替换求逆)
    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    # 前向替换: A = -strict_tril(D); for i=2..BC-1: A[i] += A[i] · A 行累加; A += I
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Ai00 = -tl.where(m_A, b_Ai00, 0.0)
    b_Ai11 = -tl.where(m_A, b_Ai11, 0.0)
    b_Ai22 = -tl.where(m_A, b_Ai22, 0.0)
    b_Ai33 = -tl.where(m_A, b_Ai33, 0.0)

    # 逐行前向替换 (i 从 2 到 BC-1); 尾 chunk 用 min(BC, T - i_tc0) 截断
    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(BC + 2, min(2 * BC, T - i_tc0)):
        b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
        b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
    for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
        b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
        b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
    for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
        b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
        b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    # ── Phase 3: 链式矩阵乘合并逆 ──
    # 第 1 层: Ai_10, Ai_21, Ai_32
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision="ieee"),
        b_Ai00,
        input_precision="ieee",
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision="ieee"),
        b_Ai11,
        input_precision="ieee",
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision="ieee"),
        b_Ai22,
        input_precision="ieee",
    )
    # 第 2 层: Ai_20, Ai_31
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision="ieee")
        + tl.dot(b_Akk21, b_Ai10, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision="ieee")
        + tl.dot(b_Akk32, b_Ai21, input_precision="ieee"),
        input_precision="ieee",
    )
    # 第 3 层: Ai_30
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision="ieee")
        + tl.dot(b_Akk31, b_Ai10, input_precision="ieee")
        + tl.dot(b_Akk32, b_Ai20, input_precision="ieee"),
        input_precision="ieee",
    )

    # ── 写回 Akk_out: 10 个子块 ──
    p_Akk00 = tl.make_block_ptr(Akk_out, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk_out, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk_out, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_Akk20 = tl.make_block_ptr(Akk_out, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk21 = tl.make_block_ptr(Akk_out, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk_out, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
    p_Akk30 = tl.make_block_ptr(Akk_out, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Akk31 = tl.make_block_ptr(Akk_out, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Akk32 = tl.make_block_ptr(Akk_out, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk_out, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk_out.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk_out.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk_out.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk_out.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk_out.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk_out.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk_out.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk_out.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk_out.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk_out.dtype.element_ty), boundary_check=(0, 1))


def inter_solve_triton(
    q, k, g, beta, Akkd, scale,
    Aqk=None, Akk_out=None,
    chunk_size=_BT, sub_chunk_size=_BC,
):
    """triton kernel 版; 返回 (Aqk, Akk_inv)。所有张量已在 NPU。"""
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    NT = _cdiv(T, BT)
    BK = triton.next_power_of_2(K)
    if Aqk is None:
        Aqk = torch.zeros(B, T, H, BT, device=q.device, dtype=q.dtype)
    if Akk_out is None:
        Akk_out = torch.zeros(B, T, H, BT, device=q.device, dtype=q.dtype)
    grid = (NT, B * H)
    _inter_solve_kernel[grid](
        q, k, g, beta, Akkd, Aqk, Akk_out, float(scale),
        T, H=H, K=K, BT=BT, BC=BC, BK=BK, num_warps=1,
    )
    torch.npu.synchronize()
    return Aqk, Akk_out
