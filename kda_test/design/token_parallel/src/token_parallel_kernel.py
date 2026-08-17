#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-2（Token Parallel：对角线 Aqk/Akk）独立实现：纯 torch + triton。

本目录不依赖 sglang 包，只用 ``torch`` / ``triton``。实现与上游
``python/sglang/kernels/ops/attention/fla/chunk_intra_token_parallel.py``
（``chunk_kda_fwd_kernel_intra_token_parallel``）数学完全一致:

    Aqk[i, j] = <q[i],  k[j] * exp2(g[i]-g[j])> * scale          (j <= i, 同 sub-chunk)
    Akk[i, j] = <k[i]·beta[i], k[j] * exp2(g[i]-g[j])>            (j < i,  同 sub-chunk)

模块提供:
  * ``token_parallel_ref``    —— 纯 torch CPU 参考（逐 token 循环，ground truth）
  * ``token_parallel_torch``  —— torch 元算子版本（**性能/精度基准**，按 sub-chunk
                                批量化 matmul，避免 Python 双层循环）
  * ``token_parallel_triton`` —— triton kernel 版（1 CTA / token / head）
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

def token_parallel_ref(q, k, gk, beta, scale, chunk_size=_BT, sub_chunk_size=_BC):
    """逐 token 循环 CPU 参考（数学原文）:

        Aqk[b][t][h][j % BT]  = <q[t], k[j] * exp2(g[t]-g[j])> * scale
                                 for j in [i_ts, min(t, i_ts+BC))
        Akk[b][t][h][j-i_ts]  = <k[t]·beta[t], k[j] * exp2(g[t]-g[j])>
                                 for j in [i_ts, min(t, i_ts+BC)), j < t

    支持任意 B/T/H/K。T 非 BT 倍数时, 尾部 token 只计算自身区间。
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    qf, kf, gf, bf = q.float(), k.float(), gk.float(), beta.float()
    Aqk = torch.zeros(B, T, H, BT, dtype=torch.float32)
    Akk = torch.zeros(B, T, H, BC, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                i_c, i_s = t // BT, (t % BT) // BC
                i_ts = i_c * BT + i_s * BC
                qt = qf[b, t, h]
                kt = kf[b, t, h] * bf[b, t, h]
                gt = gf[b, t, h]
                for j in range(i_ts, min(t + 1, min(T, i_ts + BC))):
                    kj = kf[b, j, h]
                    gj = gf[b, j, h]
                    kgj = kj * torch.exp2(gt - gj)
                    Aqk[b, t, h, j % BT] = float((qt * kgj).sum()) * scale
                    if j < t:
                        Akk[b, t, h, j - i_ts] = float((kt * kgj).sum())
    return Aqk, Akk


# ═══════════════════════════════════════════════════════════════════════════
# torch 元算子版本（性能基本准）
# ═══════════════════════════════════════════════════════════════════════════

def token_parallel_torch(q, k, gk, beta, scale, chunk_size=_BT, sub_chunk_size=_BC):
    """批量 torch 版本（按 sub-chunk 并行, 每个 sub-chunk 一次 BC×BC gated-dot）。

    与 ``token_parallel_ref`` 生成相同的 Aqk / Akk。使用 tensor 运算而非 Python
    逐 token 循环, 用作性能基准:

        for each sub-chunk sc:
            qc, kc, gc  = [B, NT, BC, H, K] 切片
            dec[i,j]    = exp2(gc[i] - gc[j])         [BC, BC]
            aqk[i,j]    = (qc[i]·(kc[j]*dec[i,j]))    [BC, BC]
            akk[i,j]    = (kbc[i]·(kc[j]*dec[i,j]))
            apply causal (j<=i) / strict (j<i) mask
            写回 Aqk[.., sc*BC+j] / Akk[.., j]
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    NT = _cdiv(T, BT)
    NC = BT // BC
    dev = q.device
    qf, kf, gf, bf = q.float(), k.float(), gk.float(), beta.float()
    # 尾部 pad 到 NT*BT 再 reshape(补零, 与 chunk 内部 cumsum 语义一致)
    pad = NT * BT - T
    if pad:
        z = torch.zeros(B, pad, H, K, dtype=torch.float32, device=dev)
        qf = torch.cat([qf, z], dim=1)
        kf = torch.cat([kf, z], dim=1)
        gf = torch.cat([gf, z.clone()], dim=1)
        bf = torch.cat([bf, torch.zeros(B, pad, H, dtype=torch.float32, device=dev)], dim=1)
    kbc = kf * bf[..., None]
    # 5D 容器: [B, NT, BT, H, BT/BC], 写回时在 sub-chunk 位置对齐后 reshape
    Aqk5 = torch.zeros(B, NT, BT, H, BT, device=dev, dtype=torch.float32)
    Akk5 = torch.zeros(B, NT, BT, H, BC, device=dev, dtype=torch.float32)
    tri = torch.tril(torch.ones(BC, BC, dtype=torch.bool, device=dev))       # j<=i
    eye = torch.eye(BC, dtype=torch.bool, device=dev)                        # j==i
    for sc in range(NC):
        a = sc * BC
        qr = qf.reshape(B, NT, BT, H, K)[:, :, a : a + BC]   # [B,NT,BC,H,K]
        kr = kf.reshape(B, NT, BT, H, K)[:, :, a : a + BC]
        gr = gf.reshape(B, NT, BT, H, K)[:, :, a : a + BC]
        kbr = kbc.reshape(B, NT, BT, H, K)[:, :, a : a + BC]
        dec = torch.exp2(gr[:, :, :, None, :, :] - gr[:, :, None, :, :, :])
        kw = kr[:, :, None, :, :, :] * dec          # [B,NT,i,j,H,K]
        aqk = (qr[:, :, :, None, :, :] * kw).sum(-1)  # [B,NT,i,j,H]
        akk = (kbr[:, :, :, None, :, :] * kw).sum(-1)
        # Aqk: 因果 j<=i, 乘 scale;  Akk: 严格因果 j<i (去掉对角线)
        aqk = aqk * tri[None, None, :, :, None] * scale
        akk = akk * tri[None, None, :, :, None] * (~eye)[None, None, :, :, None]
        # 5D 连续写回:
        #   aqk[i,j,h] -> Aqk[.., chunk 内位置 a+i, .., a+j]
        #   akk[i,j,h] -> Akk[.., chunk 内位置 a+i, .., j]   (列 = sub-chunk 内偏移)
        Aqk5[:, :, a : a + BC, :, a : a + BC] = aqk.permute(0, 1, 2, 4, 3)  # [B,NT,BC,H,BC]
        Akk5[:, :, a : a + BC, :, :] = akk.permute(0, 1, 2, 4, 3)
    Aqk = Aqk5.reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    Akk = Akk5.reshape(B, NT * BT, H, BC)[:, :T].contiguous()
    return Aqk, Akk


# ═══════════════════════════════════════════════════════════════════════════
# triton kernel：1 CTA / token / head（与上游 token-parallel 策略一致）
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit(do_not_specialize=["T"])
def _token_parallel_kernel(
    q, k, g, beta, Aqk, Akk,
    scale,
    T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
):
    """1 个 CTA 处理 1 个 (token, head)，遍历同 sub-chunk 内先前 token。"""
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)
    bos = (i_tg // T) * T
    i_t = i_tg % T
    if i_t >= T:
        return
    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_ts = i_c * BT + i_s * BC

    o_k = tl.arange(0, BK)
    m_k = o_k < K

    base_i = bos * H * K + i_t * H * K + i_hg * K
    qb = tl.load(q + base_i + o_k, mask=m_k, other=0.0).to(tl.float32)
    kb = tl.load(k + base_i + o_k, mask=m_k, other=0.0).to(tl.float32)
    gb = tl.load(g + base_i + o_k, mask=m_k, other=0.0).to(tl.float32)
    beta_v = tl.load(beta + bos * H + i_t * H + i_hg).to(tl.float32)
    kb = kb * beta_v  # k·beta[i]

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        base_j = bos * H * K + j * H * K + i_hg * K
        kj = tl.load(k + base_j + o_k, mask=m_k, other=0.0).to(tl.float32)
        gj = tl.load(g + base_j + o_k, mask=m_k, other=0.0).to(tl.float32)
        kgj = kj * tl.math.exp2(gb - gj)
        kgj = tl.where(m_k, kgj, 0.0)
        aqk = tl.sum(qb * kgj, axis=0) * scale
        akk = tl.sum(kb * kgj, axis=0) * tl.where(j < i_t, 1.0, 0.0)
        tl.store(
            Aqk + bos * H * BT + i_t * H * BT + i_hg * BT + (j % BT),
            aqk,
        )
        tl.store(
            Akk + bos * H * BC + i_t * H * BC + i_hg * BC + (j - i_ts),
            akk,
        )


def token_parallel_triton(
    q, k, gk, beta, scale,
    Aqk=None, Akk=None,
    chunk_size=_BT, sub_chunk_size=_BC,
):
    """triton kernel 版; 返回 (Aqk, Akk)。所有张量已在 NPU。"""
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    BK = triton.next_power_of_2(K)
    if Aqk is None:
        Aqk = torch.zeros(B, T, H, BT, device=q.device, dtype=torch.float32)
    if Akk is None:
        Akk = torch.zeros(B, T, H, BC, device=q.device, dtype=torch.float32)
    grid = (B * T, H)
    _token_parallel_kernel[grid](
        q, k, gk, beta, Aqk, Akk, float(scale),
        T, H=H, K=K, BT=BT, BC=BC, BK=BK, num_warps=1,
    )
    torch.npu.synchronize()
    return Aqk, Akk