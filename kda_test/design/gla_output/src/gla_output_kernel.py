# SPDX-License-Identifier: Apache-2.0
"""KDA ``chunk_gla_fwd_o_gk`` 算子（Kernel 6 / GLA Output）的独立实现。

本模块是 ``python/sglang/kernels/ops/attention/fla/kda.py`` 中
``chunk_gla_fwd_o_gk`` + ``chunk_gla_fwd_kernel_o`` 的功能等价物，但只依赖
``torch`` / ``triton``，**不 import 任何 sglang 代码**，因此可被本目录的
测试驱动独立使用。

计算内容（与上游 kernel 完全一致）::

    o[t] = o_cross[t] + o_intra[t]

  * 跨块 (cross):
        q_gated[t, k] = q[t, k] * scale * exp2(g[t, k])
        o_cross[t, v] = sum_k q_gated[t, k] * h[chunk(t), v, k]
                      = (q_gated @ h^T)[t, v]
  * 块内 (intra):
        A_masked = where(lower_triangular, Aqk, 0)
        o_intra[t, v] = sum_j A_masked[t, j] * v_new[j, v]
                      = (A_masked @ v_new)[t, v]

Triton kernel 为 tiled 分块矩阵乘实现：

    * grid = ``(cdiv(V, BV), NT, B * H)``，BK=32、BV=32、BT=chunk_size；
    * 每个 program (CTA) 处理一个 ``(V-tile, chunk, (batch, head))`` 交集，
      加载 ``[BT, BV]`` 的输出 tile；
    * K 维度 sequential loop：每次加载 ``[BT, BK]`` 的 q/g tile 和
      ``[BV, BK]`` 的 h tile，做 ``tl.dot(q_gated, h^T)`` 累加到 ``b_o``；
    * 块内部分加载 ``[BT, BT]`` 的 Aqk tile（施加下三角 mask）和
      ``[BT, BV]`` 的 v_new tile，做 ``tl.dot(A_masked, v_new)`` 累加到 ``b_o``。

尾部 (partial) chunk 处理
-------------------------
最后一个 chunk 的行数可能不足 BT，V 维最后一个 tile 也可能不足 BV。
kernel 统一用 ``boundary_check`` 处理：越界元素 load 为 0，``tl.where(m_s, ...)``
施加因果 mask 后，越界行的 ``b_A`` 也被清零，因此不会污染累加器。

环境说明
---------
本模块在顶部 ``import torch_npu``（不加会触发 "Background device ... is not
available" 错误）。真实运行环境由调用方在启动 python 前自行 source CANN 的
``set_env.sh`` 并设置 ``LD_LIBRARY_PATH``、
``TORCH_DEVICE_BACKEND_AUTOLOAD=0``。在无 NPU 设备的环境里也可以正常 import、
并运行 ``gla_output_ref``；``gla_output_kernel`` 在检测不到 NPU 时会自动退化为
参考实现（便于纯 CPU 校验逻辑）。
"""

import torch
import torch_npu  # noqa: F401  (必须在创建任何 npu 张量之前 import)

import triton
import triton.language as tl

# log2(e) = 1 / ln(2)，取 fp32 下的精确值（与 flash-linear-attention 一致）。
RCP_LN2 = 1.4426950216293335

# 编译期 tile 大小。BT 即 chunk_size；BK 为 K 维 tile，BV 为 V 维 tile。
_DEFAULT_BT = 64
_DEFAULT_BK = 32
_DEFAULT_BV = 32


def _cdiv(a: int, b: int) -> int:
    """向上取整的整数除法。"""
    return -(a // -b)


# ---------------------------------------------------------------------------
# CPU 参考实现（ground truth，可在任意 device 上运行）
# ---------------------------------------------------------------------------


def gla_output_ref(
    q,
    v_new,
    g,
    Aqk,
    h,
    scale,
    chunk_size=_DEFAULT_BT,
):
    """纯 torch CPU 参考实现（可在任意 device 上运行，典型为 CPU）。

    计算与 kernel 完全一致::

        o_cross[t] = (q[t] * exp2(g[t]) * scale) @ h[chunk(t)]^T
        o_intra[t] = (Aqk[t] * tril) @ v_new[t]
        o[t]        = o_cross[t] + o_intra[t]

    支持任意 B/T/H/K/V（包括尾 chunk 不满 BT、K/V 不是 tile 倍数），便于对
    triton kernel 结果做精确对比。

    参数:
        q:     [B, T, H, K]     query 向量（bf16/fp16/fp32 均可，内部转 fp32）
        v_new: [B, T, H, V]    修正后的 value（Kernel 5 输出）
        g:     [B, T, H, K]    累积 gate（Kernel 1 输出，log2 空间）
        Aqk:   [B, T, H, BT]   chunk 内因果注意力权重
        h:     [B, NT, H, V, K] 压缩状态快照（Kernel 5 输出）
        scale: float           注意力缩放因子 1/sqrt(K)
        chunk_size: int        chunk 大小（应与上游一致，默认 64）

    返回:
        [B, T, H, V] fp32 结果。
    """
    assert q.dim() == 4, f"q must be 4D [B,T,H,K], got shape {tuple(q.shape)}"
    assert v_new.dim() == 4, f"v_new must be 4D, got shape {tuple(v_new.shape)}"
    assert h.dim() == 5, f"h must be 5D [B,NT,H,V,K], got shape {tuple(h.shape)}"

    B, T, H, K = q.shape
    V = v_new.shape[-1]
    BT = int(chunk_size)
    NT = _cdiv(T, BT)

    qf = q.float()
    vf = v_new.float()
    gf = g.float()
    Af = Aqk.float()
    hf = h.float()

    o = torch.zeros(B, T, H, V, dtype=torch.float32)

    for b in range(B):
        for c in range(NT):
            tc = c * BT
            tc_end = min(T, tc + BT)
            BT_act = tc_end - tc
            for h_idx in range(H):
                q_chunk = qf[b, tc:tc_end, h_idx]         # [BT_act, K]
                g_chunk = gf[b, tc:tc_end, h_idx]         # [BT_act, K]
                v_chunk = vf[b, tc:tc_end, h_idx]         # [BT_act, V]
                A_chunk = Af[b, tc:tc_end, h_idx, :BT_act]  # [BT_act, BT_act]
                h_s = hf[b, c, h_idx]                     # [V, K]

                qg = q_chunk * torch.exp2(g_chunk) * scale  # [BT_act, K]
                o_cross = qg @ h_s.T                        # [BT_act, V]

                causal_mask = torch.tril(
                    torch.ones(BT_act, BT_act, dtype=torch.float32)
                )
                o_intra = (A_chunk * causal_mask) @ v_chunk  # [BT_act, V]

                o[b, tc:tc_end, h_idx] = o_cross + o_intra
    return o.contiguous()


# ---------------------------------------------------------------------------
# torch_npu 元算子版本（性能基准）
# ---------------------------------------------------------------------------


def gla_output_torch(
    q,
    v_new,
    g,
    Aqk,
    h,
    scale,
    chunk_size=_DEFAULT_BT,
):
    """元算子 (torch_npu 算子图) 版本：与 triton kernel 数学完全一致。

    这是**性能基准**实现 —— 用 torch_npu 现成的逐算子组合完成同样的计算,
    供与 triton kernel 做加速比对比。相比 ``gla_output_ref`` 的逐 chunk
    循环，本实现把 chunk/head 维度全部向量化，用批量 ``matmul`` 一次完成:

      * ``q * exp2(g) * scale`` 逐元素；
      * ``o_cross = matmul(q_gated, h.transpose(-1,-2))``  批量 [BT,K]@[K,V]；
      * ``A_masked = A * tril``，再 ``matmul(A_masked, v_new)``  批量 [BT,BT]@[BT,V]；
      * 两者相加，reshape 回 [B, T, H, V]。

    参数与 ``gla_output_kernel`` 相同。返回 [B, T, H, V] fp32 (device 与输入一致)。
    """
    assert q.dim() == 4, f"q must be 4D [B,T,H,K], got shape {tuple(q.shape)}"
    assert h.dim() == 5, f"h must be 5D [B,NT,H,V,K], got shape {tuple(h.shape)}"

    B, T, H, K = q.shape
    V = v_new.shape[-1]
    BT = int(chunk_size)
    NT = _cdiv(T, BT)

    # 尾部补零至 NT*BT，保证所有 chunk 都是 BT 行（零行不污染结果）。
    pad = NT * BT - T
    if pad:
        q = torch.cat(
            [q, torch.zeros(B, pad, H, K, dtype=q.dtype, device=q.device)], dim=1
        )
        v_new = torch.cat(
            [v_new, torch.zeros(B, pad, H, V, dtype=v_new.dtype, device=v_new.device)],
            dim=1,
        )
        g = torch.cat(
            [g, torch.zeros(B, pad, H, K, dtype=g.dtype, device=g.device)], dim=1
        )
        Aqk = torch.cat(
            [
                Aqk,
                torch.zeros(B, pad, H, BT, dtype=Aqk.dtype, device=Aqk.device),
            ],
            dim=1,
        )

    # ── reshape 到 [B, NT, BT, H, ...] ──
    q_r = q.reshape(B, NT, BT, H, K).float()
    v_r = v_new.reshape(B, NT, BT, H, V).float()
    g_r = g.reshape(B, NT, BT, H, K).float()
    A_r = Aqk.reshape(B, NT, BT, H, BT).float()
    h_r = h.float()  # [B, NT, H, V, K]

    # ── 跨块: o_cross = (q * exp2(g) * scale) @ h^T ──
    qg = q_r * torch.exp2(g_r) * scale  # [B, NT, BT, H, K]
    h_t = h_r.transpose(-1, -2)         # [B, NT, H, K, V]
    # 把 H 维挪到 matmul 的 batch 维: [B, NT, H, BT, K] @ [B, NT, H, K, V]
    qg_p = qg.permute(0, 1, 3, 2, 4)    # [B, NT, H, BT, K]
    o_cross = torch.matmul(qg_p, h_t)   # [B, NT, H, BT, V]
    o_cross = o_cross.permute(0, 1, 3, 2, 4)  # [B, NT, BT, H, V]

    # ── 块内: o_intra = (A * tril) @ v_new ──
    mask = torch.tril(
        torch.ones(BT, BT, dtype=torch.float32, device=q.device)
    )  # [BT, BT]
    # A_r: [B, NT, BT, H, BT] — mask 是 [BT, BT]，需放到最后一维 (BT)
    A_masked = A_r * mask.view(1, 1, BT, 1, BT)
    A_p = A_masked.permute(0, 1, 3, 2, 4)  # [B, NT, H, BT, BT]
    v_p = v_r.permute(0, 1, 3, 2, 4)       # [B, NT, H, BT, V]
    o_intra = torch.matmul(A_p, v_p)       # [B, NT, H, BT, V]
    o_intra = o_intra.permute(0, 1, 3, 2, 4)  # [B, NT, BT, H, V]

    o = (o_cross + o_intra).reshape(B, NT * BT, H, V)
    o = o[:, :T].contiguous()
    return o.float().contiguous()


# ---------------------------------------------------------------------------
# Triton kernel（被测对象）
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32]
        for BV in [32]
        for num_warps in [1]
        for num_stages in [1]
    ],
    key=["BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """KDA GLA Output triton kernel（固定长度模式，无 VARLEN）。

    grid = (cdiv(V, BV), NT, B * H)，每个 program 处理一个
    ``(V-tile, chunk, (batch, head))`` 交集，输出 ``[BT, BV]`` 子块:
      * K 维 sequential loop: 加载 [BT,BK] q/g + [BV,BK] h，
        ``b_o += dot(q_gated, h^T)``  (跨块路径);
      * 块内: 加载 [BT,BT] Aqk (施加下三角 mask) + [BT,BV] v_new，
        ``b_o += dot(A_masked, v_new)``  (块内路径);
      * fp32 累加器写出 (cast 回存储精度)。
    """
    i_v = tl.program_id(0)   # V 维 tile 索引
    i_t = tl.program_id(1)   # chunk 索引
    i_bh = tl.program_id(2)  # 联合 (batch, head) 索引
    i_b = i_bh // H
    i_h = i_bh % H

    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t    # 全局 tile 索引（用于访问 h）
    bos = i_b * T
    eos = bos + T

    # 下三角因果 mask: m_s[i, j] = (i >= j)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    # ── 跨块路径: K 维 sequential loop ──
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * V * K,
            (V, K), (K, 1),
            (i_v * BV, i_k * BK), (BV, BK), (1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * tl.math.exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BK] @ [BK, BV] = [BT, BV]  (h 按 [V,K] 布局，需转置)
        b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))

    # ── 块内路径: Aqk @ v_new (causal) ──
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V), (H * V, 1),
        (i_t * BT, i_v * BV), (BT, BV), (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V), (H * V, 1),
        (i_t * BT, i_v * BV), (BT, BV), (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT), (H * BT, 1),
        (i_t * BT, 0), (BT, BT), (1, 0),
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def gla_output_kernel(
    q,
    v_new,
    g,
    Aqk,
    h,
    scale,
    chunk_size=_DEFAULT_BT,
    out_dtype=None,
):
    """KDA GLA Output 的 triton 版本（在 NPU 上运行）。

    参数与上游 ``chunk_gla_fwd_o_gk`` 兼容（去掉 VARLEN/chunk_indices 路径）:

        q:     [B, T, H, K]    query 向量（bf16/fp16/fp32）
        v_new: [B, T, H, V]   修正后的 value（Kernel 5 输出）
        g:     [B, T, H, K]   累积 gate（Kernel 1 输出，log2 空间）
        Aqk:   [B, T, H, BT]  chunk 内因果注意力权重
        h:     [B, NT, H, V, K] 压缩状态快照（Kernel 5 输出）
        scale: float          注意力缩放因子 1/sqrt(K)

    返回:
        [B, T, H, V] 结果，dtype 与 q 一致（bf16/fp16），位于 NPU 上。
        NPU 不可用时自动退化为 CPU 参考（fp32）。
    """
    assert q.dim() == 4, f"q must be 4D [B,T,H,K], got shape {tuple(q.shape)}"
    assert h.dim() == 5, f"h must be 5D [B,NT,H,V,K], got shape {tuple(h.shape)}"

    # 纯 CPU / 无 NPU 环境：退化为参考实现
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return gla_output_ref(q, v_new, g, Aqk, h, scale, chunk_size=chunk_size)

    B, T, H, K = q.shape
    V = v_new.shape[-1]
    BT = int(chunk_size)
    NT = _cdiv(T, BT)

    # h 的 batch 维需展平为 [B*NT, H, V, K] 以匹配 kernel 的索引方式
    # (kernel 用 i_tg = i_b * NT + i_t 作为第一维索引)
    h_flat = h.reshape(B * NT, H, V, K).contiguous()

    # 输出 tensor（与 q 同 dtype，默认 bf16）
    if out_dtype is None:
        out_dtype = q.dtype
    o = torch.empty(B, T, H, V, dtype=out_dtype, device=q.device)

    BK = _DEFAULT_BK
    BV = _DEFAULT_BV
    grid = (_cdiv(V, BV), NT, B * H)

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v_new,
        g=g,
        h=h_flat,
        o=o,
        A=Aqk,
        scale=float(scale),
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    torch.npu.synchronize()
    return o
