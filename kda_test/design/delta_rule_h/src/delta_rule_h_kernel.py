#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-5（Delta Rule H）独立实现：纯 torch + torch_npu + triton。

本模块是 ``python/sglang/kernels/ops/attention/fla/chunk_delta_h.py`` 中
``chunk_gated_delta_rule_fwd_h`` 的功能等价物，但只依赖 ``torch`` /
``torch_npu`` / ``triton``，**不 import 任何 sglang 代码**，因此可被独立
验证目录使用。

只保留**固定长度 + USE_GK + USE_EXP2 + INPLACE_UPDATE + SAVE_NEW_VALUE +
USE_INITIAL_STATE** 路径（K==V），删除上游的 VARLEN / USE_G / USE_EXP2=False
等分支。这与本项目（Kimi-Linear Delta Attention）实际调用路径一致：上游
``chunk_kda()`` 调用本 kernel 时 ``cu_seqlens=None``、``use_exp2=True``、
``initial_state`` 非空、``save_new_value=True``。

计算内容（与上游 kernel 完全一致）:

    State h ∈ [V, K] 跨 chunk 递推:
        for c in 0..NT-1:
            h_snapshot[c] = h                       # 保存快照到 h[B, NT, H, V, K]
            v_new = u - w @ h^T                     # Delta Rule: 残差 = 原值 - 历史预测
            v_new_save[c] = v_new                   # 保存到 v_new[B, T, H, V]
            h = h * exp2(gk_last)                   # per-channel 衰减（log2 空间）
            h += k^T @ v_new                        # 外积累加

    Epilogue: 把最终 state 写回 initial_state（in-place）。

Triton kernel 为 tiled 向量化实现:

    * grid = ``(cdiv(V, BV), N * H)``，BV = 32（env ``SGLANG_GDN_CHUNK_H_BV``）；
    * 每个 program (CTA) 处理一个 ``((batch, head), V-tile)``，加载 BT=64 个
      token 的 kg/w/u 切片，K 维按 64 分 4 个 tile 展开（K≤256）；
    * 状态寄存器 ``b_h1..b_h4`` 形状 ``[BV, 64]`` fp32；K=64 时只用 b_h1，
      K=128 用 b_h1/b_h2，依此类推；
    * 由于 triton-ascend 编译器对 ``block_ptr`` store 到 ``(V, K)`` 形状的
      源寄存器有损坏 bug，快照与最终写回都用 flat 1D store（reshape 成
      ``(BV*64,)`` 后 ``tl.store`` 到连续地址），与上游一致。
"""

import os

import torch
import torch_npu  # noqa: F401  (必须在创建任何 npu 张量之前 import)
import triton
import triton.language as tl


_BT = 64                              # chunk 大小
_BV = int(os.getenv("SGLANG_GDN_CHUNK_H_BV", "32"))    # V 维 tile 大小
_NUM_WARPS = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_WARPS", "4"))
_NUM_STAGES = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_STAGES", "2"))


def _cdiv(a: int, b: int) -> int:
    """向上取整的整数除法。"""
    return -(a // -b)


# ═══════════════════════════════════════════════════════════════════════════
# torch CPU 参考（ground truth）—— 逐 chunk 递推，整体 [V, K] 计算
# ═══════════════════════════════════════════════════════════════════════════

def delta_rule_h_ref(
    k, w, u, gk, initial_state, initial_state_indices,
    chunk_size=_BT,
):
    """纯 torch CPU 参考实现（可在任意 device 上运行，典型为 CPU）。

    与上游 ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` 数学一致:
      * 不做 BV 分块（整体 [V, K] 计算），简化逻辑；
      * USE_GK + USE_EXP2 + INPLACE_UPDATE + SAVE_NEW_VALUE + USE_INITIAL_STATE；
      * K == V 约束（上游 kernel 同样要求 K==V）。

    参数:
        k (kg):  [B, T, H, K] fp32  衰减后的 key（k * beta * exp2(gk_last - gk)）
        w:       [B, T, H, K] fp32  衰减后的 w
        u:       [B, T, H, V] fp32  原始 value (= Aqk @ (v * beta))
        gk:      [B, T, H, K] fp32  per-channel gate（log2 空间，已 cumsum + scale）
        initial_state: [N, H, V, K] fp32  初始状态（in-place 更新为最终状态）
        initial_state_indices: [B] int32  每个 batch 条目指向 initial_state 的索引

    返回:
        h:     [B, NT, H, V, K] fp32  每 chunk 起始状态快照
        v_new: [B, T, H, V] fp32       Delta Rule 残差 value
        initial_state: [N, H, V, K]    （已被 in-place 更新为最终状态）
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    assert K == V, f"delta_rule_h_ref requires K==V, got K={K}, V={V}"
    BT = int(chunk_size)
    NT = _cdiv(T, BT)

    kf = k.float()
    wf = w.float()
    uf = u.float()
    gkf = gk.float()
    state0 = initial_state.float().clone()
    indices = initial_state_indices.to(torch.int64).cpu()

    h = torch.zeros(B, NT, H, V, K, dtype=torch.float32)
    v_new = torch.zeros(B, T, H, V, dtype=torch.float32)

    for b in range(B):
        idx = int(indices[b].item())
        for h_idx in range(H):
            state = state0[idx, h_idx].clone()   # [V, K]
            for c in range(NT):
                tc = c * BT
                tc_end = min(T, tc + BT)
                BT_act = tc_end - tc
                # ① 保存快照（chunk 起始状态）
                h[b, c, h_idx] = state
                # ② Delta Rule: v_new = u - w @ state^T
                w_chunk = wf[b, tc:tc_end, h_idx]          # [BT_act, K]
                u_chunk = uf[b, tc:tc_end, h_idx]          # [BT_act, V]
                k_chunk = kf[b, tc:tc_end, h_idx]          # [BT_act, K]
                v_c = u_chunk - w_chunk @ state.T          # [BT_act, V]
                v_new[b, tc:tc_end, h_idx] = v_c
                # ③ per-channel gate 衰减: state *= exp2(gk_last)
                last = tc_end - 1
                gk_last = gkf[b, last, h_idx]              # [K]
                state = state * torch.exp2(gk_last[None, :])   # [V, K]
                # ④ 外积累加: state += k^T @ v_c  (=[K, BT] @ [BT, V] -> [K, V], 再转置)
                state = state + v_c.T @ k_chunk            # [V, K] += [V, BT] @ [BT, K]
            # 写回最终 state
            state0[idx, h_idx] = state

    # in-place 更新 initial_state
    initial_state.copy_(state0.to(initial_state.dtype))
    return h, v_new


# ═══════════════════════════════════════════════════════════════════════════
# torch_npu 元算子版本（精度/性能基准）—— 逐 chunk 串行，每 chunk 内用 matmul
# ═══════════════════════════════════════════════════════════════════════════

def delta_rule_h_torch(
    k, w, u, gk, initial_state, initial_state_indices,
    chunk_size=_BT,
):
    """torch_npu 元算子版本: 与 ``delta_rule_h_ref`` 数学一致, 在 NPU 上运行。

    chunk 间有依赖（state 跨 chunk 传递），无法完全批量化；每个 chunk 内用
    ``torch.matmul`` / ``torch.exp2`` / 广播乘法（在 NPU 上各为一个 kernel）。
    作为精度基准（与 ref 一致）与性能基准（多 kernel 拼接 vs triton 单 kernel）。

    参数与 ``delta_rule_h_ref`` 相同，所有张量须在 NPU 上。
    返回 (h, v_new)；initial_state 被 in-place 更新。
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    assert K == V, f"delta_rule_h_torch requires K==V, got K={K}, V={V}"
    BT = int(chunk_size)
    NT = _cdiv(T, BT)

    kf = k.to(torch.float32)
    wf = w.to(torch.float32)
    uf = u.to(torch.float32)
    gkf = gk.to(torch.float32)
    dev = k.device
    state0 = initial_state.to(torch.float32).clone()
    indices = initial_state_indices.to(torch.int64)

    h = torch.zeros(B, NT, H, V, K, dtype=torch.float32, device=dev)
    v_new = torch.zeros(B, T, H, V, dtype=torch.float32, device=dev)

    for b in range(B):
        idx = int(indices[b].item())
        for h_idx in range(H):
            state = state0[idx, h_idx].clone()   # [V, K]
            for c in range(NT):
                tc = c * BT
                tc_end = min(T, tc + BT)
                # ① 快照
                h[b, c, h_idx] = state
                # ② Delta Rule
                w_chunk = wf[b, tc:tc_end, h_idx]
                u_chunk = uf[b, tc:tc_end, h_idx]
                k_chunk = kf[b, tc:tc_end, h_idx]
                v_c = u_chunk - w_chunk @ state.T
                v_new[b, tc:tc_end, h_idx] = v_c
                # ③ per-channel 衰减
                last = tc_end - 1
                gk_last = gkf[b, last, h_idx]              # [K]
                state = state * torch.exp2(gk_last[None, :])
                # ④ 外积更新
                state = state + v_c.T @ k_chunk
            state0[idx, h_idx] = state

    initial_state.copy_(state0.to(initial_state.dtype))
    return h, v_new


# ═══════════════════════════════════════════════════════════════════════════
# triton kernel：固定长度 + USE_GK + USE_EXP2 子集（与上游一致）
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _exp2(x):
    """log2 空间的 exp2; 上游用 tl.math.exp2（非 fast_expf 路径）。"""
    return tl.math.exp2(x)


@triton.jit(do_not_specialize=["T"])
def _delta_rule_h_kernel(
    k, v, w, v_new, gk, h, initial_state, initial_state_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    """Delta Rule H triton kernel（固定长度 + USE_GK + USE_EXP2 子集）。

    Grid = (cdiv(V, BV), N * H)；每个 CTA 处理 ((batch, head), V-tile i_v)。
    递推: snapshot -> Delta Rule -> per-channel decay -> 外积更新 -> 写回。
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    # 固定长度: 每 batch 长度相同 T
    bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    # [BV, 64] state 寄存器（K 维按 64 分 tile 展开，覆盖 K≤256）
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # 偏移到本 (batch, head)
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    index = tl.load(initial_state_indices + i_n).to(tl.int32)
    h0 = initial_state + index * stride_h
    ht = initial_state + index * stride_h
    # USE_INITIAL_STATE=True: 加载 h0
    h0 = h0 + i_h * V * K
    # INPLACE_UPDATE=True: 写回 ht
    ht = ht + i_h * V * K

    # 加载初始状态（分 4 个 K-tile）
    p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
    b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
    if K > 64:
        p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
        b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
    if K > 128:
        p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
        b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
    if K > 192:
        p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
        b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # 主循环: 逐 chunk 递推
    for i_t in range(NT):
        # ① 保存快照: 用 flat 1D store 规避 triton-ascend block_ptr store bug
        b_h1_store = b_h1 + tl.zeros([BV, 64], dtype=tl.float32)
        b_h1_flat = tl.reshape(b_h1_store, (BV * 64,))
        p_h1 = h + i_t * stride_h + i_v * BV * K + tl.arange(0, BV * 64)
        tl.store(p_h1, b_h1_flat.to(h.dtype.element_ty))
        if K > 64:
            b_h2_store = b_h2 + tl.zeros([BV, 64], dtype=tl.float32)
            b_h2_flat = tl.reshape(b_h2_store, (BV * 64,))
            p_h2 = h + i_t * stride_h + i_v * BV * K + 64 + tl.arange(0, BV * 64)
            tl.store(p_h2, b_h2_flat.to(h.dtype.element_ty))
        if K > 128:
            b_h3_store = b_h3 + tl.zeros([BV, 64], dtype=tl.float32)
            b_h3_flat = tl.reshape(b_h3_store, (BV * 64,))
            p_h3 = h + i_t * stride_h + i_v * BV * K + 128 + tl.arange(0, BV * 64)
            tl.store(p_h3, b_h3_flat.to(h.dtype.element_ty))
        if K > 192:
            b_h4_store = b_h4 + tl.zeros([BV, 64], dtype=tl.float32)
            b_h4_flat = tl.reshape(b_h4_store, (BV * 64,))
            p_h4 = h + i_t * stride_h + i_v * BV * K + 192 + tl.arange(0, BV * 64)
            tl.store(p_h4, b_h4_flat.to(h.dtype.element_ty))

        # ② Delta Rule: b_v = u - w @ h^T  (分 K-tile 累加)
        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        # 保存 v_new (SAVE_NEW_VALUE=True)
        p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        # ③ per-channel gate 衰减 (USE_GK + USE_EXP2)
        last_idx = min((i_t + 1) * BT, T) - 1
        o_k1 = tl.arange(0, 64)
        b_gk_last1 = tl.load(
            gk + (bos + last_idx) * H * K + i_h * K + o_k1,
            mask=(o_k1 < K), other=0.0,
        )
        b_h1 *= _exp2(b_gk_last1)[None, :]
        if K > 64:
            o_k2 = 64 + o_k1
            b_gk_last2 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                mask=(o_k2 < K), other=0.0,
            )
            b_h2 *= _exp2(b_gk_last2)[None, :]
        if K > 128:
            o_k3 = 128 + o_k1
            b_gk_last3 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                mask=(o_k3 < K), other=0.0,
            )
            b_h3 *= _exp2(b_gk_last3)[None, :]
        if K > 192:
            o_k4 = 192 + o_k1
            b_gk_last4 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                mask=(o_k4 < K), other=0.0,
            )
            b_h4 *= _exp2(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        # ④ 外积更新: b_h += tl.trans(tl.dot(k, b_v))
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    # Epilogue: 写回最终 state (INPLACE_UPDATE=True, flat 1D store)
    b_h1_flat = tl.reshape(b_h1, (BV * 64,))
    p_ht = ht + i_v * BV * K + tl.arange(0, BV * 64)
    tl.store(p_ht, b_h1_flat.to(ht.dtype.element_ty))
    if K > 64:
        b_h2_flat = tl.reshape(b_h2, (BV * 64,))
        p_ht = ht + i_v * BV * K + 64 + tl.arange(0, BV * 64)
        tl.store(p_ht, b_h2_flat.to(ht.dtype.element_ty))
    if K > 128:
        b_h3_flat = tl.reshape(b_h3, (BV * 64,))
        p_ht = ht + i_v * BV * K + 128 + tl.arange(0, BV * 64)
        tl.store(p_ht, b_h3_flat.to(ht.dtype.element_ty))
    if K > 192:
        b_h4_flat = tl.reshape(b_h4, (BV * 64,))
        p_ht = ht + i_v * BV * K + 192 + tl.arange(0, BV * 64)
        tl.store(p_ht, b_h4_flat.to(ht.dtype.element_ty))


def delta_rule_h_triton(
    k, w, u, gk, initial_state, initial_state_indices,
    chunk_size=_BT, BV=None, num_warps=None, num_stages=None,
):
    """triton kernel 版: 在 NPU 上运行，返回 (h, v_new)，initial_state 被 in-place 更新。

    参数:
        k (kg):  [B, T, H, K]  衰减后的 key (kg = k * beta * exp2(gk_last - gk))
        w:       [B, T, H, K]  衰减后的 w
        u:       [B, T, H, V]  原始 value (= Aqk @ (v * beta))
        gk:      [B, T, H, K]  per-channel gate (log2 空间, 已 cumsum + scale)
        initial_state: [N, H, V, K]  初始状态（in-place 更新为最终状态）
        initial_state_indices: [B] int32  每个 batch 指向 initial_state 的索引
        chunk_size: chunk 大小（默认 64，与上游一致）
        BV: V 维 tile 大小（默认 env SGLANG_GDN_CHUNK_H_BV=32）
        num_warps: 每 CTA warp 数（默认 env SGLANG_GDN_CHUNK_H_NUM_WARPS=4）
        num_stages: pipeline stage 数（默认 env SGLANG_GDN_CHUNK_H_NUM_STAGES=2）

    返回:
        h:     [B, NT, H, V, K]  每 chunk 起始状态快照
        v_new: [B, T, H, V]      Delta Rule 残差 value
        initial_state: [N, H, V, K]  （已被 in-place 更新）

    纯 CPU / 无 NPU 环境下不可用（请用 ``delta_rule_h_ref``）。
    """
    B, T, Hg, K = k.shape
    V = u.shape[-1]
    H = u.shape[-2]
    assert K == V, f"delta_rule_h_triton requires K==V, got K={K}, V={V}"
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    BT = int(chunk_size)
    NT = _cdiv(T, BT)
    if BV is None:
        BV = _BV
    if num_warps is None:
        num_warps = _NUM_WARPS
    if num_stages is None:
        num_stages = _NUM_STAGES

    h = k.new_empty(B, NT, H, V, K)
    v_new = torch.empty_like(u)

    grid = (_cdiv(V, BV), B * H)
    _delta_rule_h_kernel[grid](
        k, u, w, v_new, gk, h, initial_state, initial_state_indices,
        T,
        H=H, Hg=Hg, K=K, V=V, BT=BT, BV=BV,
        num_warps=num_warps, num_stages=num_stages,
    )
    torch.npu.synchronize()
    return h, v_new
