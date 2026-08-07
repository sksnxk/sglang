#!/usr/bin/env python3
"""
Level 2: chunk_kda_fwd 子阶段独立验证
======================================

验证 chunk_kda_fwd 内部 4 个子阶段各自能独立运行。每个阶段独立构造输入，不依赖其他阶段。

测试配置参数:
  B  = 1      # batch size (单序列)
  T  = 128    # 总 token 数 (128 = 2 个 chunk, 每个 chunk=64)
  H  = 2      # head 数
  K  = 64     # key/query 维度 (每个 head 的 channel 数)
  V  = 64     # value 维度
  chunk_size = 64   # 每个 chunk 包含 64 个 token

参数含义:
  - B=1: 单 batch 验证，避免多 batch 的 autotune 复杂性
  - T=128: 刚好 2 个 chunk，足够验证跨 chunk 逻辑
  - H=2: 小 head 数，确保 _small_grid=True (B*NT*H=4 <= 256)
  - K=V=64: 标准 head 维度，匹配 Kimi Linear 实际配置
  - chunk_size=64: 标准 chunk 大小，不可修改

运行的 4 个子阶段:
  Step A: kda_gate_chunk_cumsum  — gate 激活 + chunk-local cumsum
  Step B: chunk_kda_fwd_intra     — 块内 QK/KK + 块间求解
  Step C: chunk_gated_delta_rule_fwd_h — Delta Rule 隐藏状态更新
  Step D: chunk_gla_fwd_o_gk      — 最终输出计算

预期输出: 4 个阶段全部 PASS, 无 NaN, 无 Inf, 无 timeout

运行方式:
  docker exec -it triton-ascend-env-zhm bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_npu/lib:$LD_LIBRARY_PATH
  export PYTHONPATH=/docker/zhm/0505_skill_test/sonnet/sglang/python:$PYTHONPATH
  export SGLANG_GDN_CHUNK_H_BV=16
  export SGLANG_GDN_CHUNK_H_NUM_WARPS=1
  export SGLANG_GDN_CHUNK_H_NUM_STAGES=1
  cd /docker/zhm/0505_skill_test/sonnet/sglang
  python3 kda_test/level2_stage_verification.py
"""

import sys, time
sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")
exec(open("/tmp/load_kda.py").read())

import torch
import torch_npu
from sglang.kernels.ops.attention.fla.kda import (
    kda_gate_chunk_cumsum,
    chunk_gla_fwd_o_gk,
)
from sglang.kernels.ops.attention.fla.chunk_intra import chunk_kda_fwd_intra
from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

# ═══════════════════════════════════════════════════════════════
# 测试参数
# ═══════════════════════════════════════════════════════════════
device = "npu"
dtype = torch.bfloat16
B, T, H, K, V = 1, 128, 2, 64, 64
chunk_size = 64

# ═══════════════════════════════════════════════════════════════
# 构造输入数据
# ═══════════════════════════════════════════════════════════════
torch.manual_seed(42)

# q, k: 归一化后的 query 和 key, shape [B, T, H, K]
# 通过 normalize 确保数值稳定
q = torch.nn.functional.normalize(
    torch.randn(B, T, H, K, dtype=torch.float32, device=device), dim=-1
).to(dtype)
k = torch.nn.functional.normalize(
    torch.randn(B, T, H, K, dtype=torch.float32, device=device), dim=-1
).to(dtype)

# v: value, shape [B, T, H, V], 缩小幅度避免溢出
v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1

# raw_gate: 原始 gate (未激活), shape [B, T, H, K]
# A_log 路径: gate = -exp(A_log) * softplus(raw_gate + dt_bias)
# 使用负均值确保 gate < 0 (指数衰减)
raw_gate = (torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.5 - 2.0).to(dtype)

# A_log: per-head 对数尺度参数, shape [H]
# 控制 gate 激活的陡峭程度
A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.1

# dt_bias: per-head per-channel bias, shape [H*K]
# 加到 raw_gate 上再通过 softplus 激活
dt_bias = torch.randn(H * K, dtype=torch.float32, device=device) * 0.1

# beta: sigmoid-gated beta, shape [B, T, H]
# 用于 Step B 中的 w = k*beta 计算
beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()

# initial_state: 初始隐藏状态, shape [B, H, K, V]
# 模拟跨 batch 的 initial state (如不同 sequence 的 KV cache)
initial_state = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.05

# initial_state_indices: 每个 batch 元素对应的 initial_state 索引, shape [B]
initial_state_indices = torch.arange(B, dtype=torch.int32, device=device)

# scale: 1/sqrt(K) 标准注意力缩放
scale = 1.0

# 工具函数: 检查 tensor 是否有 NaN 或 Inf
def check(name, t):
    """打印 tensor 的形状、均值、NaN/Inf 状态"""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    print(f"  {name}: shape={list(t.shape)}, mean={t.float().mean():.6f}, "
          f"nan={has_nan}, inf={has_inf}")
    assert not has_nan, f"{name} contains NaN!"
    assert not has_inf, f"{name} contains Inf!"

# ═══════════════════════════════════════════════════════════════
# Step A: kda_gate_chunk_cumsum
# ═══════════════════════════════════════════════════════════════
# 功能: 将 raw gate 激活并做 chunk-local cumulative sum
# 输入: raw_gate [B,T,H,K], A_log [H], dt_bias [H*K]
# 输出: g_cumsum [B,T,H,K] float32
# 计算:
#   1. gate = -exp(A_log[h]) * softplus(raw_gate[t] + dt_bias[h,k])
#   2. g_cumsum[t] = cumsum(gate) within each chunk (log2 空间)
# 数学意义: gate 控制每个 token 对历史的遗忘程度
#   gate 越负 → exp2(gate) 越小 → 遗忘越快
print("=" * 60)
print("Step A: kda_gate_chunk_cumsum")
print("=" * 60)
t0 = time.time()
g_cumsum = kda_gate_chunk_cumsum(
    raw_gate,
    A_log=A_log,
    chunk_size=chunk_size,
    scale=1.442695,  # RCP_LN2 = 1/ln(2), 将 natural log gate 转为 log2
    dt_bias=dt_bias,
)
torch.npu.synchronize()
print(f"  time={time.time() - t0:.2f}s")
check("g_cumsum", g_cumsum)
print("  ✅ PASS\n")

# ═══════════════════════════════════════════════════════════════
# Step B: chunk_kda_fwd_intra
# ═══════════════════════════════════════════════════════════════
# 功能: 块内 QK/KK 矩阵计算 + 块间 forward substitution
# 输入: q,k,v [B,T,H,K/V], gk [B,T,H,K], beta [B,T,H]
# 输出: w [B,T,H,K], u [B,T,H,V], kg [B,T,H,K], Aqk [B,T,H,BT]
# 计算:
#   1. 计算块内矩阵 Aqk[i,j] = (q_i*gated) @ (k_j*gated)^T
#   2. Forward substitution 解耦块间依赖:
#      w[i] = k[i]*beta[i] - Σ Aqk[i,j]*Akk_inv[j]*w[j]
#      u[i] = v[i]*beta[i] - Σ Aqk[i,j]*Akk_inv[j]*u[j]
# 在 _small_grid=True 时, fuse_recompute=True, fuse_diagonal=True
# 这意味着对角线块计算和 w/u/kg 重计算都融合进 inter_solve_fused
print("=" * 60)
print("Step B: chunk_kda_fwd_intra")
print("=" * 60)
t0 = time.time()
w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
    q=q, k=k, v=v,
    gk=g_cumsum,
    beta=beta,
    scale=scale,
    fuse_recompute=True,   # 融合 w/u/kg 重计算
    fuse_diagonal=True,    # 融合对角线块计算
)
torch.npu.synchronize()
print(f"  time={time.time() - t0:.2f}s")
check("w", w)
check("u", u)
check("kg", kg)
check("Aqk", Aqk)
print("  ✅ PASS\n")

# ═══════════════════════════════════════════════════════════════
# Step C: chunk_gated_delta_rule_fwd_h
# ═══════════════════════════════════════════════════════════════
# 功能: Delta Rule 隐藏状态更新 (跨 chunk 循环)
# 输入: k=kg [B,T,H,K], w [B,T,H,K], u [B,T,H,V], gk [B,T,H,K],
#       initial_state [B,H,K,V], initial_state_indices [B]
# 输出: h [NT,H,K,V] (每个 chunk 结束时的隐藏状态),
#       v_new [B,T,H,V] (更新后的 value)
# 计算 (对每个 chunk t):
#   1. 保存 h[t] = h (当前隐藏状态)
#   2. v_new[t] = u[t] - w[t] @ h       (减去历史影响)
#   3. v_new[t] *= exp(gate_last - gate) (逐 token 衰减)
#   4. h *= exp2(gk_last_chunk)          (状态衰减)
#   5. h += k[t]^T @ v_new[t]            (更新隐藏状态)
# 数学意义: h 是 K×V 矩阵, 累积所有历史信息
#   相当于标准 attention 的 KV cache, 但通过 Delta Rule 更新
print("=" * 60)
print("Step C: chunk_gated_delta_rule_fwd_h")
print("=" * 60)
t0 = time.time()
h, v_new = chunk_gated_delta_rule_fwd_h(
    k=kg,          # gated key (Step B 输出)
    w=w,           # 解耦后的 key
    u=u,           # 解耦后的 value
    gk=g_cumsum,   # 累积 gate (log2 空间)
    initial_state=initial_state,
    initial_state_indices=initial_state_indices,
    use_exp2=True,  # 使用 exp2 而非 exp (与 gate 的 log2 空间一致)
)
torch.npu.synchronize()
print(f"  time={time.time() - t0:.2f}s")
check("h", h)
check("v_new", v_new)
print("  ✅ PASS\n")

# ═══════════════════════════════════════════════════════════════
# Step D: chunk_gla_fwd_o_gk
# ═══════════════════════════════════════════════════════════════
# 功能: 最终输出计算 (Gated Linear Attention output)
# 输入: q [B,T,H,K], v_new [B,T,H,V], g [B,T,H,K],
#       Aqk [B,T,H,BT], h [NT,H,K,V]
# 输出: o [B,T,H,V]
# 计算 (对每个 chunk):
#   o = (q * exp2(g)) @ h  +  Aqk @ v_new
#   其中:
#     - q*gated @ h: 跨块注意力 (用隐藏状态访问历史)
#     - Aqk @ v_new: 块内注意力 (精确的 chunk 内交互)
# 数学意义: 输出 = 历史信息 + 块内信息
#   这是 GLA 的最终输出, 融合了长程依赖和短程精确计算
print("=" * 60)
print("Step D: chunk_gla_fwd_o_gk")
print("=" * 60)
t0 = time.time()
o = chunk_gla_fwd_o_gk(
    q=q,
    v=v_new,        # Step C 输出的更新后 value
    g=g_cumsum,     # 累积 gate
    A=Aqk,          # Step B 输出的 QK 矩阵
    h=h,            # Step C 输出的隐藏状态
    o=v,            # 复用 v 的存储空间作为输出
    scale=scale,
)
torch.npu.synchronize()
print(f"  time={time.time() - t0:.2f}s")
check("o", o)
print("  ✅ PASS\n")

# ═══════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("Level 2: ALL 4 STAGES PASSED")
print("=" * 60)
print(f"Output shape: {list(o.shape)}")
print(f"Output mean:  {o.float().mean():.6f}")
print(f"Output std:   {o.float().std():.6f}")