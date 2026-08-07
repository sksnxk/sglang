#!/usr/bin/env python3
"""
Level 3: chunk_kda() 完整端到端测试
====================================

验证完整 chunk_kda() 函数能正常运行。这是 Level 2 的 4 个子阶段串联后的端到端测试。

测试配置参数:
  B  = 1      # batch size
  T  = 128    # 总 token 数 (2 个 chunk)
  H  = 2      # head 数
  K  = 64     # key/query 维度
  V  = 64     # value 维度
  chunk_size = 64

参数含义同 Level 2。

chunk_kda() 内部调用链 (A_log 路径):
  chunk_kda() → chunk_kda_fwd()
    ├── kda_gate_chunk_cumsum()          # Step A: gate 激活 + cumsum
    ├── chunk_kda_fwd_intra(fuse=True)   # Step B: 块内 + 块间求解
    ├── chunk_gated_delta_rule_fwd_h()   # Step C: Delta Rule H
    └── chunk_gla_fwd_o_gk()            # Step D: 最终输出

注意:
  - 第一次运行包含 autotune 预热, 耗时较长
  - 第二次运行是纯执行时间
  - 如果遇到 aicore timeout (507014), 先删除 ~/.triton/cache/ 然后重试
  - 如果输出 NaN, 可能是 autotune cache 问题, 删除 cache 后重试

运行方式:
  docker exec -it triton-ascend-env-zhm bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_npu/lib:$LD_LIBRARY_PATH
  export PYTHONPATH=/docker/zhm/0505_skill_test/sonnet/sglang/python:$PYTHONPATH
  export SGLANG_GDN_CHUNK_H_BV=16
  export SGLANG_GDN_CHUNK_H_NUM_WARPS=1
  export SGLANG_GDN_CHUNK_H_NUM_STAGES=1
  cd /docker/zhm/0505_skill_test/sonnet/sglang
  python3 kda_test/level3_end_to_end.py
"""

import sys, time
sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")
exec(open("/tmp/load_kda.py").read())

import torch
import torch_npu

# ═══════════════════════════════════════════════════════════════
# 测试参数
# ═══════════════════════════════════════════════════════════════
device = "npu"
dtype = torch.bfloat16
B, T, H, K, V = 1, 128, 2, 64, 64

# ═══════════════════════════════════════════════════════════════
# 构造输入数据
# ═══════════════════════════════════════════════════════════════
torch.manual_seed(42)

# q, k: 归一化后的 query 和 key, shape [B, T, H, K]
q = torch.nn.functional.normalize(
    torch.randn(B, T, H, K, dtype=torch.float32, device=device), dim=-1
).to(dtype)
k = torch.nn.functional.normalize(
    torch.randn(B, T, H, K, dtype=torch.float32, device=device), dim=-1
).to(dtype)

# v: value, shape [B, T, H, V]
v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1

# raw_gate: 原始 gate (未激活), shape [B, T, H, K]
# 使用 A_log 路径: gate = -exp(A_log) * softplus(raw_gate + dt_bias)
raw_gate = (torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.5 - 2.0).to(dtype)

# A_log: per-head 参数, shape [H]
A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.1

# dt_bias: per-head per-channel bias, shape [H*K]
dt_bias = torch.randn(H * K, dtype=torch.float32, device=device) * 0.1

# beta: sigmoid-gated beta, shape [B, T, H]
beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()

# initial_state: 初始隐藏状态, shape [B, H, K, V]
initial_state = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.05

# initial_state_indices: 每个 batch 的索引, shape [B]
initial_state_indices = torch.arange(B, dtype=torch.int32, device=device)

print(f"Input: q={list(q.shape)}, k={list(k.shape)}, v={list(v.shape)}")
print(f"       g={list(raw_gate.shape)}, beta={list(beta.shape)}")
print(f"       A_log={list(A_log.shape)}, dt_bias={list(dt_bias.shape)}")
print(f"       initial_state={list(initial_state.shape)}")
print()

# ═══════════════════════════════════════════════════════════════
# Run 1: Autotune warmup (首次运行, 包含 autotune 搜索)
# ═══════════════════════════════════════════════════════════════
print("Run 1 (autotune warmup, may be slow)...")
t0 = time.time()
try:
    output = chunk_kda(
        q=q, k=k, v=v,
        g=raw_gate,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    torch.npu.synchronize()
    t1 = time.time()
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    print(f"  time={t1 - t0:.2f}s")
    print(f"  shape={list(output.shape)}")
    print(f"  mean={output.float().mean():.6f}")
    print(f"  nan={has_nan}, inf={has_inf}")
    if has_nan:
        print("  ⚠️  WARNING: Output contains NaN! Try deleting ~/.triton/cache/ and rerun")
    else:
        print("  ✅ Run 1 PASS")
except Exception as e:
    print(f"  ❌ FAIL after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ═══════════════════════════════════════════════════════════════
# Run 2: Pure execution (cached, 纯执行时间)
# ═══════════════════════════════════════════════════════════════
print("Run 2 (cached execution)...")
t0 = time.time()
try:
    output = chunk_kda(
        q=q, k=k, v=v,
        g=raw_gate,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    torch.npu.synchronize()
    t1 = time.time()
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    print(f"  time={t1 - t0:.4f}s")
    print(f"  shape={list(output.shape)}")
    print(f"  mean={output.float().mean():.6f}")
    print(f"  nan={has_nan}, inf={has_inf}")
    if has_nan:
        print("  ⚠️  WARNING: Output contains NaN!")
    else:
        print("  ✅ Run 2 PASS")
except Exception as e:
    print(f"  ❌ FAIL after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ═══════════════════════════════════════════════════════════════
# Run 3: pre-activated gate 路径 (A_log=None)
# ═══════════════════════════════════════════════════════════════
# 验证不使用 A_log 激活的路径: 直接传入已激活的 gate
# 此路径走 chunk_local_cumsum 而非 kda_gate_chunk_cumsum
print("Run 3 (pre-activated gate, A_log=None)...")
g_pre_activated = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.5
t0 = time.time()
try:
    output3 = chunk_kda(
        q=q, k=k, v=v,
        g=g_pre_activated,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        A_log=None,  # 跳过 gate 激活, 直接 cumsum
    )
    torch.npu.synchronize()
    t1 = time.time()
    has_nan = torch.isnan(output3).any().item()
    print(f"  time={t1 - t0:.4f}s")
    print(f"  shape={list(output3.shape)}")
    print(f"  mean={output3.float().mean():.6f}")
    print(f"  nan={has_nan}")
    if has_nan:
        print("  ⚠️  WARNING: Output contains NaN!")
    else:
        print("  ✅ Run 3 PASS")
except Exception as e:
    print(f"  ❌ FAIL after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print()

# ═══════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("Level 3: chunk_kda() end-to-end test completed")
print("=" * 60)