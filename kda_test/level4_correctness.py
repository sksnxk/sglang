#!/usr/bin/env python3
"""
Level 4: KDA 正确性验证 (与 naive 递归实现对比)
================================================

将 chunk_kda() 的输出与逐 token 的 naive 指数递归实现对比，
验证 chunkwise 分块算法与精确递归在数值上一致。

测试配置参数:
  case 1: lengths=[129],            use_varlen=False, fuse_gate=False
  case 2: lengths=[15,16,17,63,65], use_varlen=True,  fuse_gate=True
  case 3: lengths=[2]*129,          use_varlen=True,  fuse_gate=False

参数含义:
  - lengths: 每个序列的 token 数
    * case 1: 单序列 129 tokens, 跨 3 个 chunk (129 = 2*64+1)
    * case 2: 5 个变长序列, 测试 varlen 路径
    * case 3: 129 个极短序列 (各 2 tokens), 测试 _small_grid=False 的大 grid 路径
  - use_varlen: 是否使用 cu_seqlens 变长模式
  - fuse_gate: True → A_log 路径 (gate 在 kernel 内激活)
               False → 调用方已激活 gate

Naive 递归算法 (逐 token):
  对于每个 token t:
    state *= exp(g[t])                    # gate 衰减隐藏状态
    residual = v[t] - state @ k[t]        # Delta Rule: 减去历史影响
    state += residual ⊗ k[t] * beta[t]    # 更新隐藏状态
    output[t] = state @ q[t] * scale      # 线性注意力输出

chunk_kda 算法 (chunkwise):
  将 T 分块为 NT 个 chunk, 每块内精确计算 QK/KK, 跨块用隐藏状态近似

正确性判据: relative_rmse < 0.01 (1%)

注意:
  - 此测试需要完整 sglang 安装才能运行 (需要 sglang.srt.utils.common 等)
  - 如果只安装了 KDA 的 mock 环境, 请先运行 Level 1-3
  - 此测试的 naive 实现直接内嵌, 不依赖 sglang 测试框架

运行方式:
  docker exec -it triton-ascend-env-zhm bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_npu/lib:$LD_LIBRARY_PATH
  export PYTHONPATH=/docker/zhm/0505_skill_test/sonnet/sglang/python:$PYTHONPATH
  export SGLANG_GDN_CHUNK_H_BV=16
  export SGLANG_GDN_CHUNK_H_NUM_WARPS=1
  export SGLANG_GDN_CHUNK_H_NUM_STAGES=1
  cd /docker/zhm/0505_skill_test/sonnet/sglang
  python3 kda_test/level4_correctness.py
"""

import sys, time
sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")
exec(open("/tmp/load_kda.py").read())

import torch
import torch_npu

# ═══════════════════════════════════════════════════════════════
# Naive 递归参考实现 (逐 token)
# ═══════════════════════════════════════════════════════════════
def naive_recurrent(q, k, v, g, beta, initial_state, lengths):
    """
    逐 token 的 Delta Rule 递归计算。

    参数:
      q,k,v: [1, T, H, K/V]
      g:     [1, T, H, K] 已激活的 gate (natural log 空间)
      beta:  [1, T, H]
      initial_state: [B, H, K, V]
      lengths: 每个序列的 token 数

    返回:
      output: [1, T, H, V]
      final_state: [B, H, K, V]
    """
    q, k, v, g, beta = (tensor.float() for tensor in (q, k, v, g, beta))
    scale = q.shape[-1] ** -0.5
    output = torch.empty_like(v)
    final_state = initial_state.float().clone()

    offset = 0
    for sequence_index, length in enumerate(lengths):
        state = final_state[sequence_index]  # [H, K, V]
        for token_index in range(offset, offset + length):
            # gate 衰减: state *= exp(g[t])
            state = state * g[0, token_index].exp().unsqueeze(-2)
            # Delta Rule residual: v[t] - state[:, k] @ k[t]
            residual = v[0, token_index] - torch.einsum(
                "hvk,hk->hv", state, k[0, token_index]
            )
            # 更新 state: state += residual ⊗ k[t] * beta[t]
            state = state + torch.einsum(
                "hv,hk->hvk",
                residual * beta[0, token_index, :, None],
                k[0, token_index],
            )
            # 输出: o[t] = state @ q[t] * scale
            output[0, token_index] = (
                torch.einsum("hvk,hk->hv", state, q[0, token_index]) * scale
            )
        final_state[sequence_index] = state
        offset += length
    return output, final_state


def relative_rmse(actual, expected):
    """计算相对 RMSE: sqrt(mean((actual-expected)^2)) / sqrt(mean(expected^2))"""
    error = (actual.float() - expected.float()).square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return (error / baseline).item()


# ═══════════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════════
device = "npu"
dtype = torch.bfloat16
num_heads, head_dim = 2, 64

# 3 个测试用例覆盖不同路径
# case 1: 单序列 129 tokens, 不 fuse gate (gate 已激活)
# case 2: 5 个变长序列, fuse gate (A_log 路径)
# case 3: 129 个极短序列, 大 grid (_small_grid=False)
cases = (
    ([129], False, False),
    ([15, 16, 17, 63, 65], True, True),
    ([2] * 129, True, False),
)

all_passed = True
for lengths, use_varlen, fuse_gate in cases:
    print("=" * 60)
    print(f"Case: lengths={lengths[:5]}{'...' if len(lengths)>5 else ''} "
          f"(total={sum(lengths)}), varlen={use_varlen}, fuse_gate={fuse_gate}")
    print("=" * 60)

    torch.manual_seed(42)
    total_tokens = sum(lengths)
    shape = (1, total_tokens, num_heads, head_dim)

    # 构造输入
    q = torch.nn.functional.normalize(
        torch.randn(shape, dtype=torch.float32, device=device), dim=-1
    ).to(dtype)
    k = torch.nn.functional.normalize(
        torch.randn(shape, dtype=torch.float32, device=device), dim=-1
    ).to(dtype)
    v = torch.randn(shape, dtype=dtype, device=device) * 0.1
    raw_gate = (
        torch.randn(shape, dtype=torch.float32, device=device) * 0.5 - 2.0
    ).to(dtype)
    A_log = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = (
        torch.randn(num_heads * head_dim, dtype=torch.float32, device=device) * 0.1
    )

    # 激活 gate: gate = -exp(A_log) * softplus(raw_gate + dt_bias)
    activated_gate = -torch.exp(
        A_log.view(1, 1, num_heads, 1)
    ) * torch.nn.functional.softplus(
        raw_gate.float() + dt_bias.view(1, 1, num_heads, head_dim)
    )

    kernel_gate = raw_gate if fuse_gate else activated_gate.to(dtype)
    reference_gate = activated_gate if fuse_gate else kernel_gate.float()

    beta = torch.rand(1, total_tokens, num_heads, dtype=dtype, device=device).sigmoid()
    initial_state = (
        torch.randn(len(lengths), num_heads, head_dim, head_dim,
                    dtype=torch.float32, device=device) * 0.05
    )

    # ── 计算 naive 参考输出 (CPU) ──
    t0 = time.time()
    expected_output, expected_state = naive_recurrent(
        q=q.cpu(), k=k.cpu(), v=v.cpu(),
        g=reference_gate.cpu(),
        beta=beta.cpu(),
        initial_state=initial_state.cpu(),
        lengths=lengths,
    )
    print(f"  Naive recurrent: {time.time() - t0:.2f}s")

    # ── 计算 chunk_kda 输出 (NPU) ──
    actual_state = initial_state.clone()
    cu_seqlens = None
    if use_varlen:
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32, device=device,
        )

    t0 = time.time()
    actual_output = chunk_kda(
        q=q.clone(), k=k.clone(), v=v.clone(),
        g=kernel_gate.clone(),
        beta=beta.clone(),
        initial_state=actual_state,
        initial_state_indices=torch.arange(
            len(lengths), dtype=torch.int32, device=device
        ),
        cu_seqlens=cu_seqlens,
        A_log=A_log if fuse_gate else None,
        dt_bias=dt_bias if fuse_gate else None,
    )
    torch.npu.synchronize()
    print(f"  chunk_kda (NPU):  {time.time() - t0:.2f}s")

    # ── 对比 ──
    output_error = relative_rmse(actual_output.cpu(), expected_output)
    state_error = relative_rmse(actual_state.cpu(), expected_state)

    # 移除 padding 的影响
    if use_varlen:
        mask = torch.zeros(total_tokens, dtype=torch.bool)
        offset = 0
        for length in lengths:
            mask[offset:offset + length] = True
            offset += length
        valid_output = actual_output[0, mask].cpu()
        valid_expected = expected_output[0, mask]
        output_error = relative_rmse(valid_output, valid_expected)

    print(f"  output RMSE: {output_error:.6f}  (threshold: 0.01)")
    print(f"  state  RMSE: {state_error:.6f}")

    if output_error < 0.01:
        print(f"  ✅ PASS")
    else:
        print(f"  ❌ FAIL: output RMSE {output_error:.6f} exceeds 0.01")
        all_passed = False
    print()

# ═══════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
if all_passed:
    print("Level 4: ALL CORRECTNESS TESTS PASSED")
else:
    print("Level 4: SOME TESTS FAILED")
print("=" * 60)