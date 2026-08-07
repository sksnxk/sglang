#!/usr/bin/env python3
"""Profile KDA kernels with Python timing (no msprof overhead).

Kimi K3 KDA layer shapes (from linear_attn_config):
  - num_heads (H) = 32
  - head_dim (K=V) = 128
  - chunk_size = 64, sub_chunk (BC) = 16

Profiles individual kernels at multiple T sizes to understand scaling.
"""
import gc
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")

from sglang.kernels.ops.attention.fla.kda import (
    RCP_LN2,
    chunk_gla_fwd_o_gk,
    chunk_kda_fwd_intra,
    chunk_local_cumsum,
    kda_gate_chunk_cumsum,
)
from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.kernels.ops.attention.fla.chunk_intra_token_parallel import (
    chunk_kda_fwd_intra_token_parallel,
)

DEVICE = "npu"
DTYPE = torch.bfloat16
CHUNK_SIZE = 64
BC = 16
K = 128
V = 128

# Profile at multiple T sizes to observe scaling
T_SIZES = [128, 256, 512, 1024, 2048, 4096]
H_VALUES = [2, 4, 8]  # H=2 for base timing, H=4/8 to check H scaling

# For extrapolation to Kimi K3 (H=32, T=131072)
SCALE_TARGET_H = 32
SCALE_TARGET_T = 131072

N_WARMUP = 2
N_ITER = 3  # keep low to avoid timeouts

torch.manual_seed(42)

print("=== KDA Kernel Timing vs Sequence Length ===")
print(f"  K={K}, V={V}, chunk_size={CHUNK_SIZE}, sub_chunk={BC}")
print(f"  T sizes: {T_SIZES}")
print(f"  H values: {H_VALUES}")
print()

all_results = {}

for H in H_VALUES:
    for T in T_SIZES:
        B = 1
        NT = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
        SCALE = K**-0.5

        print(f"--- T={T}, H={H}, NT={NT} ---", flush=True)

        # Create inputs
        q = F.normalize(
            torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE), dim=-1
        ).to(DTYPE)
        k = F.normalize(
            torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE), dim=-1
        ).to(DTYPE)
        v = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE) * 0.1
        raw_gate = (
            torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.5 - 2.0
        ).to(DTYPE)
        A_log = torch.randn(H, dtype=torch.float32, device=DEVICE) * 0.1
        dt_bias = torch.randn(H * K, dtype=torch.float32, device=DEVICE) * 0.1
        beta = torch.rand(B, T, H, dtype=DTYPE, device=DEVICE).sigmoid()
        initial_state = (
            torch.randn(B, H, K, V, dtype=torch.float32, device=DEVICE) * 0.05
        )
        indices = torch.arange(B, dtype=torch.int32, device=DEVICE)

        # Warmup
        g_cumsum = kda_gate_chunk_cumsum(
            raw_gate, A_log=A_log, chunk_size=CHUNK_SIZE, scale=RCP_LN2, dt_bias=dt_bias
        )
        torch.npu.synchronize()

        w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g_cumsum, beta=beta,
            scale=SCALE, cu_seqlens=None, chunk_size=CHUNK_SIZE,
            fuse_diagonal=False, fuse_recompute=False,
        )
        torch.npu.synchronize()

        h, v_new = chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=g_cumsum,
            initial_state=initial_state.clone(),
            initial_state_indices=indices,
            cu_seqlens=None, use_exp2=True,
        )
        torch.npu.synchronize()

        o = chunk_gla_fwd_o_gk(
            q=q, v=v_new, g=g_cumsum, A=Aqk, h=h, o=v, scale=SCALE,
        )
        torch.npu.synchronize()
        del w, u, kg, Aqk, h, v_new, o, g_cumsum
        gc.collect()

        # Timing
        results = {}

        # Step A
        times = []
        for _ in range(N_ITER):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            g_cumsum = kda_gate_chunk_cumsum(
                raw_gate, A_log=A_log, chunk_size=CHUNK_SIZE, scale=RCP_LN2, dt_bias=dt_bias
            )
            torch.npu.synchronize()
            times.append(time.perf_counter() - t0)
        results["StepA_gate_cumsum"] = sum(times) / len(times) * 1000

        # Step B1: token_parallel
        Aqk_tp = torch.zeros(B, T, H, CHUNK_SIZE, device=DEVICE, dtype=torch.float32)
        Akk_tp = torch.zeros(B, T, H, BC, device=DEVICE, dtype=torch.float32)
        times = []
        for _ in range(N_ITER):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            chunk_kda_fwd_intra_token_parallel(
                q=q, k=k, gk=g_cumsum, beta=beta, Aqk=Aqk_tp, Akk=Akk_tp,
                scale=SCALE, cu_seqlens=None, chunk_size=CHUNK_SIZE, sub_chunk_size=BC,
            )
            torch.npu.synchronize()
            times.append(time.perf_counter() - t0)
        results["StepB1_token_parallel"] = sum(times) / len(times) * 1000
        del Aqk_tp, Akk_tp

        # Step B full: chunk_kda_fwd_intra
        times = []
        for _ in range(N_ITER):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g_cumsum, beta=beta,
                scale=SCALE, cu_seqlens=None, chunk_size=CHUNK_SIZE,
                fuse_diagonal=False, fuse_recompute=False,
            )
            torch.npu.synchronize()
            times.append(time.perf_counter() - t0)
        results["StepB_intra_full"] = sum(times) / len(times) * 1000

        # Step C: delta_h
        times = []
        for _ in range(N_ITER):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            h, v_new = chunk_gated_delta_rule_fwd_h(
                k=kg, w=w, u=u, gk=g_cumsum,
                initial_state=initial_state.clone(),
                initial_state_indices=indices,
                cu_seqlens=None, use_exp2=True,
            )
            torch.npu.synchronize()
            times.append(time.perf_counter() - t0)
        results["StepC_delta_h"] = sum(times) / len(times) * 1000

        # Step D: gla_output
        times = []
        for _ in range(N_ITER):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            o = chunk_gla_fwd_o_gk(
                q=q, v=v_new, g=g_cumsum, A=Aqk, h=h, o=v, scale=SCALE,
            )
            torch.npu.synchronize()
            times.append(time.perf_counter() - t0)
        results["StepD_gla_output"] = sum(times) / len(times) * 1000

        # Step A alt: local_cumsum
        activated_gate = -torch.exp(A_log.view(1, 1, H, 1)) * F.softplus(
            raw_gate.float() + dt_bias.view(1, 1, H, K)
        )
        activated_gate = activated_gate.to(DTYPE)
        times = []
        for _ in range(N_ITER):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            _ = chunk_local_cumsum(activated_gate.clone(), chunk_size=CHUNK_SIZE, scale=RCP_LN2)
            torch.npu.synchronize()
            times.append(time.perf_counter() - t0)
        results["StepA_local_cumsum"] = sum(times) / len(times) * 1000

        total = sum(results.values())
        key = f"H={H},T={T}"
        all_results[key] = {**results, "total": total}

        # Print per-kernel summary
        for name, ms in sorted(results.items(), key=lambda x: -x[1]):
            pct = ms / total * 100 if total > 0 else 0
            print(f"    {name:30s}: {ms:8.2f}ms ({pct:5.1f}%)", flush=True)
        print(f"    {'TOTAL':30s}: {total:8.2f}ms", flush=True)

        # Cleanup
        del w, u, kg, Aqk, h, v_new, o, g_cumsum, q, k, v, raw_gate, A_log, dt_bias, beta, initial_state, indices
        gc.collect()
        torch.npu.empty_cache()

# ── Summary Table ──
print(f"\n{'='*100}")
print(f"=== Summary: Per-Kernel Time (ms) vs Sequence Length ===")
print(f"{'='*100}")
kernels = ["StepA_gate_cumsum", "StepA_local_cumsum", "StepB1_token_parallel",
           "StepB_intra_full", "StepC_delta_h", "StepD_gla_output"]

# Header
header = f"{'H':>2s} {'T':>6s} " + " ".join(f"{k.replace('Step','')[3:]:>14s}" for k in kernels) + f" {'total':>10s}"
print(header)
print("-" * len(header))

for H in H_VALUES:
    for T in T_SIZES:
        key = f"H={H},T={T}"
        if key not in all_results:
            continue
        r = all_results[key]
        row = f"{H:2d} {T:6d} "
        for k in kernels:
            row += f"{r[k]:14.2f}"
        row += f"{r['total']:10.2f}"
        print(row)

# ── Extrapolation to Kimi K3 128K ──
print(f"\n{'='*100}")
print(f"=== Extrapolation to Kimi K3 (H=32, T=131072) ===")
print(f"{'='*100}")
print(f"Using H=8, T=4096 as base (closest to target while staying within NPU limits)")
print(f"  Scale_H = 32/8 = 4x, Scale_T = 131072/4096 = 32x")
print()

base_key = "H=8,T=4096"
if base_key in all_results:
    base = all_results[base_key]
    print(f"{'Kernel':30s} {'base_ms':>10s} {'scale_factor':>14s} {'est_128K_ms':>14s} {'%':>8s}")
    print("-" * 80)
    est_total = 0
    est_results = {}
    for k in kernels:
        if "delta_h" in k:
            scale = 4  # only H scaling
        else:
            scale = 4 * 32  # H * T scaling
        est = base[k] * scale
        est_results[k] = est
        est_total += est
        print(f"{k:30s} {base[k]:10.2f} {scale:14.0f}x {est:14.1f}")

    for k in kernels:
        pct = est_results[k] / est_total * 100 if est_total > 0 else 0
        print(f"{k:30s} {base[k]:10.2f} {scale:14.0f}x {est_results[k]:14.1f} {pct:7.1f}%")

    print("-" * 80)
    print(f"{'TOTAL':30s} {base['total']:10.2f} {'':14s} {est_total:14.1f}ms")

    # Bottleneck
    bottleneck = max(est_results, key=est_results.get)
    print(f"\n  Bottleneck: {bottleneck} ({est_results[bottleneck]:.1f}ms, {est_results[bottleneck]/est_total*100:.1f}%)")
else:
    print(f"WARNING: {base_key} not available, using max available size")
    last_key = f"H={H_VALUES[-1]},T={T_SIZES[-1]}"
    if last_key in all_results:
        print(f"Using {last_key}: {all_results[last_key]}")

print(f"\nDone.")