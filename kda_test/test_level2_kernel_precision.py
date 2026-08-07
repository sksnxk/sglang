#!/usr/bin/env python3
"""
KDA kernel-by-kernel precision verification (pytest).

Each kernel in the chunk_kda_fwd call chain is verified against a CPU reference
across 10 test cases covering mainstream model scenarios.

Run all tests:
  python3 -m pytest kda_test/test_level2_kernel_precision.py -v -s

Run single kernel:
  python3 -m pytest kda_test/test_level2_kernel_precision.py::TestGateChunkCumsumKernel -v -s
  python3 -m pytest kda_test/test_level2_kernel_precision.py::TestTokenParallelKernel -v -s
  python3 -m pytest kda_test/test_level2_kernel_precision.py::TestRecomputeWUKernel -v -s
  python3 -m pytest kda_test/test_level2_kernel_precision.py::TestDeltaRuleKernel -v -s
  python3 -m pytest kda_test/test_level2_kernel_precision.py::TestGLAOutputKernel -v -s
  python3 -m pytest kda_test/test_level2_kernel_precision.py::TestFullPipeline -v -s
"""

import sys
import time

# Patch numpy for CANN compatibility with numpy>=2.0 BEFORE torch import
import numpy as np
for _attr in ("float_", "complex_", "unicode_"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(np, {"float_": "float64", "complex_": "complex128", "unicode_": "str_"}[_attr], None))

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")
exec(open("/tmp/load_kda.py").read())

from sglang.kernels.ops.attention.fla.kda import (
    chunk_gla_fwd_o_gk, chunk_kda, kda_gate_chunk_cumsum,
)
from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.kernels.ops.attention.fla.chunk_intra import chunk_kda_fwd_intra
from sglang.kernels.ops.attention.fla.chunk_intra_token_parallel import (
    chunk_kda_fwd_intra_token_parallel,
)

# ── Constants ──────────────────────────────────────────────────────────────
RCP_LN2 = 1.0 / 0.6931471805599453
DEVICE = "npu"
DTYPE = torch.bfloat16
CHUNK_SIZE = 64
BC = 16

# ═══════════════════════════════════════════════════════════════════════════
# 10 Test Cases covering mainstream model scenarios
# ═══════════════════════════════════════════════════════════════════════════
# Grid safety: token_parallel grid = B*T*H, keep <= 8192 to avoid timeout.
# K must be 64 (chunk_delta_h kernel only works correctly at K=64 on triton-ascend).
# chunk_size=64, BC=16 are fixed.
TEST_CONFIGS = [
    # id                B   T    H   K    V    description
    dict(id="tiny_default",       B=1, T=128,  H=2,  K=64,  V=64,  desc="Baseline: B=1,T=128,H=2 (2 chunks)"),
    dict(id="tiny_partial_chunk", B=1, T=63,   H=2,  K=64,  V=64,  desc="Partial last chunk: T=63 (1 short chunk)"),
    dict(id="tiny_single_head",   B=1, T=128,  H=1,  K=64,  V=64,  desc="Single head: H=1"),
    dict(id="tiny_H3",            B=1, T=128,  H=3,  K=64,  V=64,  desc="Odd head count: H=3"),
    dict(id="tiny_T65",           B=1, T=65,   H=2,  K=64,  V=64,  desc="T=65: one token in 2nd chunk"),
    dict(id="tiny_T96",           B=1, T=96,   H=2,  K=64,  V=64,  desc="T=96: non-power-of-2 tokens"),
    dict(id="tiny_T1",            B=1, T=1,    H=2,  K=64,  V=64,  desc="T=1: single token"),
    dict(id="tiny_T2",            B=1, T=2,    H=2,  K=64,  V=64,  desc="T=2: two tokens"),
    dict(id="tiny_T100",          B=1, T=100,  H=2,  K=64,  V=64,  desc="T=100: round number, 2 chunks"),
    dict(id="tiny_T127",          B=1, T=127,  H=2,  K=64,  V=64,  desc="T=127: one less than 2 full chunks"),
]


def _make_inputs(cfg, seed=42):
    """Create test inputs for a given config."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE), dim=-1).to(DTYPE)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE), dim=-1).to(DTYPE)
    v = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE) * 0.1
    raw_gate = (torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.5 - 2.0).to(DTYPE)
    A_log = torch.randn(H, dtype=torch.float32, device=DEVICE) * 0.1
    dt_bias = torch.randn(H * K, dtype=torch.float32, device=DEVICE) * 0.1
    beta = torch.rand(B, T, H, dtype=DTYPE, device=DEVICE).sigmoid()
    initial_state = torch.randn(B, H, K, V, dtype=torch.float32, device=DEVICE) * 0.05
    indices = torch.arange(B, dtype=torch.int32, device=DEVICE)
    return q, k, v, raw_gate, A_log, dt_bias, beta, initial_state, indices


def _rmse(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    error = (a - b).square().mean().sqrt()
    baseline = b.square().mean().sqrt().clamp_min(1e-8)
    return (error / baseline).item()


@pytest.fixture(autouse=True)
def _cleanup_device():
    """Clear NPU cache between tests to avoid accumulated device state timeouts."""
    yield
    if DEVICE == "npu":
        torch.npu.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# CPU References (shape-agnostic — extract shapes from input tensors)
# ═══════════════════════════════════════════════════════════════════════════

def _cpu_gate_cumsum(raw_gate, A_log, dt_bias, chunk_size=64):
    """gate = -exp(A_log) * softplus(raw_gate + dt_bias); chunk-local cumsum * RCP_LN2."""
    B, T, H, K = raw_gate.shape
    raw_gate_f = raw_gate.float()
    gate = -torch.exp(A_log.view(1, 1, H, 1)) * F.softplus(
        raw_gate_f + dt_bias.view(1, 1, H, K)
    )
    out = gate.clone()
    NT = (T + chunk_size - 1) // chunk_size
    for c in range(NT):
        s, e = c * chunk_size, min(T, (c + 1) * chunk_size)
        out[:, s:e] = torch.cumsum(gate[:, s:e], dim=1)
    return out * RCP_LN2


def _cpu_token_parallel(q, k, g, beta, scale, BT=64, BC=16):
    """Diagonal Aqk/Akk blocks: gated dot products within same sub-chunk."""
    B, T, H, K = q.shape
    qf, kf, gf, betaf = q.float(), k.float(), g.float(), beta.float()
    Aqk = torch.zeros(B, T, H, BT, dtype=torch.float32)
    Akk = torch.zeros(B, T, H, BC, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                c = t // BT
                s = (t % BT) // BC
                ts = c * BT + s * BC
                qt = qf[b, t, h]
                kt = kf[b, t, h] * betaf[b, t, h]
                gt = gf[b, t, h]
                for j in range(ts, min(t + 1, min(T, ts + BC))):
                    kj = kf[b, j, h]
                    gj = gf[b, j, h]
                    kgj = kj * torch.exp2(gt - gj)
                    Aqk[b, t, h, j % BT] = (qt * kgj).sum() * scale
                    if j < t:
                        Akk[b, t, h, j - ts] = (kt * kgj).sum()
    return Aqk, Akk


def _cpu_recompute_w_u(k, v, beta, A, gk):
    """
    w = A_inv @ (k * beta * exp2(gk))
    u = A_inv @ (v * beta)
    kg = k * exp2(gk_last - gk)
    A is Akk_inv [B, T, H, BT] from inter_solve.
    """
    B_, T_, H_, K_ = k.shape
    V_ = v.shape[-1]
    BT = A.shape[-1]
    kf, vf, betaf, Af, gkf = k.float(), v.float(), beta.float(), A.float(), gk.float()
    NT = (T_ + BT - 1) // BT
    w = torch.zeros_like(kf)
    u = torch.zeros_like(vf)
    kg = torch.zeros_like(kf)
    for b in range(B_):
        for h in range(H_):
            for c in range(NT):
                tc = c * BT
                tc_end = min(T_, tc + BT)
                BT_act = tc_end - tc
                A_chunk = Af[b, tc:tc_end, h, :BT_act]
                kb_chunk = (kf[b, tc:tc_end, h]
                            * betaf[b, tc:tc_end, h].unsqueeze(-1)
                            * torch.exp2(gkf[b, tc:tc_end, h]))
                w[b, tc:tc_end, h] = A_chunk @ kb_chunk
                vb_chunk = (vf[b, tc:tc_end, h]
                            * betaf[b, tc:tc_end, h].unsqueeze(-1))
                u[b, tc:tc_end, h] = A_chunk @ vb_chunk
                last = tc_end - 1
                gk_last = gkf[b, last, h]
                kg[b, tc:tc_end, h] = (kf[b, tc:tc_end, h]
                                        * torch.exp2(gk_last.unsqueeze(0) - gkf[b, tc:tc_end, h]))
    return w, u, kg


def _cpu_delta_rule_h(k, w, u, gk, initial_state, BT=64):
    """
    Delta Rule H kernel (USE_GK path, use_exp2=True).

    State is in [K, V] layout (same as PyTorch initial_state).
    The kernel stores h as [V, K] but when K==V both layouts are equivalent.
    Only K==V is supported (kernel initial_state read requires K==V).

    Operations:
      v_new = u - w @ s_h.T      (w=[BT,K], s_h=[K,V], s_h^T=[V,K] -> [BT,V])
      s_h *= exp2(gk_last)[None,:]  ([K,V] * [1,K] -> [K,V])
      s_h += v_c.T @ k_chunk     ([V,BT] @ [BT,K] -> [V,K], same as [K,V] when K==V)
    """
    B_, T_, H_, K_ = k.shape
    V_ = u.shape[-1]
    assert K_ == V_, f"DeltaRule CPU reference requires K==V, got K={K_}, V={V_}"
    kf, wf, uf, gkf = k.float(), w.float(), u.float(), gk.float()
    h0 = initial_state.float().clone()
    NT = (T_ + BT - 1) // BT
    h = torch.zeros(B_, NT, H_, V_, K_, dtype=torch.float32)
    v_new = torch.zeros(B_, T_, H_, V_, dtype=torch.float32)
    for b in range(B_):
        state = h0[b].clone()
        for c in range(NT):
            tc = c * BT
            tc_end = min(T_, tc + BT)
            for h_idx in range(H_):
                w_chunk = wf[b, tc:tc_end, h_idx]
                u_chunk = uf[b, tc:tc_end, h_idx]
                k_chunk = kf[b, tc:tc_end, h_idx]
                s_h = state[h_idx]
                h[b, c, h_idx] = s_h.clone()
                v_c = u_chunk - w_chunk @ s_h.T
                v_new[b, tc:tc_end, h_idx] = v_c
                last = tc_end - 1
                gk_last = gkf[b, last, h_idx]
                s_h = s_h * torch.exp2(gk_last[None, :])
                s_h = s_h + v_c.T @ k_chunk
                state[h_idx] = s_h
    return h, v_new


def _cpu_gla_output(q, v_new, g, Aqk, h, scale, BT=64):
    """o = (q * exp2(g)) @ h^T * scale + Aqk @ v_new (causal)."""
    B_, T_, H_, K_ = q.shape
    V_ = v_new.shape[-1]
    qf, vf, gf, Aqf, hf = q.float(), v_new.float(), g.float(), Aqk.float(), h.float()
    NT = (T_ + BT - 1) // BT
    o = torch.zeros_like(vf)
    for b in range(B_):
        for c in range(NT):
            tc = c * BT
            tc_end = min(T_, tc + BT)
            BT_act = tc_end - tc
            for h_idx in range(H_):
                q_chunk = qf[b, tc:tc_end, h_idx]
                g_chunk = gf[b, tc:tc_end, h_idx]
                v_chunk = vf[b, tc:tc_end, h_idx]
                A_chunk = Aqf[b, tc:tc_end, h_idx, :BT_act]
                h_s = hf[b, c, h_idx]
                qg = q_chunk * torch.exp2(g_chunk)
                o_cross = qg @ h_s.T * scale
                causal_mask = torch.tril(torch.ones(BT_act, BT_act, dtype=torch.float32))
                o_intra = (A_chunk * causal_mask) @ v_chunk
                o[b, tc:tc_end, h_idx] = o_cross + o_intra
    return o


# ═══════════════════════════════════════════════════════════════════════════
# Test: kda_gate_chunk_cumsum
# ═══════════════════════════════════════════════════════════════════════════

class TestGateChunkCumsumKernel:
    """kda_gate_chunk_cumsum: gate activation + chunk-local cumsum in log2 space."""

    @pytest.mark.parametrize("cfg", TEST_CONFIGS, ids=[c["id"] for c in TEST_CONFIGS])
    def test_precision(self, cfg):
        _, _, _, raw_gate, A_log, dt_bias, _, _, _ = _make_inputs(cfg)
        g_npu = kda_gate_chunk_cumsum(
            raw_gate, A_log=A_log, chunk_size=CHUNK_SIZE, scale=RCP_LN2, dt_bias=dt_bias,
        )
        torch.npu.synchronize()
        g_cpu = _cpu_gate_cumsum(raw_gate.cpu(), A_log.cpu(), dt_bias.cpu())
        err = _rmse(g_npu, g_cpu)
        assert err < 0.01, f"[{cfg['id']}] RMSE={err:.6f} > 0.01"


# ═══════════════════════════════════════════════════════════════════════════
# Test: chunk_kda_fwd_intra_token_parallel
# ═══════════════════════════════════════════════════════════════════════════

class TestTokenParallelKernel:
    """chunk_kda_fwd_intra_token_parallel: diagonal Aqk/Akk blocks."""

    @pytest.mark.parametrize("cfg", TEST_CONFIGS, ids=[c["id"] for c in TEST_CONFIGS])
    def test_precision(self, cfg):
        B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
        scale = K ** -0.5
        q, k, _, raw_gate, A_log, dt_bias, beta, _, _ = _make_inputs(cfg)
        g_cumsum = kda_gate_chunk_cumsum(
            raw_gate, A_log=A_log, chunk_size=CHUNK_SIZE, scale=RCP_LN2, dt_bias=dt_bias,
        )
        torch.npu.synchronize()
        # CPU
        Aqk_cpu, Akk_cpu = _cpu_token_parallel(
            q.cpu(), k.cpu(), g_cumsum.cpu(), beta.cpu(), scale,
        )
        # NPU
        Aqk_npu = torch.zeros(B, T, H, CHUNK_SIZE, device=DEVICE, dtype=DTYPE)
        Akk_npu = torch.zeros(B, T, H, BC, device=DEVICE, dtype=torch.float32)
        chunk_kda_fwd_intra_token_parallel(
            q=q, k=k, gk=g_cumsum, beta=beta, Aqk=Aqk_npu, Akk=Akk_npu,
            scale=scale, cu_seqlens=None, chunk_size=CHUNK_SIZE, sub_chunk_size=BC,
        )
        torch.npu.synchronize()
        err_aqk = _rmse(Aqk_npu, Aqk_cpu)
        err_akk = _rmse(Akk_npu, Akk_cpu)
        assert err_aqk < 0.01, f"[{cfg['id']}] Aqk RMSE={err_aqk:.6f} > 0.01"
        assert err_akk < 0.01, f"[{cfg['id']}] Akk RMSE={err_akk:.6f} > 0.01"


# ═══════════════════════════════════════════════════════════════════════════
# Test: recompute_w_u_fwd
# ═══════════════════════════════════════════════════════════════════════════

class TestRecomputeWUKernel:
    """recompute_w_u_fwd: w/u/kg recompute from Akk_inv."""

    @pytest.mark.parametrize("cfg", TEST_CONFIGS, ids=[c["id"] for c in TEST_CONFIGS])
    def test_precision(self, cfg):
        K, V = cfg["K"], cfg["V"]
        scale = K ** -0.5
        q, k, v, raw_gate, A_log, dt_bias, beta, _, _ = _make_inputs(cfg)
        g_cumsum = kda_gate_chunk_cumsum(
            raw_gate, A_log=A_log, chunk_size=CHUNK_SIZE, scale=RCP_LN2, dt_bias=dt_bias,
        )
        torch.npu.synchronize()
        # Run full intra (non-fused) to get w/u/kg from NPU and Akk_inv
        w_npu, u_npu, _, kg_npu, _, Akk_inv = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g_cumsum, beta=beta, scale=scale,
            cu_seqlens=None, chunk_size=CHUNK_SIZE,
            fuse_diagonal=False, fuse_recompute=False,
        )
        torch.npu.synchronize()
        # CPU reference using NPU-produced Akk_inv
        w_cpu, u_cpu, kg_cpu = _cpu_recompute_w_u(
            k.cpu(), v.cpu(), beta.cpu(), Akk_inv.cpu(), g_cumsum.cpu(),
        )
        err_w = _rmse(w_npu, w_cpu)
        err_u = _rmse(u_npu, u_cpu)
        err_kg = _rmse(kg_npu, kg_cpu)
        assert err_w < 0.01, f"[{cfg['id']}] w RMSE={err_w:.6f} > 0.01"
        assert err_u < 0.01, f"[{cfg['id']}] u RMSE={err_u:.6f} > 0.01"
        assert err_kg < 0.01, f"[{cfg['id']}] kg RMSE={err_kg:.6f} > 0.01"


# ═══════════════════════════════════════════════════════════════════════════
# Test: chunk_gated_delta_rule_fwd_h
# ═══════════════════════════════════════════════════════════════════════════

class TestDeltaRuleKernel:
    """chunk_gated_delta_rule_fwd_h: Delta Rule hidden state update."""

    @pytest.mark.parametrize("cfg", TEST_CONFIGS, ids=[c["id"] for c in TEST_CONFIGS])
    def test_precision(self, cfg):
        K = cfg["K"]
        scale = K ** -0.5
        q, k, v, raw_gate, A_log, dt_bias, beta, initial_state, indices = _make_inputs(cfg)
        g_cumsum = kda_gate_chunk_cumsum(
            raw_gate, A_log=A_log, chunk_size=CHUNK_SIZE, scale=RCP_LN2, dt_bias=dt_bias,
        )
        torch.npu.synchronize()
        w, u, _, kg, _, _ = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g_cumsum, beta=beta, scale=scale,
            cu_seqlens=None, chunk_size=CHUNK_SIZE,
        )
        torch.npu.synchronize()
        # NPU
        h_npu, v_new_npu = chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=g_cumsum,
            initial_state=initial_state.clone(), initial_state_indices=indices,
            cu_seqlens=None, use_exp2=True,
        )
        torch.npu.synchronize()
        # CPU
        h_cpu, v_new_cpu = _cpu_delta_rule_h(
            kg.cpu(), w.cpu(), u.cpu(), g_cumsum.cpu(), initial_state.cpu(),
        )
        err_h = _rmse(h_npu, h_cpu)
        err_v = _rmse(v_new_npu, v_new_cpu)
        assert err_h < 0.01, f"[{cfg['id']}] h RMSE={err_h:.6f} > 0.01"
        assert err_v < 0.01, f"[{cfg['id']}] v_new RMSE={err_v:.6f} > 0.01"


# ═══════════════════════════════════════════════════════════════════════════
# Test: chunk_gla_fwd_o_gk
# ═══════════════════════════════════════════════════════════════════════════

class TestGLAOutputKernel:
    """chunk_gla_fwd_o_gk: final output computation."""

    @pytest.mark.parametrize("cfg", TEST_CONFIGS, ids=[c["id"] for c in TEST_CONFIGS])
    def test_precision(self, cfg):
        K = cfg["K"]
        scale = K ** -0.5
        q, k, v, raw_gate, A_log, dt_bias, beta, initial_state, indices = _make_inputs(cfg)
        g_cumsum = kda_gate_chunk_cumsum(
            raw_gate, A_log=A_log, chunk_size=CHUNK_SIZE, scale=RCP_LN2, dt_bias=dt_bias,
        )
        torch.npu.synchronize()
        w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g_cumsum, beta=beta, scale=scale,
            cu_seqlens=None, chunk_size=CHUNK_SIZE,
        )
        torch.npu.synchronize()
        h_npu, v_new_npu = chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=g_cumsum,
            initial_state=initial_state.clone(), initial_state_indices=indices,
            cu_seqlens=None, use_exp2=True,
        )
        torch.npu.synchronize()
        # NPU
        o_npu = chunk_gla_fwd_o_gk(
            q=q, v=v_new_npu, g=g_cumsum, A=Aqk, h=h_npu, o=v, scale=scale,
        )
        torch.npu.synchronize()
        # CPU
        o_cpu = _cpu_gla_output(
            q.cpu(), v_new_npu.cpu(), g_cumsum.cpu(), Aqk.cpu(), h_npu.cpu(), scale,
        )
        err = _rmse(o_npu, o_cpu)
        assert err < 0.01, f"[{cfg['id']}] RMSE={err:.6f} > 0.01"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Integration test: chunk_kda() runs without NaN/Inf and produces
    finite output with reasonable magnitude, across all configs."""

    @pytest.mark.parametrize("cfg", TEST_CONFIGS, ids=[c["id"] for c in TEST_CONFIGS])
    def test_precision(self, cfg):
        K = cfg["K"]
        scale = K ** -0.5
        q, k, v, raw_gate, A_log, dt_bias, beta, initial_state, indices = _make_inputs(cfg)

        actual_state = initial_state.clone()
        actual_o = chunk_kda(
            q=q.clone(), k=k.clone(), v=v.clone(), g=raw_gate.clone(),
            beta=beta.clone(), scale=scale, initial_state=actual_state,
            initial_state_indices=indices,
            cu_seqlens=None, A_log=A_log, dt_bias=dt_bias,
        )
        torch.npu.synchronize()

        assert not actual_o.isnan().any(), f"[{cfg['id']}] output has NaN"
        assert not actual_o.isinf().any(), f"[{cfg['id']}] output has Inf"
        assert actual_o.float().abs().max() < 50.0, \
            f"[{cfg['id']}] max_abs={actual_o.float().abs().max().item():.4f} > 50.0"


# ═══════════════════════════════════════════════════════════════════════════
# Test: All kernels unified (single config, comprehensive report)
# ═══════════════════════════════════════════════════════════════════════════

class TestAllKernels:
    """Run all kernels sequentially with the default config, report timing + RMSE."""

    def test_all_kernels(self):
        cfg = TEST_CONFIGS[0]  # tiny_default
        B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
        scale = K ** -0.5
        q, k, v, raw_gate, A_log, dt_bias, beta, initial_state, indices = _make_inputs(cfg)
        results = {}

        # ── Gate Cumsum ──
        t0 = time.time()
        g_cumsum = kda_gate_chunk_cumsum(
            raw_gate, A_log=A_log, chunk_size=CHUNK_SIZE, scale=RCP_LN2, dt_bias=dt_bias,
        )
        torch.npu.synchronize()
        results["GateCumsum"] = time.time() - t0
        g_cpu = _cpu_gate_cumsum(raw_gate.cpu(), A_log.cpu(), dt_bias.cpu())
        results["GateCumsum_rmse"] = _rmse(g_cumsum, g_cpu)

        # ── TokenParallel ──
        t0 = time.time()
        Aqk_tp = torch.zeros(B, T, H, CHUNK_SIZE, device=DEVICE, dtype=DTYPE)
        Akk_tp = torch.zeros(B, T, H, BC, device=DEVICE, dtype=torch.float32)
        chunk_kda_fwd_intra_token_parallel(
            q=q, k=k, gk=g_cumsum, beta=beta, Aqk=Aqk_tp, Akk=Akk_tp,
            scale=scale, cu_seqlens=None, chunk_size=CHUNK_SIZE, sub_chunk_size=BC,
        )
        torch.npu.synchronize()
        results["TokenParallel"] = time.time() - t0
        Aqk_cpu, Akk_cpu = _cpu_token_parallel(q.cpu(), k.cpu(), g_cumsum.cpu(), beta.cpu(), scale)
        results["TokenParallel_Aqk_rmse"] = _rmse(Aqk_tp, Aqk_cpu)
        results["TokenParallel_Akk_rmse"] = _rmse(Akk_tp, Akk_cpu)

        # ── Full Intra (includes inter_solve + recompute_w_u) ──
        t0 = time.time()
        w, u, _, kg, Aqk_all, Akk_inv = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g_cumsum, beta=beta, scale=scale,
            cu_seqlens=None, chunk_size=CHUNK_SIZE,
        )
        torch.npu.synchronize()
        results["FullIntra"] = time.time() - t0

        # ── RecomputeWU (CPU ref using NPU Akk_inv) ──
        w_cpu, u_cpu, kg_cpu = _cpu_recompute_w_u(
            k.cpu(), v.cpu(), beta.cpu(), Akk_inv.cpu(), g_cumsum.cpu(),
        )
        results["RecomputeWU_w_rmse"] = _rmse(w, w_cpu)
        results["RecomputeWU_u_rmse"] = _rmse(u, u_cpu)
        results["RecomputeWU_kg_rmse"] = _rmse(kg, kg_cpu)

        # ── DeltaRule ──
        t0 = time.time()
        h_npu, v_new_npu = chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=g_cumsum,
            initial_state=initial_state.clone(), initial_state_indices=indices,
            cu_seqlens=None, use_exp2=True,
        )
        torch.npu.synchronize()
        results["DeltaRule"] = time.time() - t0
        h_cpu, v_new_cpu = _cpu_delta_rule_h(
            kg.cpu(), w.cpu(), u.cpu(), g_cumsum.cpu(), initial_state.cpu(),
        )
        results["DeltaRule_h_rmse"] = _rmse(h_npu, h_cpu)
        results["DeltaRule_v_new_rmse"] = _rmse(v_new_npu, v_new_cpu)

        # ── GLAOutput ──
        t0 = time.time()
        o_npu = chunk_gla_fwd_o_gk(
            q=q, v=v_new_npu, g=g_cumsum, A=Aqk_all, h=h_npu, o=v, scale=scale,
        )
        torch.npu.synchronize()
        results["GLAOutput"] = time.time() - t0
        o_cpu = _cpu_gla_output(
            q.cpu(), v_new_npu.cpu(), g_cumsum.cpu(), Aqk_all.cpu(), h_npu.cpu(), scale,
        )
        results["GLAOutput_o_rmse"] = _rmse(o_npu, o_cpu)

        # ── Full pipeline (integration) ──
        actual_state = initial_state.clone()
        t0 = time.time()
        actual_o = chunk_kda(
            q=q.clone(), k=k.clone(), v=v.clone(), g=raw_gate.clone(),
            beta=beta.clone(), scale=scale, initial_state=actual_state,
            initial_state_indices=indices,
            cu_seqlens=None, A_log=A_log, dt_bias=dt_bias,
        )
        torch.npu.synchronize()
        results["FullPipeline"] = time.time() - t0
        results["FullPipeline_ok"] = float(
            not actual_o.isnan().any() and not actual_o.isinf().any()
            and actual_o.float().abs().max() < 50.0
        )

        # ── Summary ──
        print("\n" + "=" * 70)
        print(f"{'Kernel':<30} {'Time':>10} {'RMSE':>12}")
        print("-" * 70)
        for key in ["GateCumsum", "TokenParallel", "FullIntra", "DeltaRule", "GLAOutput", "FullPipeline"]:
            t = results.get(key, 0)
            if key == "FullPipeline":
                ok = "OK" if results.get("FullPipeline_ok", 0) == 1.0 else "FAIL"
                print(f"{key:<30} {t:>8.2f}s        {ok}")
                continue
            rmse_keys = [k for k in results if k.startswith(key + "_") and "rmse" in k]
            for rk in rmse_keys[:1]:
                label = rk[len(key)+1:]
                print(f"{key:<30} {t:>8.2f}s {results[rk]:>12.6f}  ({label})")
            for rk in rmse_keys[1:]:
                label = rk[len(key)+1:]
                print(f"{'':<30} {'':>10} {results[rk]:>12.6f}  ({label})")
        print("=" * 70)

        # Assertions
        for key, val in results.items():
            if "rmse" in key:
                assert val < 0.01, f"{key}={val:.6f} > 0.01"
        assert results.get("FullPipeline_ok", 0) == 1.0, "FullPipeline has NaN/Inf or large magnitude"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])