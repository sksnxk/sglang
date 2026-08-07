#!/usr/bin/env python3
"""Final test: verify the actual fixed kernel works E2E."""
import sys, os, shutil
# Clear cache BEFORE any imports
os.system("rm -rf ~/.triton/cache")
sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")
exec(open("/tmp/load_kda.py").read())
import torch, torch.nn.functional as F
from sglang.kernels.ops.attention.fla.kda import kda_gate_chunk_cumsum, RCP_LN2
from sglang.kernels.ops.attention.fla.chunk_intra import chunk_kda_fwd_intra
from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

DEVICE="npu"; DTYPE=torch.bfloat16; BT=64
B,H,K,V=1,2,64,64; T2=128
torch.manual_seed(42)
q2=F.normalize(torch.randn(B,T2,H,K,dtype=torch.float32,device=DEVICE),dim=-1).to(DTYPE)
k2=F.normalize(torch.randn(B,T2,H,K,dtype=torch.float32,device=DEVICE),dim=-1).to(DTYPE)
v2_t=torch.randn(B,T2,H,V,dtype=DTYPE,device=DEVICE)*0.1
raw_gate2=(torch.randn(B,T2,H,K,dtype=torch.float32,device=DEVICE)*0.5-2.0).to(DTYPE)
A_log2=torch.randn(H,dtype=torch.float32,device=DEVICE)*0.1
dt_bias2=torch.randn(H*K,dtype=torch.float32,device=DEVICE)*0.1
beta2=torch.rand(B,T2,H,dtype=DTYPE,device=DEVICE).sigmoid()
initial_state2=torch.randn(B,H,K,V,dtype=torch.float32,device=DEVICE)*0.05
g_cumsum2=kda_gate_chunk_cumsum(raw_gate2,A_log=A_log2,chunk_size=BT,scale=RCP_LN2,dt_bias=dt_bias2)
torch.npu.synchronize()
w2,u2,_,kg2,_,_=chunk_kda_fwd_intra(q=q2,k=k2,v=v2_t,gk=g_cumsum2,beta=beta2,scale=K**-0.5,cu_seqlens=None,chunk_size=BT)

rmse = lambda a,b:(a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8)
wf2=w2.float().cpu(); uf2=u2.float().cpu()
kgf2=kg2.float().cpu(); gkf2=g_cumsum2.float().cpu(); h0f2=initial_state2.float().cpu()

# CPU ref
v_cpu = torch.zeros(B, T2, H, V)
h_cpu = torch.zeros(B, 2, H, V, K)
for b in range(B):
    for h_idx in range(H):
        s = h0f2[b, h_idx].clone()
        for c in range(2):
            tc = c * BT; te = min(T2, tc + BT)
            wc = wf2[b, tc:te, h_idx]; uc = uf2[b, tc:te, h_idx]
            kc = kgf2[b, tc:te, h_idx]; gl = gkf2[b, te-1, h_idx]
            h_cpu[b, c, h_idx] = s.clone()
            v_c = uc - wc @ s.T
            v_cpu[b, tc:te, h_idx] = v_c
            s = s * torch.exp2(gl)[None, :] + (v_c.T @ kc)

# Run actual fixed kernel
indices = torch.arange(B, dtype=torch.int32, device=DEVICE)
h_act, v_act = chunk_gated_delta_rule_fwd_h(
    k=kg2, w=w2, u=u2, gk=g_cumsum2,
    initial_state=initial_state2.clone(),
    initial_state_indices=indices,
    cu_seqlens=None, use_exp2=True)
torch.npu.synchronize()

print("=== ACTUAL FIXED KERNEL ===")
print(f"  h RMSE:   {rmse(h_act.float().cpu(), h_cpu):.6f}")
print(f"  v_new RMSE: {rmse(v_act.float().cpu(), v_cpu):.6f}")
print(f"  h[0,0,0,:4] = {h_act[0,0,0,0,:4].float().cpu().tolist()}")
print(f"  CPU          = {h_cpu[0,0,0,0,:4].tolist()}")
print(f"  h[0,1,0,:4] = {h_act[0,1,0,0,:4].float().cpu().tolist()}")
print(f"  CPU          = {h_cpu[0,1,0,0,:4].tolist()}")
print(f"  v[0,0,0,:8] = {v_act[0,0,0,:8].float().cpu().tolist()}")
print(f"  CPU          = {v_cpu[0,0,0,:8].tolist()}")
print(f"  v[0,64,0,:8] = {v_act[0,64,0,:8].float().cpu().tolist()}")
print(f"  CPU          = {v_cpu[0,64,0,:8].tolist()}")

if rmse(v_act.float().cpu(), v_cpu) < 0.01:
    print("\n*** SUCCESS! Actual kernel is fixed! ***")
else:
    print(f"\n*** FAILED! v_new RMSE={rmse(v_act.float().cpu(), v_cpu):.6f} ***")