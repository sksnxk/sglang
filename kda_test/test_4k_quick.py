#!/usr/bin/env python3
"""Quick test: KDA kernels at 4K without msprof."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")
from sglang.kernels.ops.attention.fla.kda import RCP_LN2, kda_gate_chunk_cumsum
from sglang.kernels.ops.attention.fla.chunk_intra import chunk_kda_fwd_intra
from sglang.kernels.ops.attention.fla.chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel

DEVICE="npu"; DTYPE=torch.bfloat16; CHUNK_SIZE=64; BC=16
B,T,H,K,V=1,4096,2,128,128; SCALE=K**-0.5
torch.manual_seed(42)

q=F.normalize(torch.randn(B,T,H,K,dtype=torch.float32,device=DEVICE),dim=-1).to(DTYPE)
k=F.normalize(torch.randn(B,T,H,K,dtype=torch.float32,device=DEVICE),dim=-1).to(DTYPE)
v=torch.randn(B,T,H,V,dtype=DTYPE,device=DEVICE)*0.1
raw_gate=(torch.randn(B,T,H,K,dtype=torch.float32,device=DEVICE)*0.5-2.0).to(DTYPE)
A_log=torch.randn(H,dtype=torch.float32,device=DEVICE)*0.1
dt_bias=torch.randn(H*K,dtype=torch.float32,device=DEVICE)*0.1
beta=torch.rand(B,T,H,dtype=DTYPE,device=DEVICE).sigmoid()

print("Step A...", flush=True)
g_cumsum=kda_gate_chunk_cumsum(raw_gate,A_log=A_log,chunk_size=CHUNK_SIZE,scale=RCP_LN2,dt_bias=dt_bias)
print("Step A done", flush=True)

print("Step B1 token_parallel...", flush=True)
Aqk_tp=torch.zeros(B,T,H,CHUNK_SIZE,device=DEVICE,dtype=torch.float32)
Akk_tp=torch.zeros(B,T,H,BC,device=DEVICE,dtype=torch.float32)
chunk_kda_fwd_intra_token_parallel(q=q,k=k,gk=g_cumsum,beta=beta,Aqk=Aqk_tp,Akk=Akk_tp,scale=SCALE,cu_seqlens=None,chunk_size=CHUNK_SIZE,sub_chunk_size=BC)
print("Step B1 done", flush=True)

print("Step B full...", flush=True)
w,u,_,kg,Aqk,_=chunk_kda_fwd_intra(q=q,k=k,v=v,gk=g_cumsum,beta=beta,scale=SCALE,cu_seqlens=None,chunk_size=CHUNK_SIZE,fuse_diagonal=False,fuse_recompute=False)
print("Step B done", flush=True)
print("ALL OK")
