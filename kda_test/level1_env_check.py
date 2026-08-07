#!/usr/bin/env python3
"""
Level 1: KDA 环境验证 + 模块加载
===================================

验证项目:
  1. NPU 设备可用性 (torch_npu)
  2. Triton backend 为 npu
  3. KDA 模块能通过 mock 方式加载 (无需完整 sglang)
  4. 关键函数可用

预期输出: 所有检查项 PASS, 无报错

参数说明:
  无

运行方式:
  docker exec -it triton-ascend-env-zhm bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_npu/lib:$LD_LIBRARY_PATH
  export PYTHONPATH=/docker/zhm/0505_skill_test/sonnet/sglang/python:$PYTHONPATH
  cd /docker/zhm/0505_skill_test/sonnet/sglang
  python3 kda_test/level1_env_check.py
"""

import sys

# ── 1. 检查 NPU 设备 ────────────────────────────────────────────
print("=" * 60)
print("Check 1: NPU Device Availability")
print("=" * 60)
import torch
import torch_npu

npu_available = torch.npu.is_available()
npu_count = torch.npu.device_count() if npu_available else 0
npu_name = torch.npu.get_device_name(0) if npu_available else "N/A"
print(f"  torch:        {torch.__version__}")
print(f"  torch_npu:    {torch_npu.__version__}")
print(f"  NPU available: {npu_available}")
print(f"  NPU count:     {npu_count}")
print(f"  Device name:   {npu_name}")
assert npu_available, "NPU not available!"
assert npu_count >= 1, "No NPU devices found!"
print("  ✅ PASS\n")

# ── 2. 检查 Triton Backend ──────────────────────────────────────
print("=" * 60)
print("Check 2: Triton Backend")
print("=" * 60)
import triton
import triton.language as tl

triton_version = triton.__version__
try:
    backend = triton.runtime.driver.active.get_current_target().backend
except Exception:
    backend = "unknown"
print(f"  triton:   {triton_version}")
print(f"  backend:  {backend}")
assert backend == "npu", f"Expected backend=npu, got {backend}"
print("  ✅ PASS\n")

# ── 3. 加载 KDA 模块 (mock sglang.srt) ──────────────────────────
print("=" * 60)
print("Check 3: KDA Module Loading (mock sglang.srt)")
print("=" * 60)

sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")
exec(open("/tmp/load_kda.py").read())

from sglang.kernels.ops.attention.fla.kda import (
    chunk_kda,
    chunk_kda_fwd,
    kda_gate_chunk_cumsum,
    chunk_gla_fwd_o_gk,
)
from sglang.kernels.ops.attention.fla.chunk_intra import chunk_kda_fwd_intra
from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum

print("  chunk_kda:                    OK")
print("  chunk_kda_fwd:                OK")
print("  kda_gate_chunk_cumsum:        OK")
print("  chunk_gla_fwd_o_gk:           OK")
print("  chunk_kda_fwd_intra:          OK")
print("  chunk_gated_delta_rule_fwd_h: OK")
print("  chunk_local_cumsum:           OK")
print("  ✅ PASS\n")

# ── 4. 简单 Triton kernel 在 NPU 上运行 ─────────────────────────
print("=" * 60)
print("Check 4: Simple Triton Kernel on NPU")
print("=" * 60)

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    i = tl.program_id(0)
    idx = i * 256 + tl.arange(0, 256)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask)
    y = tl.load(y_ptr + idx, mask=mask)
    tl.store(out_ptr + idx, x + y, mask=mask)

N = 1024
x = torch.randn(N, device="npu", dtype=torch.float32)
y = torch.randn(N, device="npu", dtype=torch.float32)
out = torch.empty_like(x)
grid = (triton.cdiv(N, 256),)
add_kernel[grid](x, y, out, N)
torch.npu.synchronize()

expected = x + y
max_diff = (out - expected).abs().max().item()
print(f"  Max diff: {max_diff:.10f}")
assert max_diff < 1e-5, f"Triton kernel incorrect: max_diff={max_diff}"
print("  ✅ PASS\n")

# ── 总结 ────────────────────────────────────────────────────────
print("=" * 60)
print("Level 1: ALL CHECKS PASSED")
print("=" * 60)