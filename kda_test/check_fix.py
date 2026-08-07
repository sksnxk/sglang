#!/usr/bin/env python3
"""Check if the flat store fix is applied to the actual kernel."""
import sys, os
sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")
exec(open("/tmp/load_kda.py").read())
import importlib
cdh = sys.modules["sglang.kernels.ops.attention.fla.chunk_delta_h"]
importlib.reload(cdh)
from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
import inspect
src = inspect.getsource(chunk_gated_delta_rule_fwd_h._kernel.fn)
if "flat pointer" in src:
    print("FIX APPLIED")
    idx = src.index("Store h via")
    print(src[idx:idx+300])
else:
    print("FIX NOT FOUND - still using block_ptr")
    # Find the h store section
    if "p_h1 = tl.make_block_ptr" in src:
        print("Found block_ptr h store")
    else:
        print("No block_ptr h store either?")