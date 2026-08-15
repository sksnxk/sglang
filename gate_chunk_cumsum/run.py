#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gate+chunk-cumsum 算子的精度 & 性能对比测试: torch_npu 元算子 vs triton kernel。

这是本目录的主测试脚本。它读取 CSV 中定义的**各种测试用例**,对每个用例依次:

1. 运行 **torch_npu 元算子** 版本 ``gate_cumsum_torch``, 作为**精度基本准**和**性能基本准**;
2. 运行 **triton kernel** 版本 ``gate_chunk_cumsum``;
3. 两项对比验证 (均通过才该 case PASS):
   - 精度: ``triton 结果 vs torch_npu 结果`` / ``torch_npu 结果 vs CPU 参考`` 的
     max-diff (参考 ``gate_cumsum_ref`` 为逐 chunk 循环的纯 CPU ground truth);
   - 性能: 预热后各跑 N 次计时, 输出 ``speedup = torch_npu_time / triton_time``;
4. 汇总输出表格 + 退出码 (任一 case 超限则非 0)。

env 要求: 运行前必须已 source CANN set_env.sh, 并设置
  export LD_LIBRARY_PATH=<torch>/lib:<torch_npu>/lib:$LD_LIBRARY_PATH
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
(见 run_cpu.sh / README.md)。

用法:
  python3 run.py [CSV] [--repeats N] [--warmup N] [--max-diff LIMIT] [--no-ref]
  msprof --output=./prof python3 run.py [--repeats N] [--warmup N]
     用 msprof 采集 op_summary 时建议 --repeats ~5 --warmup ~2, 产出的
     op_summary_*.csv 交给 analyze_profile.py 做 单算子拼接 vs triton 加速比统计。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from src.gate_kernel import (  # noqa: E402
    RCP_LN2,
    gate_chunk_cumsum,
    gate_cumsum_ref,
    gate_cumsum_torch,
)
from util import csvb64  # noqa: E402

_DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testcases.csv")
_WARMUP = 5
_REPEATS = 30
maxDiffLimit = 1e-2
_PREC = {
    "max_internal_tensor": 5.0,
    "n": 24.0,  # fp32 逐元素 maxdiff 上限
}

__all__ = ["main"]


def _arg1(tensors, key, default):
    val = tensors.get(key)
    if val is None:
        return default
    return float(val.reshape(-1)[0])


def _timeit(fn, warmup, repeats, sync):
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if sync is not None:
        sync()
    return (time.perf_counter() - t0) / repeats


def _maxdiff(out, ref):
    return (out.float().cpu() - ref.float().cpu()).abs().max().item()


def main(argv=None):
    p = argparse.ArgumentParser(description="gate+chunk-cumsum: torch_npu vs triton")
    p.add_argument("csv", nargs="?", default=_DEFAULT_CSV)
    p.add_argument("--repeats", type=int, default=_REPEATS)
    p.add_argument("--warmup", type=int, default=_WARMUP)
    p.add_argument("--max-diff", type=float, default=maxDiffLimit, help="精度上限(默认1e-2)")
    p.add_argument("--no-ref", action="store_true", help="跳过 CPU 参考精度校验(只比 torch_npu 与 triton)")
    a = p.parse_args(argv)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[!] 未检测到可用 NPU — 本脚本的两个对比方(torch_npu 与 triton)都需要 NPU。")
        return 2

    cases = csvb64.load_testcases(a.csv)
    results = []
    for cid, tensors in cases.items():
        x = tensors["input"].to("npu")
        A_log = tensors["A_log"].to("npu")
        dt = tensors["dt_bias"].to("npu") if tensors.get("dt_bias") is not None else None
        chunk_size = int(_arg1(tensors, "chunk_size", 64))
        scale = _arg1(tensors, "scale", RCP_LN2)

        sync = torch.npu.synchronize

        # ── 1) torch_npu 元算子 (精度&性能基准) ──
        out_torch = gate_cumsum_torch(x, A_log, dt_bias=dt, chunk_size=chunk_size, scale=scale)
        t_torch = _timeit(
            lambda: gate_cumsum_torch(x, A_log, dt_bias=dt, chunk_size=chunk_size, scale=scale),
            a.warmup, a.repeats, sync,
        )

        # ── 2) triton kernel ──
        out_tri = gate_chunk_cumsum(x, A_log, dt_bias=dt, chunk_size=chunk_size, scale=scale)
        t_tri = _timeit(
            lambda: gate_chunk_cumsum(x, A_log, dt_bias=dt, chunk_size=chunk_size, scale=scale),
            a.warmup, a.repeats, sync,
        )

        # ── 3) 精度: triton vs torch_npu, 及 (可选) 两者 vs CPU 参考 ──
        d_tri_vs_torch = _maxdiff(out_tri, out_torch)
        d_pass = d_tri_vs_torch < a.max_diff
        msg = f"tri_vs_torch={d_tri_vs_torch:.3e}"
        if not a.no_ref:
            ref = gate_cumsum_ref(
                tensors["input"], tensors["A_log"],
                dt_bias=tensors.get("dt_bias"), chunk_size=chunk_size, scale=scale,
            )
            d_torch_vs_ref = _maxdiff(out_torch, ref)
            d_tri_vs_ref = _maxdiff(out_tri, ref)
            d_pass &= d_torch_vs_ref < a.max_diff and d_tri_vs_ref < a.max_diff
            msg += f" | torch_vs_ref={d_torch_vs_ref:.3e} tri_vs_ref={d_tri_vs_ref:.3e}"
        passed = d_pass and t_tri > 0

        speedup = (t_torch / t_tri) if t_tri > 0 else float("nan")
        results.append({
            "id": cid, "shape": list(x.shape),
            "t_torch_ms": t_torch * 1e3, "t_tri_ms": t_tri * 1e3,
            "speedup": speedup,
            "d_tri_vs_torch": d_tri_vs_torch, "pass": passed, "msg": msg,
        })
        print(f"[{cid:>16}] shape={str(list(x.shape)):<24} "
              f"torch_npu={t_torch*1e3:7.3f}ms triton={t_tri*1e3:7.3f}ms "
              f"speedup={speedup:6.2f}x  max_diff={d_tri_vs_torch:.3e} "
              f"{'PASS' if passed else 'FAIL'}")

    # ── 汇总 ──────────────────────────────────────────────
    npass = sum(1 for r in results if r["pass"])
    print("\n" + "=" * 90)
    print(f"{'case':>16} {'torch_ms':>9} {'triton_ms':>9} {'speedup':>8}  max_diff      verdict")
    for r in results:
        print(f"{r['id']:>16} {r['t_torch_ms']:>9.3f} {r['t_tri_ms']:>9.3f} "
              f"{r['speedup']:>7.2f}x {r['d_tri_vs_torch']:>10.3e}  "
              f"{'OK' if r['pass'] else 'FAIL'}")
    print("=" * 90)
    print(f"summary: {npass}/{len(results)} PASS")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())