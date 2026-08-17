#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-3 (Inter-Solve Fused: 非对角 Aqk/Akk + 对角块前向替换 + 链式求逆)
精度 & 性能对比测试。

    python3 run.py [CSV] [--repeats N] [--warmup N] [--max-diff LIMIT] [--no-ref]

流程（与本目录 README/DESIGN 一致）:
  1. 读 CSV 中每个用例 (q, k, g, beta, Akkd, scale);
  2. 先跑 **torch_npu 元算子** ``inter_solve_torch`` —— 精度 & 性能**基准**;
  3. 再跑 **triton kernel** ``inter_solve_triton`` —— 被测对象;
  4. 精度: 两个实现各自与 CPU 参考 ``inter_solve_ref`` 的 max-diff
     (``--no-ref`` 时只比 torch vs triton), 并输出 ``speedup = t_torch / t_triton``;
  5. 汇总表格 + 退出码 (任一 case 超限则非 0)。

env 要求: 运行前必须已 source CANN set_env.sh, 并设置
  export LD_LIBRARY_PATH=<torch>/lib:<torch_npu>/lib:$LD_LIBRARY_PATH
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
(见 run_cpu.sh / README.md)。

msprof 级性能采集:
  msprof --output=./prof python3 run.py [--repeats 5] [--warmup 2]
  采集后用 analyze_profile.py / per_case_profile.py 解析（见 README.md）
  注意: msprof 采集时建议 --repeats ~5 --warmup ~2, 产出的 op_summary 里
  每个 triton kernel 调用次数 = warmup + repeats + 1 (per_case_profile 默认按此分组)。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from src.inter_solve_kernel import (  # noqa: E402
    _BT,
    _BC,
    inter_solve_ref,
    inter_solve_torch,
    inter_solve_triton,
)
from util import csvb64  # noqa: E402

_DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testcases.csv")
_WARMUP = 5
_REPEATS = 30
_MAX_DIFF = 1e-2


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
    p = argparse.ArgumentParser(description="inter-solve: torch_npu vs triton")
    p.add_argument("csv", nargs="?", default=_DEFAULT_CSV)
    p.add_argument("--repeats", type=int, default=_REPEATS)
    p.add_argument("--warmup", type=int, default=_WARMUP)
    p.add_argument("--max-diff", type=float, default=_MAX_DIFF, help="精度上限(默认1e-2)")
    p.add_argument("--no-ref", action="store_true", help="跳过 CPU 参考精度校验(只比 torch 与 triton)")
    a = p.parse_args(argv)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[!] 未检测到可用 NPU — 本脚本的两个对比方(torch_npu 与 triton)都需要 NPU。")
        return 2

    cases = csvb64.load_testcases(a.csv)
    results = []
    for cid, tensors in cases.items():
        q = tensors["q"].to("npu")
        k = tensors["k"].to("npu")
        g = tensors["g"].to("npu")
        beta = tensors["beta"].to("npu")
        Akkd = tensors["Akkd"].to("npu")
        scale = _arg1(tensors, "scale", 0.1768)
        sync = torch.npu.synchronize

        # ── 1) torch_npu 元算子 (精度&性能基准) ──
        At, Bt = inter_solve_torch(q, k, g, beta, Akkd, scale)
        t_torch = _timeit(
            lambda: inter_solve_torch(q, k, g, beta, Akkd, scale),
            a.warmup, a.repeats, sync,
        )

        # ── 2) triton kernel ──
        Ap, Bp = inter_solve_triton(q, k, g, beta, Akkd, scale)
        t_tri = _timeit(
            lambda: inter_solve_triton(q, k, g, beta, Akkd, scale),
            a.warmup, a.repeats, sync,
        )

        # ── 3) 精度: 两个实现彼此对比, 及(可选)各自 vs CPU 参考 ──
        d_qk = _maxdiff(Ap, At)   # triton Aqk vs torch Aqk
        d_kk = _maxdiff(Bp, Bt)   # triton Akk_inv vs torch Akk_inv
        msg = f"tri_vs_torch Aqk={d_qk:.3e} Akk={d_kk:.3e}"
        d_pass = d_qk < a.max_diff and d_kk < a.max_diff
        if not a.no_ref:
            Ar, Br = inter_solve_ref(
                tensors["q"], tensors["k"], tensors["g"],
                tensors["beta"], tensors["Akkd"], scale,
            )
            d_p_torch = max(_maxdiff(At, Ar), _maxdiff(Bt, Br))
            d_p_tri = max(_maxdiff(Ap, Ar), _maxdiff(Bp, Br))
            d_pass &= d_p_torch < a.max_diff and d_p_tri < a.max_diff
            msg += f" | torch_vs_ref={d_p_torch:.3e} tri_vs_ref={d_p_tri:.3e}"
        passed = d_pass and t_tri > 0

        speedup = (t_torch / t_tri) if t_tri > 0 else float("nan")
        results.append({
            "id": cid, "shape": list(q.shape),
            "t_torch_ms": t_torch * 1e3, "t_tri_ms": t_tri * 1e3,
            "speedup": speedup,
            "d_qk": d_qk, "d_kk": d_kk, "pass": passed, "msg": msg,
        })
        print(f"[{cid:>16}] shape={str(list(q.shape)):<20} "
              f"torch_npu={t_torch*1e3:7.3f}ms triton={t_tri*1e3:7.3f}ms "
              f"speedup={speedup:6.2f}x  max_diff={max(d_qk, d_kk):.3e} "
              f"{'PASS' if passed else 'FAIL'}")

    # ── 汇总 ──────────────────────────────────────────────
    npass = sum(1 for r in results if r["pass"])
    print("\n" + "=" * 100)
    print(f"{'case':>16} {'torch_ms':>9} {'triton_ms':>9} {'speedup':>8}  "
          f"Aqk_diff     Akk_diff      verdict")
    for r in results:
        print(f"{r['id']:>16} {r['t_torch_ms']:>9.3f} {r['t_tri_ms']:>9.3f} "
              f"{r['speedup']:>7.2f}x {r['d_qk']:>11.3e} {r['d_kk']:>12.3e}  "
              f"{'OK' if r['pass'] else 'FAIL'}")
    print("=" * 100)
    print(f"summary: {npass}/{len(results)} PASS")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
