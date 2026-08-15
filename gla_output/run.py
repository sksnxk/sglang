#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GLA Output (Kernel 6) 算子的精度 & 性能对比测试:
torch_npu 元算子 vs triton kernel。

这是本目录的主测试脚本。它读取 CSV 中定义的各种测试用例,对每个用例:

1. 在 NPU 上跑 KDA 流水线 Kernel 1-5 (gate_cumsum → intra → delta_rule),
   产出 Kernel 6 的全部输入: g_cumsum, v_new, Aqk, h；
2. 运行 **torch_npu 元算子** 版本 ``gla_output_torch``, 作为精度和性能基准;
3. 运行 **triton kernel** 版本 ``gla_output_kernel``;
4. 两项对比验证 (均通过才该 case PASS):
   - 精度: ``triton vs torch_npu`` / ``torch_npu vs CPU 参考`` / ``triton vs CPU 参考``
     的 max-diff (参考 ``gla_output_ref`` 为逐 chunk 循环的纯 CPU ground truth);
   - 性能: 预热后各跑 N 次计时, 输出 ``speedup = torch_npu_time / triton_time``;
5. 汇总输出表格 + 退出码 (任一 case 超限则非 0)。

env 要求: 运行前必须已 source CANN set_env.sh, 并设置
  export LD_LIBRARY_PATH=<torch>/lib:<torch_npu>/lib:$LD_LIBRARY_PATH
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
(见 run_cpu.sh / README.md)。

用法:
  python3 run.py [CSV] [--repeats N] [--warmup N] [--max-diff LIMIT] [--no-ref]
  msprof --output=./prof python3 run.py [--repeats N] [--warmup N]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

# 顶部仍需 import torch_npu 以触发 "Background device ... is available" 的注册
import torch_npu  # noqa: F401,E402

from src.gla_output_kernel import (  # noqa: E402
    gla_output_kernel,
    gla_output_ref,
    gla_output_torch,
)
from util import csvb64  # noqa: E402

_DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testcases.csv")
_WARMUP = 5
_REPEATS = 30
RCP_LN2 = 1.4426950216293335
maxDiffLimit = 1e-2


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


def _prepare_inputs_npu(tensors):
    """在 NPU 上跑 K1-K5 产出 Kernel 6 的全部输入。

    与 test_level2_kernel_precision.TestGLAOutputKernel.test_precision 的
    流水线一致: kda_gate_chunk_cumsum → chunk_kda_fwd_intra →
    chunk_gated_delta_rule_fwd_h，得到 g_cumsum / w / u / kg / Aqk / h / v_new。
    """
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from sglang.kernels.ops.attention.fla.chunk_intra import chunk_kda_fwd_intra
    from sglang.kernels.ops.attention.fla.kda import kda_gate_chunk_cumsum

    q = tensors["q"].to("npu")
    k = tensors["k"].to("npu")
    v = tensors["v"].to("npu")
    raw_gate = tensors["raw_gate"].to("npu")
    A_log = tensors["A_log"].to("npu")
    dt_bias = tensors["dt_bias"].to("npu")
    beta = tensors["beta"].to("npu")
    initial_state = tensors["initial_state"].to("npu")
    indices = torch.arange(
        tensors["initial_state"].shape[0], dtype=torch.int32, device="npu"
    )
    chunk_size = int(_arg1(tensors, "chunk_size", 64))
    scale = _arg1(tensors, "scale", 1.0 / (tensors["q"].shape[-1] ** 0.5))

    # ── K1: gate + chunk cumsum ──
    g_cumsum = kda_gate_chunk_cumsum(
        raw_gate, A_log=A_log, chunk_size=chunk_size, scale=RCP_LN2, dt_bias=dt_bias,
    )
    torch.npu.synchronize()

    # ── K2-K4: intra (token_parallel + inter_solve + recompute_w_u) ──
    w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g_cumsum, beta=beta, scale=scale,
        cu_seqlens=None, chunk_size=chunk_size,
    )
    torch.npu.synchronize()

    # ── K5: delta rule h ──
    h_npu, v_new_npu = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g_cumsum,
        initial_state=initial_state.clone(), initial_state_indices=indices,
        cu_seqlens=None, use_exp2=True,
    )
    torch.npu.synchronize()

    return {
        "q": q, "v_new": v_new_npu, "g": g_cumsum,
        "Aqk": Aqk, "h": h_npu, "scale": scale, "chunk_size": chunk_size,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="GLA Output: torch_npu vs triton")
    p.add_argument("csv", nargs="?", default=_DEFAULT_CSV)
    p.add_argument("--repeats", type=int, default=_REPEATS)
    p.add_argument("--warmup", type=int, default=_WARMUP)
    p.add_argument("--max-diff", type=float, default=maxDiffLimit, help="精度上限(默认1e-2)")
    p.add_argument("--no-ref", action="store_true",
                   help="跳过 CPU 参考精度校验(只比 torch_npu 与 triton)")
    a = p.parse_args(argv)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[!] 未检测到可用 NPU — 本脚本的两个对比方(torch_npu 与 triton)都需要 NPU。")
        return 2

    cases = csvb64.load_testcases(a.csv)
    results = []
    for cid, tensors in cases.items():
        shape = list(tensors["q"].shape)
        inp = _prepare_inputs_npu(tensors)
        q, v_new, g, Aqk, h = inp["q"], inp["v_new"], inp["g"], inp["Aqk"], inp["h"]
        scale, chunk_size = inp["scale"], inp["chunk_size"]
        sync = torch.npu.synchronize

        # ── 1) torch_npu 元算子 (精度&性能基准) ──
        out_torch = gla_output_torch(
            q, v_new, g, Aqk, h, scale=scale, chunk_size=chunk_size,
        )
        t_torch = _timeit(
            lambda: gla_output_torch(
                q, v_new, g, Aqk, h, scale=scale, chunk_size=chunk_size,
            ),
            a.warmup, a.repeats, sync,
        )

        # ── 2) triton kernel ──
        out_tri = gla_output_kernel(
            q, v_new, g, Aqk, h, scale=scale, chunk_size=chunk_size,
            out_dtype=out_torch.dtype,
        )
        t_tri = _timeit(
            lambda: gla_output_kernel(
                q, v_new, g, Aqk, h, scale=scale, chunk_size=chunk_size,
                out_dtype=out_torch.dtype,
            ),
            a.warmup, a.repeats, sync,
        )

        # ── 3) 精度: triton vs torch_npu, 及 (可选) 两者 vs CPU 参考 ──
        d_tri_vs_torch = _maxdiff(out_tri, out_torch)
        d_pass = d_tri_vs_torch < a.max_diff
        msg = f"tri_vs_torch={d_tri_vs_torch:.3e}"
        if not a.no_ref:
            # CPU 参考使用 NPU 产生的 v_new/h (下游 kernel 的输出)，与 level2 一致
            ref = gla_output_ref(
                q.cpu(), v_new.cpu(), g.cpu(), Aqk.cpu(), h.cpu(),
                scale=scale, chunk_size=chunk_size,
            )
            d_torch_vs_ref = _maxdiff(out_torch, ref)
            d_tri_vs_ref = _maxdiff(out_tri, ref)
            d_pass &= d_torch_vs_ref < a.max_diff and d_tri_vs_ref < a.max_diff
            msg += f" | torch_vs_ref={d_torch_vs_ref:.3e} tri_vs_ref={d_tri_vs_ref:.3e}"
        passed = d_pass and t_tri > 0

        speedup = (t_torch / t_tri) if t_tri > 0 else float("nan")
        results.append({
            "id": cid, "shape": shape,
            "t_torch_ms": t_torch * 1e3, "t_tri_ms": t_tri * 1e3,
            "speedup": speedup,
            "d_tri_vs_torch": d_tri_vs_torch, "pass": passed, "msg": msg,
        })
        print(f"[{cid:>20}] shape={str(shape):<22} "
              f"torch_npu={t_torch*1e3:7.3f}ms triton={t_tri*1e3:7.3f}ms "
              f"speedup={speedup:6.2f}x  max_diff={d_tri_vs_torch:.3e} "
              f"{'PASS' if passed else 'FAIL'}")

    # ── 汇总 ──────────────────────────────────────────────
    npass = sum(1 for r in results if r["pass"])
    print("\n" + "=" * 96)
    print(f"{'case':>20} {'torch_ms':>9} {'triton_ms':>9} {'speedup':>8}  max_diff      verdict")
    for r in results:
        print(f"{r['id']:>20} {r['t_torch_ms']:>9.3f} {r['t_tri_ms']:>9.3f} "
              f"{r['speedup']:>7.2f}x {r['d_tri_vs_torch']:>10.3e}  "
              f"{'OK' if r['pass'] else 'FAIL'}")
    print("=" * 96)
    print(f"summary: {npass}/{len(results)} PASS")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
