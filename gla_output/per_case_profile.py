#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""逐 case 解析 msprof op_summary: torch_npu 算子图 vs triton kernel 加速比(按程序顺序拆分)。

这是 ``analyze_profile.py`` 的逐 case 版本。用 op_summary 输出中每条 kernel 的
**程序顺序**(行序, 与提交顺序一致, 同 Stream 内按序)把整段 trace 划分为 N 个 case 组
(每组 = 该 case 的 torch_npu 算子拼接 + 其 triton kernel 调用, 由于 run.py 对每个实现
各 ``warmup+repeats+1`` 次, 每个 case 恰好 ``2*(warmup+repeats+1)`` 组 triton kernel)，
从而对每个 case 给出:

  * torch_npu 单算子拼接总时长(该 case 的 torch 元算子 kernel 的 Task Duration 之和)
  * triton kernel 总时长
  * 加速比 = torch_sum / triton_sum

对比 ``run.py`` 里用 ``torch.npu.synchronize()`` 包住的两者 wall-clock 时间, 这里给出的是
**msprof 记录的内核执行时间**(不含 python/GE 调度开销), 说明拼接开销主要来自多次 kernel
启动与中间张量读写, 而非 python 层。

用法:
    python3 per_case_profile.py [--latest-dir /path/to/msprof] [--csv op_summary.csv]
                                [--scope {all,k6}] [--case CASE]

参数:
    --scope {all,k6}  all=包含上游 K1-K5 时间 (与 run.py 的 wall-clock 口径不同,
                      但反映端到端设备时间); k6=只统计 gla_output_torch 发出的
                      aclnn* 算子 + K6 triton kernel (与 run.py wall-clock 口径一致)
    --case CASE       只输出单个 case (如 tiny_default), 默认输出全部
"""
import argparse
import csv
import os

# run.py 里对每个 case: torch_npu 与 triton 都先 warmup 后 repeats+1 次。
# gla_output 每次调用在 op_summary 里出现一次, 故每个 case 的 triton kernel
# 调用次数 = 2 * (warmup + repeats + 1)。脚本用 TRITON_PER_CASE 界定 case 边界。
# 默认 warmup=2, repeats=5 → 2*(2+5+1) = 16。
TRITON_PER_CASE = 16

CASE_IDS = [
    "tiny_default", "tiny_partial_chunk", "tiny_single_head", "tiny_H3", "tiny_T65",
    "tiny_T96", "tiny_T1", "tiny_T2", "tiny_T100", "tiny_T127",
]
TRITON_PREFIX = "chunk_gla_fwd_kernel_o"

# gla_output_torch (torch_npu 算子图) 实际发出的算子名模式。
# --scope=k6 时只统计这些 + K6 triton kernel。
K6_TORCH_PREFIXES = (
    "aclnnAdd_",
    "aclnnCat_",
    "aclnnExp2_",
    "aclnnInplaceZero_",
    "aclnnInplaceOne_",
    "aclnnMatmul_",
    "aclnnMul_",
    "aclnnMuls_",
    "aclnnTril_",
    "aclnnArange_",
)

# 上游 K1-K5 算子名模式。--scope=all 时也算入 torch_npu 拼接时间。
K15_PREFIXES = (
    "kda_gate_chunk_cumsum",
    "chunk_kda_fwd_kernel",
    "recompute_w_u_fwd",
    "chunk_gated_delta_rule",
)


def _latest_op_summary(out_dir):
    hits = []
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if fn.startswith("op_summary_") and fn.endswith(".csv"):
                hits.append(os.path.join(root, fn))
    if not hits:
        raise SystemExit(f"[!] 在 {out_dir!r} 下未找到 op_summary_*.csv")
    return max(hits, key=os.path.getmtime)


def _is_k6_torch(name):
    return any(name.startswith(p) for p in K6_TORCH_PREFIXES)


def _is_k15(name):
    return any(name.startswith(p) for p in K15_PREFIXES)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv")
    p.add_argument("--latest-dir", default="prof")
    p.add_argument("--triton-calls", type=int, default=TRITON_PER_CASE,
                   help="每个 case 的 triton kernel 调用次数(默认 run.py 的 warmup=2,repeats=5 → 16)")
    p.add_argument("--scope", choices=["all", "k6"], default="k6",
                   help="all=包含上游 K1-K5 时间; k6=只统计 gla_output_torch 的 aclnn* 算子 + K6 triton")
    p.add_argument("--case", help="只输出单个 case(如 tiny_default), 默认输出全部")
    a = p.parse_args(argv)

    csv_path = a.csv or _latest_op_summary(a.latest_dir)
    print(f"# op_summary: {os.path.basename(csv_path)}  "
          f"(每个 case 按 triton 调用 x{a.triton_calls} 分组, scope={a.scope})\n")

    # 行序 = 提交顺序; 按 triton kernel 出现次数切分出 N 个 case 段。
    # 每个段内: 名字以 TRITON_PREFIX 开头的算 triton; 其余按 scope 决定是否计入。
    cases = []
    cur_torch = cur_tri = 0.0
    seen_tri = 0
    for row in csv.reader(open(csv_path, encoding="utf-8")):
        if len(row) < 10:
            continue
        name = row[4].strip()
        if not name or name == "Op Name":
            continue
        try:
            d = float(row[9].strip())
        except ValueError:
            continue
        if name.startswith(TRITON_PREFIX):
            cur_tri += d
            seen_tri += 1
            if seen_tri % a.triton_calls == 0:
                cases.append((cur_torch, cur_tri))
                cur_torch = cur_tri = 0.0
        else:
            if a.scope == "k6":
                if _is_k6_torch(name):
                    cur_torch += d
                # 上游 K1-K5 kernel 在 k6 scope 下不计入
            else:  # all
                cur_torch += d

    n = len(cases)
    if n != len(CASE_IDS):
        print(f"[!] 预期 {len(CASE_IDS)} 个 case 组, 实际切出 {n} 组; "
              f"可能 --repeats/--warmup 与 {a.triton_calls} 不一致。仍按组序展示。",
              file=os.sys.stderr)
        ids = [f"group#{i}" for i in range(n)]
    else:
        ids = CASE_IDS

    print(f"{'case':>22} {'torch_sum_us':>12} {'tri_sum_us':>11} {'speedup':>8}")
    total_torch = total_tri = 0.0
    for cid, (tor, tri) in zip(ids, cases):
        if a.case and cid != a.case:
            continue
        t = tor
        total_torch += t
        total_tri += tri
        sp = t / tri if tri > 0 else float("nan")
        print(f"{cid:>22} {t:>12.2f} {tri:>11.2f} {sp:>7.2f}x")
    print("-" * 90)
    print(f"({a.case or '所有 case'}) torch_npu 拼接总时长 {total_torch:12.2f} us, "
          f"triton 总时长 {total_tri:12.2f} us, 加速比 {total_torch/total_tri:5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
