#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""逐 case 解析 msprof op_summary: torch_npu 算子图 vs triton kernel 加速比(按程序顺序拆分)。

这是 ``analyze_profile.py`` 的逐 case 版本。用 op_summary 输出中每条 kernel 的
**程序顺序**(行序, 与提交顺序一致, 同 Stream 内按序)把整段 trace 划分为 15 个 case 组
(每组 = 该 case 的 torch_npu 算子拼接 + 其 triton kernel 调用, 由于 run.py 对每个实现
各 ``warmup+repeats+1`` 次, 每个 case 恰好 8 组 triton kernel), 从而对每个 case 给出:

  * torch_npu 单算子拼接总时长(该 case 的 torch 元算子 kernel 的 Task Duration 之和)
  * triton kernel 总时长
  * 加速比 = torch_sum / triton_sum

对比 ``run.py`` 里用 ``torch.npu.synchronize()`` 包住的两者 wall-clock 时间, 这里给出的是
**msprof 记录的内核执行时间**(不含 python/GE 调度开销), 说明拼接开销主要来自多次 kernel
启动与中间张量读写, 而非 python 层。

用法:
    python3 per_case_profile.py [--latest-dir /path/to/msprof] [--csv op_summary.csv]
                                [--triton-calls N] [--case xx]

参数:
    --triton-calls N  每个 case 的 triton kernel 调用次数(默认 8 = run.py warmup=2,repeats=5)
    --case xx         只输出单个 case(如 tiny_default), 默认输出全部
"""
import argparse
import csv
import os

# run.py 里对每个 case: (ref 不算) torch_npu 与 triton 都先 warmup 后 repeats+1 次。
# recompute_w_u_triton 每次调用在 op_summary 里出现一次, 故每个 case 的 triton kernel
# 调用次数 = 1(warmup) + repeats + 1. 脚本用 TRITON_PER_CASE 界定 case 边界。
TRITON_PER_CASE = 8

CASE_IDS = [
    "tiny_default", "tiny_partial_chunk", "tiny_single_head", "tiny_H3", "tiny_T65",
    "tiny_T96", "tiny_T1", "tiny_T2", "tiny_T100", "tiny_T127", "big_B2_H3_K128",
    "big_single_T193", "k32_basic", "k128t127", "tiny_T2562",
]
TRITON_PREFIX = "_recompute_w_u_kernel"


def _latest_op_summary(out_dir):
    hits = []
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if fn.startswith("op_summary_") and fn.endswith(".csv"):
                hits.append(os.path.join(root, fn))
    if not hits:
        raise SystemExit(f"[!] 在 {out_dir!r} 下未找到 op_summary_*.csv")
    return max(hits, key=os.path.getmtime)


def _read_rows(csv_path: str):
    with open(csv_path, encoding="utf-8") as f:
        header = next(csv.reader(f))
    try:
        i_name = header.index("Op Name")
        i_dur = header.index("Task Duration(us)")
    except ValueError:
        raise SystemExit("[!] op_summary 缺少 Op Name / Task Duration(us) 列")
    rows = []
    for line in csv.reader(open(csv_path, encoding="utf-8")):
        if len(line) <= max(i_name, i_dur):
            continue
        name = line[i_name].strip()
        if not name or name == "Op Name":
            continue
        try:
            d = float(line[i_dur].replace(",", "").strip())
        except ValueError:
            continue
        if d > 0:
            rows.append((name, d))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv")
    p.add_argument("--latest-dir", default="prof")
    p.add_argument("--triton-calls", type=int, default=TRITON_PER_CASE,
                   help="每个 case 的 triton kernel 调用次数(默认 run.py 的 warmup=2,repeats=5 → 8)")
    p.add_argument("--case", help="只输出单个 case(如 tiny_default), 默认输出全部")
    a = p.parse_args(argv)

    csv_path = a.csv or _latest_op_summary(a.latest_dir)
    print(f"# op_summary: {os.path.basename(csv_path)}  (每个 case 按 triton 调用 x{a.triton_calls} 分组)\n")

    # 行序 = 提交顺序; 按 triton kernel 出现次数切分出 15 个 case 段。
    # 每个段内: 名字以 TRITON_PREFIX 开头的算 triton; 其余全为 torch_npu 拼接。
    cur_torch = cur_tri = 0.0
    seen_tri = 0
    cases = []
    for name, d in _read_rows(csv_path):
        if name.startswith(TRITON_PREFIX):
            cur_tri += d
            seen_tri += 1
            if seen_tri % a.triton_calls == 0:
                cases.append((cur_torch, cur_tri))
                cur_torch = cur_tri = 0.0
        else:
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
        total_torch += tor
        total_tri += tri
        sp = tor / tri if tri > 0 else float("nan")
        print(f"{cid:>22} {tor:>12.2f} {tri:>11.2f} {sp:>7.2f}x")
    print("-" * 90)
    print(f"({a.case or '所有 case'}) torch_npu 拼接总时长 {total_torch:12.2f} us, "
          f"triton 总时长 {total_tri:12.2f} us, 加速比 {total_torch/total_tri:5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
