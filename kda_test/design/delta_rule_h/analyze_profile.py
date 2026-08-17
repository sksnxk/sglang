#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""解析 msprof 生成的 op_summary_*.csv，对比 torch_npu 单算子拼接 vs triton kernel。

用法:
    python3 analyze_profile.py [--csv op_summary.csv] [--by-triton NAME] [--pad-us F] ...
    python3 analyze_profile.py --latest-dir prof       # 默认：prof 下最新的 PROF_*/.../op_summary_*.csv

背景（见 delta_rule_h/DESIGN.md §6/§7）
-----------------------------------------------
* run.py 里有两个实现：
    - ``delta_rule_h_torch``：torch_npu 元算子拼接——逐 chunk 串行的 matmul /
      exp2 / 广播乘法 + 外积（每个 chunk 多条 kernel，另有 zeros / clone /
      copy 等）。这些 kernel 都算做 ``torch_npu 拼接时间``；
    - ``delta_rule_h_triton``（triton kernel，op 名
      ``_delta_rule_h_kernel``）：单 kernel，每 (batch, head, V-tile) 一个 CTA。
* msprof（--task-time 默认开）会把每个设备 kernel 记成 op_summary 一行，列里有
  ``Task Duration(us)``（AIV 侧时间，含 aicore/aiv）。
* 本脚本聚合每个 op name 的总耗时，triton kernel 也包含在内的**全部内核**求和
  即为 torch_npu 单算子拼接的总时间。triton kernel 的加速比
  = ``sum(torch_npu 全内核) / sum(该 triton kernel 全部调用)``。
"""

import argparse
import csv
import os

# 需要统计进 torch_npu 单算子拼接总时间的内核名模式。
# 简化方案：除 triton kernel 外，所有内核都算“torch_npu 拼接”。
TRITON_NAMES = ("_delta_rule_h_kernel",)


def _find_latest_op_summary(output_dir: str) -> str:
    """在 msprof --output 目录下找到最新的 op_summary_*.csv（跨 PROF_* 子目录）。"""
    hits = []
    for root, _, files in os.walk(output_dir):
        for fn in files:
            if fn.startswith("op_summary_") and fn.endswith(".csv"):
                hits.append(os.path.join(root, fn))
    if not hits:
        raise SystemExit(f"[!] 在 {output_dir!r} 下没有找到 op_summary_*.csv")
    return max(hits, key=os.path.getmtime)


def _read_rows(csv_path: str):
    """读取 op_summary csv 并定位 Op Name / Task Duration(us) / aiv_time 列。

    返回 ``[(name_ok, dur_us, aiv_us), ...]``：
      * ``dur_us``: Task Duration(us)，含调度/等待在内的整体核时间；
      * ``aiv_us``: aiv_time(us)，纯 AIV(向量)侧执行时间。两个口径都保留，
        统计时用 ``--metric`` 选择。
    """
    with open(csv_path, encoding="utf-8") as f:
        header = next(csv.reader(f))

    def _idx(name):
        try:
            return header.index(name)
        except ValueError:
            return None

    i_name, i_dur, i_aiv = _idx("Op Name"), _idx("Task Duration(us)"), _idx("aiv_time(us)")

    rows = []
    for line in csv.reader(open(csv_path, encoding="utf-8")):
        if not line or i_name is None or i_name >= len(line):
            continue
        name = line[i_name].strip()
        if not name or "Duration" in name:
            continue

        def _f(idx, default=0.0):
            if idx is None or not line[idx].strip():
                return default
            try:
                return float(line[idx].replace(",", "").strip())
            except ValueError:
                return default

        dur, aiv = _f(i_dur), _f(i_aiv)
        if dur > 0 or aiv > 0:
            rows.append((name, dur, aiv))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="torch_npu 拼接 vs triton kernel (msprof op_summary)")
    p.add_argument("--csv", help="op_summary csv 路径（默认 --latest-dir 下最新一个）")
    p.add_argument("--latest-dir", default="prof", help="msprof 输出目录（默认 %(default)s）")
    p.add_argument("--triton-name", action="append", default=list(TRITON_NAMES),
                   help="triton kernel 名（默认 %(default)s，可多次给）")
    p.add_argument("--pad-us", type=float, default=2.0,
                   help="每一条 torch_npu kernel 的执行开销（us, 默认 2.0，来自 AIV ~40 cyc）")
    p.add_argument("--no-pad", action="store_true", help="不算 torch_npu kernel 启动/等待开销")
    p.add_argument("--metric", choices=["task", "aiv", "max"], default="task",
                   help="取哪一列作为 kernel 耗时: task=Task Duration(us)（端点到端点）; "
                        "aiv=aiv_time(us)（纯 AIV 执行）; max=两者取大（接近真实并行流水）")
    a = p.parse_args(argv)

    csv_path = a.csv or _find_latest_op_summary(a.latest_dir)
    print(f"# op_summary: {os.path.basename(csv_path)}")
    pad = 0.0 if a.no_pad else a.pad_us

    rows = _read_rows(csv_path)
    if not rows:
        raise SystemExit("[!] op_summary 没有有效内核行")

    def _pick(dur, aiv):
        return {"task": dur, "aiv": aiv, "max": max(dur, aiv)}[a.metric]

    agg = {}
    for name, dur, aiv in rows:
        agg[name] = agg.get(name, 0.0) + _pick(dur, aiv)

    is_tri = lambda name: any(t in name for t in a.triton_name)  # noqa: E731

    n_torch = sum(1 for name, *_ in rows if not is_tri(name))
    # torch_us 单位为 us:  sum_us + n_torch * pad
    torch_us = sum(d for n, d in agg.items() if not is_tri(n)) + n_torch * pad
    tri_us = sum(d for n, d in agg.items() if is_tri(n))
    speedup = torch_us / tri_us if tri_us > 0 else float("nan")

    print(f"metric={a.metric}  (torch_npu 每 kernel +{pad:.1f}us 启动开销)")
    print(f"{'op name':<55}{'calls':>7}{'sum_us':>12}{'type':>10}")
    for name in sorted(agg):
        tag = "triton" if is_tri(name) else "torch"
        print(f"{name:<55}{sum(1 for n, *_ in rows if n == name):>7}{agg[name]:>12.3f}{tag:>10}")
    print("-" * 84)
    print(f"torch_npu 拼接总时间: {torch_us*1e-3:.3f} ms ({n_torch} kernels)")
    print(f"triton kernel 总时间: {tri_us*1e-3:.3f} ms")
    print(f"speedup (torch/triton) = {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
