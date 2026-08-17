#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""逐 (case, kernel) 解析 msprof op_summary: marker 切分 + torch/triton 时长。

配合 bench.py 模式 B（``--msprof``）使用。bench.py 对每个 (case, kernel) 段
在 marker kernel（``_kda_bench_marker``）分界内先跑 ``K_torch`` N 次、再跑
``K_triton`` N 次（warmup 也计入段内）。op_summary 中:

  * ``_kda_bench_marker`` 行是 (case, kernel) 段的分界 —— 相邻两个 marker
    之间的行属于同一段（首个 marker 之前的行归入虚拟 ``setup`` 段，不计入）;
  * 段内 Op Name 以该 kernel 的 triton op 名（见 KERNEL_META）开头的行 =
    triton 时间; 其余（aclnn* 等）= torch_npu 拼接时间;
  * ``profile_meta.json`` 兜底交叉校验段数与 kernel 名映射。

输出 ``results.csv``（与 UNIFIED_BENCH_PLAN.md §4.4 schema 一致）::

    case_id, B, T, H, K, V, kernel, torch_us, triton_us, speedup, max_diff, status

并合并 ``correctness.csv`` 的精度/支持性信息。

用法:
    python3 per_case_profile.py [--latest-dir ./prof] [--correctness correctness.csv]
    python3 per_case_profile.py --csv op_summary_*.csv --profile-meta profile_meta.json
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# 6 kernel 的 triton op 名 + id 映射（与 bench.py::KERNEL_META 一致）
KERNEL_META = [
    {"id": "K1", "name": "gate_chunk_cumsum",  "triton_op": "_gate_cumsum_kernel"},
    {"id": "K2", "name": "token_parallel",     "triton_op": "_token_parallel_kernel"},
    {"id": "K3", "name": "inter_solve",        "triton_op": "_inter_solve_kernel"},
    {"id": "K4", "name": "recompute_w_u",     "triton_op": "_recompute_w_u_kernel"},
    {"id": "K5", "name": "delta_rule_h",      "triton_op": "_delta_rule_h_kernel"},
    {"id": "K6", "name": "gla_output",        "triton_op": "chunk_gla_fwd_kernel_o"},
]
MARKER_NAME = "_kda_bench_marker"
TRITON_OPS = {km["triton_op"]: km["id"] for km in KERNEL_META}


def _find_latest_op_summary(output_dir: str) -> str:
    """在 msprof --output 目录下找到最新的 op_summary_*.csv。"""
    hits = []
    for root, _, files in os.walk(output_dir):
        for fn in files:
            if fn.startswith("op_summary_") and fn.endswith(".csv"):
                hits.append(os.path.join(root, fn))
    if not hits:
        raise SystemExit(f"[!] 在 {output_dir!r} 下没有找到 op_summary_*.csv")
    return max(hits, key=os.path.getmtime)


def _find_all_op_summary(output_dir: str):
    """返回 op_summary_*.csv 路径列表（按文件名排序，slice_0 在前）。"""
    hits = []
    for root, _, files in os.walk(output_dir):
        for fn in files:
            if fn.startswith("op_summary_") and fn.endswith(".csv"):
                hits.append(os.path.join(root, fn))
    return sorted(hits)


def _read_op_summary(csv_path: str):
    """读 op_summary csv，返回 [(op_name, duration_us), ...] 按行序。

    Task Duration(us) 列作 kernel 耗时（端点到端点，含调度/等待）。
    支持多 slice 文件: 若 csv_path 是目录或含通配符，合并所有匹配文件。
    """
    import glob as _glob
    if os.path.isdir(csv_path):
        paths = sorted(_glob.glob(os.path.join(csv_path, "**", "op_summary_*.csv"), recursive=True))
    elif "*" in csv_path or "?" in csv_path:
        paths = sorted(_glob.glob(csv_path))
    else:
        paths = [csv_path]
    if not paths:
        raise SystemExit(f"[!] 没有找到 op_summary 文件: {csv_path}")

    all_rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue
        i_name = header.index("Op Name")
        i_dur = header.index("Task Duration(us)")
        for line in csv.reader(open(p, encoding="utf-8")):
            if not line or len(line) <= max(i_name, i_dur):
                continue
            name = line[i_name].strip()
            if not name or name == "Op Name":
                continue
            try:
                dur = float(line[i_dur].replace(",", "").strip())
            except ValueError:
                continue
            if dur > 0 or name == MARKER_NAME:
                all_rows.append((name, dur))
    return all_rows


def _segment_by_markers(rows, profile_meta):
    """按 _kda_bench_marker 行切分 op_summary 行序为 (case, kernel) 段。

    返回 list[dict]: 每段 {case_id, kernel_id, torch_us, triton_us, triton_calls}。
    段顺序对应 profile_meta（bench.py 写入顺序）: marker 成对出现，
    相邻两个 marker 之间的行 = 一个 (case, kernel) 段。
    首个 marker 之前的行归入虚拟 setup 段（不计入）。

    profile_meta 中 skipped=True 的段没有 marker（bench.py 跳过时不发 marker），
    所以实际段数 = 非 skipped 段数。这里用 profile_meta 的非 skipped 子集
    来标注段。
    """
    # 收集所有 marker 行的索引
    marker_idx = [i for i, (n, _) in enumerate(rows) if n == MARKER_NAME]
    if len(marker_idx) < 2:
        raise SystemExit(
            f"[!] 只找到 {len(marker_idx)} 个 marker 行，无法分段。"
            "请确认 bench.py --msprof 模式已正确运行。"
        )

    # marker 成对: [0,1)=seg0, [2,3)=seg1, ...; 相邻 marker 之间也可能有内容
    # 但 bench.py 逻辑是: marker(开始) -> kernel 调用 -> marker(结束)
    # 所以 segment[i] = rows[marker_idx[2i]+1 : marker_idx[2i+1]]
    n_seg = len(marker_idx) // 2

    # profile_meta 中非 skipped 的段（这些才有实际 marker）
    if profile_meta:
        non_skipped = [pm for pm in profile_meta if not pm.get("skipped", False)]
    else:
        non_skipped = []

    segments = []
    for i in range(n_seg):
        start = marker_idx[2 * i] + 1
        end = marker_idx[2 * i + 1]
        seg_rows = rows[start:end]
        # 段内分类: triton op 行 vs torch_npu 拼接行
        torch_us = 0.0
        triton_us = 0.0
        triton_calls = 0
        triton_op_seen = None
        for name, dur in seg_rows:
            # 匹配某个 kernel 的 triton op 名（前缀匹配，因 op_summary 可能
            # 带后缀或完整路径）
            matched = None
            for top, kid in TRITON_OPS.items():
                if name == top or name.startswith(top):
                    matched = kid
                    triton_op_seen = top
                    break
            if matched is not None:
                triton_us += dur
                triton_calls += 1
            else:
                # 排除其他 marker 或非算子行
                if name == MARKER_NAME:
                    continue
                torch_us += dur
        # 从 non_skipped profile_meta 取 case_id / kernel_id（兜底: 用段序号）
        if i < len(non_skipped):
            pm = non_skipped[i]
            case_id = pm.get("case_id", f"seg{i}")
            kernel_id = pm.get("kernel", f"K{i+1}")
        else:
            case_id = f"seg{i}"
            kernel_id = f"K{i+1}"
        segments.append({
            "case_id": case_id,
            "kernel_id": kernel_id,
            "torch_us": torch_us,
            "triton_us": triton_us,
            "triton_calls": triton_calls,
            "skipped": False,  # 实际有 marker 的段都是非 skipped 的
            "triton_op_seen": triton_op_seen,
        })
    return segments


def _load_correctness(csv_path: str):
    """读 correctness.csv，返回 {(case_id, kernel_id): (max_diff, status)}。"""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    out = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["case_id"], row["kernel"])
            out[key] = (row["max_diff"], row["status"])
    return out


def _load_cases_meta(meta_path: str):
    """读 cases_meta.json，返回 {case_id: {B,T,H,K,V,...}}。"""
    if not meta_path or not os.path.exists(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="逐 (case, kernel) 解析 msprof: marker 切分 + torch/triton 时长")
    p.add_argument("--csv", help="op_summary csv 路径（默认 --latest-dir 下最新）")
    p.add_argument("--latest-dir", default=os.path.join(HERE, "prof"),
                   help="msprof 输出目录（默认 %(default)s）")
    p.add_argument("--profile-meta", default=os.path.join(HERE, "profile_meta.json"),
                   help="bench.py --msprof 写出的段元数据（默认 %(default)s）")
    p.add_argument("--correctness", default=os.path.join(HERE, "correctness.csv"),
                   help="correctness.csv 路径（默认 %(default)s）")
    p.add_argument("--cases-meta", default=os.path.join(HERE, "cases_meta.json"),
                   help="cases_meta.json 路径（默认 %(default)s）")
    p.add_argument("--out", default=os.path.join(HERE, "results.csv"),
                   help="输出 results.csv 路径（默认 %(default)s）")
    a = p.parse_args(argv)

    # 若有 --csv 用之；否则在 --latest-dir 下合并所有 op_summary slice
    if a.csv:
        csv_path = a.csv
    else:
        all_hits = _find_all_op_summary(a.latest_dir)
        if not all_hits:
            raise SystemExit(f"[!] 在 {a.latest_dir!r} 下没有找到 op_summary_*.csv")
        csv_path = a.latest_dir  # _read_op_summary 会展开为目录、合并所有 slice
    print(f"# op_summary: {len(all_hits) if a.csv is None and a.latest_dir else 1} file(s)")

    # 读 profile_meta.json（段元数据，bench.py 写出）
    profile_meta = []
    if a.profile_meta and os.path.exists(a.profile_meta):
        with open(a.profile_meta, encoding="utf-8") as f:
            profile_meta = json.load(f)
        print(f"# profile_meta: {len(profile_meta)} 段")

    rows = _read_op_summary(csv_path)
    print(f"# op_summary 行数: {len(rows)} (含 marker)")

    segments = _segment_by_markers(rows, profile_meta)

    correctness = _load_correctness(a.correctness)
    cases_meta = _load_cases_meta(a.cases_meta)

    # 写 results.csv
    # 同时输出 skipped 段（来自 profile_meta 中 skipped=True 的条目）以保证
    # results.csv 覆盖所有 (case, kernel) 组合（与 correctness.csv 对齐）。
    out_rows = []
    n_seg = 0
    n_skip = 0

    # 构建已解析段的索引集合（这些段有实际 marker + op_summary 数据）
    parsed_keys = {(s["case_id"], s["kernel_id"]) for s in segments}
    for s in segments:
        cid = s["case_id"]
        kid = s["kernel_id"]
        cm = cases_meta.get(cid, {})
        max_diff, status = correctness.get((cid, kid), ("", ""))
        torch_us = f"{s['torch_us']:.3f}"
        triton_us = f"{s['triton_us']:.3f}"
        if s["triton_us"] > 0:
            speedup = f"{s['torch_us'] / s['triton_us']:.3f}"
        else:
            speedup = "-"
        n_seg += 1
        out_rows.append({
            "case_id": cid,
            "B": cm.get("B", ""),
            "T": cm.get("T", ""),
            "H": cm.get("H", ""),
            "K": cm.get("K", ""),
            "V": cm.get("V", ""),
            "kernel": kid,
            "torch_us": torch_us,
            "triton_us": triton_us,
            "speedup": speedup,
            "max_diff": max_diff,
            "status": status or "OK",
        })

    # 补上 profile_meta 中 skipped=True 的段（无 op_summary 数据，标 "不支持"）
    if profile_meta:
        for pm in profile_meta:
            if pm.get("skipped", False):
                cid = pm.get("case_id", "")
                kid = pm.get("kernel", "")
                if (cid, kid) in parsed_keys:
                    continue  # 已在上面输出
                cm = cases_meta.get(cid, {})
                max_diff, status = correctness.get((cid, kid), ("", ""))
                out_rows.append({
                    "case_id": cid,
                    "B": cm.get("B", ""),
                    "T": cm.get("T", ""),
                    "H": cm.get("H", ""),
                    "K": cm.get("K", ""),
                    "V": cm.get("V", ""),
                    "kernel": kid,
                    "torch_us": "-",
                    "triton_us": "-",
                    "speedup": "-",
                    "max_diff": max_diff,
                    "status": status or "不支持",
                })
                n_skip += 1

    with open(a.out, "w", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case_id", "B", "T", "H", "K", "V", "kernel",
            "torch_us", "triton_us", "speedup", "max_diff", "status"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nresults.csv -> {a.out}")
    print(f"有效段: {n_seg}  跳过(不支持): {n_skip}  总段: {n_seg + n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
