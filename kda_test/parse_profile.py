#!/usr/bin/env python3
"""Parse msprof op_summary CSV to extract per-kernel timing.

Usage:
  python3 kda_test/parse_profile.py ./prof_kda_128k
"""
import csv
import os
import sys
from collections import defaultdict


def parse_op_summary(prof_dir):
    """Parse op_summary_*.csv files in the profile directory."""
    csv_files = []
    for root, dirs, files in os.walk(prof_dir):
        for f in files:
            if f.startswith("op_summary") and f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        print(f"ERROR: No op_summary_*.csv found in {prof_dir}")
        return None

    print(f"Found {len(csv_files)} op_summary CSV files")

    all_rows = []
    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    return all_rows


def aggregate_by_kernel(rows):
    """Aggregate timing by kernel name."""
    kernel_stats = defaultdict(lambda: {"count": 0, "total_time_us": 0, "min_time_us": float("inf"), "max_time_us": 0})

    for row in rows:
        name = row.get("Task Type", "") or row.get("Name", "") or row.get("Op Name", "")
        if not name:
            continue
        try:
            time_us = float(row.get("Task Time(us)", 0) or row.get("Time(us)", 0) or row.get("Duration(us)", 0))
        except (ValueError, KeyError):
            continue

        stats = kernel_stats[name]
        stats["count"] += 1
        stats["total_time_us"] += time_us
        stats["min_time_us"] = min(stats["min_time_us"], time_us)
        stats["max_time_us"] = max(stats["max_time_us"], time_us)

    return kernel_stats


def group_kda_kernels(kernel_stats):
    """Group kernel stats into KDA pipeline categories."""
    categories = {
        "StepA_gate_cumsum": [],
        "StepB_intra": [],
        "StepC_delta_h": [],
        "StepD_gla_output": [],
        "Other_Ascend": [],
        "Memcpy": [],
    }

    for name, stats in kernel_stats.items():
        name_lower = name.lower()
        if "gate" in name_lower and ("cumsum" in name_lower or "kda_gate" in name_lower):
            categories["StepA_gate_cumsum"].append((name, stats))
        elif "cumsum" in name_lower and "chunk_local" in name_lower:
            categories["StepA_gate_cumsum"].append((name, stats))
        elif "intra" in name_lower or "token_parallel" in name_lower or "inter_solve" in name_lower or "recompute_w_u" in name_lower or "kkt" in name_lower:
            categories["StepB_intra"].append((name, stats))
        elif "delta" in name_lower and "rule" in name_lower or "gated_delta" in name_lower:
            categories["StepC_delta_h"].append((name, stats))
        elif "gla" in name_lower and "output" in name_lower or "chunk_gla_fwd_o" in name_lower:
            categories["StepD_gla_output"].append((name, stats))
        elif "memcpy" in name_lower or "mem" in name_lower:
            categories["Memcpy"].append((name, stats))
        else:
            categories["Other_Ascend"].append((name, stats))

    return categories


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 kda_test/parse_profile.py <prof_dir>")
        sys.exit(1)

    prof_dir = sys.argv[1]
    rows = parse_op_summary(prof_dir)
    if rows is None:
        sys.exit(1)

    print(f"Total rows: {len(rows)}")

    # Print available columns for debugging
    if rows:
        print(f"Available columns: {list(rows[0].keys())}")

    kernel_stats = aggregate_by_kernel(rows)
    categories = group_kda_kernels(kernel_stats)

    print(f"\n{'='*90}")
    print(f"=== KDA Kernel Profile Summary (from msprof op_summary) ===")
    print(f"{'='*90}")

    total_kda_us = 0
    for cat, kernels in categories.items():
        if not kernels:
            continue
        cat_total = sum(s["total_time_us"] for _, s in kernels)
        cat_count = sum(s["count"] for _, s in kernels)
        print(f"\n--- {cat} ({cat_count} calls, {cat_total/1000:.2f}ms total) ---")
        for name, stats in sorted(kernels, key=lambda x: -x[1]["total_time_us"]):
            avg_us = stats["total_time_us"] / stats["count"]
            print(f"  {name:60s} calls={stats['count']:4d}  total={stats['total_time_us']/1000:8.2f}ms  avg={avg_us:8.2f}us  min={stats['min_time_us']:8.2f}us  max={stats['max_time_us']:8.2f}us")
        if cat not in ("Other_Ascend", "Memcpy"):
            total_kda_us += cat_total

    print(f"\n{'='*90}")
    print(f"Total KDA kernel time: {total_kda_us/1000:.2f}ms")

    # Bottleneck analysis
    print(f"\n=== Bottleneck Analysis ===")
    kda_categories = {k: v for k, v in categories.items() if k not in ("Other_Ascend", "Memcpy")}
    if kda_categories:
        cat_totals = {}
        for cat, kernels in kda_categories.items():
            cat_totals[cat] = sum(s["total_time_us"] for _, s in kernels)
        total_kda = sum(cat_totals.values())
        for cat in sorted(cat_totals, key=lambda c: -cat_totals[c]):
            pct = cat_totals[cat] / total_kda * 100 if total_kda > 0 else 0
            print(f"  {cat:40s}: {cat_totals[cat]/1000:8.2f}ms ({pct:5.1f}%)")

        bottleneck = max(cat_totals, key=cat_totals.get)
        print(f"\n  Bottleneck: {bottleneck} ({cat_totals[bottleneck]/1000:.2f}ms, {cat_totals[bottleneck]/total_kda*100:.1f}%)")


if __name__ == "__main__":
    main()