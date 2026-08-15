#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-5 (Delta Rule H) 独立测试目录的 CSV 驱动脚本。

用法:
    python3 gen_csv.py                 # 生成 testcases.csv(默认)
    python3 gen_csv.py --out data.csv   # 指定输出路径

生成 testcases.csv 中的每个用例都包含:
    k (kg)                 [B,T,H,K]   衰减后的 key (kg = k*beta*exp2(gk_last-gk))
    w                      [B,T,H,K]   衰减后的 w (由 Kernel-4 输出)
    u                      [B,T,H,V]   原始 value (= Aqk @ (v*beta))
    gk                     [B,T,H,K]   per-channel gate (log2 空间, 已 cumsum+scale)
    initial_state          [N,H,V,K]   初始状态 (N=B, 由 initial_state_indices 索引)
    initial_state_indices  [B] int32   指向 initial_state 的索引 (恒为 arange(B))

行数/规模: 15 个用例沿用 gate_chunk_cumsum / inter_solve 的边界覆盖思路,
从 T=1 到 T=2562, 覆盖 B=1..2, H=1..3。

重要约束: **只测 K=V=64** —— 上游 chunk_delta_h kernel 的 flat 1D store
(`tl.arange(0, BV*64)`) 假定行步长 K=64, 在 K!=64 (如 K=32/128) 时布局错乱
会产生错误结果。这与上游 `test_level2_kernel_precision.py` 的约定一致
(该文件注释: "K must be 64 (chunk_delta_h kernel only works correctly at
K=64 on triton-ascend)")。本目录的 torch_npu / CPU 参考实现数学上支持任意 K,
但 triton kernel 受此限制, 故 CSV 中所有 case 都用 K=64。
T 非 BT(=64) 倍数 / 尾 chunk 不满的部分块由 T 变化覆盖。

注意: Kernel-5 数学上不使用 scale (scale 在 Kernel-2 token_parallel 已用),
所以 gen_csv.py 不写 scale 字段。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from util import csvb64  # noqa: E402

_BT = 64


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


def _seed() -> None:
    torch.manual_seed(20260730)


def _mk_inputs(B, T, H, K):
    """q/k L2-归一化(模长=1, 避免 Akkd 对角块量级过大);
    g = randn*0.5 - 2.0 (log2 空间, 量级 0.13~7);
    w/u 量级 ~0.1; initial_state ~0.05.

    与 test_level2_kernel_precision._make_inputs 的分布一致 (但 K==V)。
    """
    V = K  # 本目录只测 K==V 用例 (上游 kernel 要求 K==V)
    q = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1)
    # w / u 量级与上游一致 (Kernel-4 输出)
    w = torch.randn(B, T, H, K, dtype=torch.float32) * 0.1
    u = torch.randn(B, T, H, V, dtype=torch.float32) * 0.1
    # gk: log2 空间, 已 chunk-local cumsum + RCP_LN2 缩放
    gk = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5 - 2.0
    # initial_state: N=B, 每个 batch 一份独立状态 (indices=arange(B))
    initial_state = torch.randn(B, H, V, K, dtype=torch.float32) * 0.05
    initial_state_indices = torch.arange(B, dtype=torch.int32)
    return {
        "k": k,
        "w": w,
        "u": u,
        "gk": gk,
        "initial_state": initial_state,
        "initial_state_indices": initial_state_indices,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "testcases.csv"))
    a = p.parse_args(argv)

    _seed()
    db = {}
    # (id, B, T, H, K, desc)
    cases = [
        ("tiny_default",       1, 128,  2, 64,  "baseline: 完整 2 chunks"),
        ("tiny_partial_chunk", 1, 63,   2, 64,  "尾 chunk 不满 BT=64"),
        ("tiny_single_head",   1, 128,  1, 64,  "单 head"),
        ("tiny_H3",            1, 128,  3, 64,  "奇数 head"),
        ("tiny_T65",           1, 65,   2, 64,  "T=65: 第二 chunk 只有 1 个 token"),
        ("tiny_T96",           1, 96,   2, 64,  "T=96 非 2 的幂"),
        ("tiny_T1",            1, 1,    2, 64,  "单 token"),
        ("tiny_T2",            1, 2,    2, 64,  "两个 token"),
        ("tiny_T100",          1, 100,  2, 64,  "T=100: 2 chunks(尾空 28)"),
        ("tiny_T127",          1, 127,  2, 64,  "T=127: 第二 chunk 缺 1"),
        ("big_B2_H3",          2, 100,  3, 64,  "多 batch/head, K=64"),
        ("big_B2_T193",        2, 193,  2, 64,  "多 batch + T 非 BT 倍数"),
        ("big_T256",           1, 256,  2, 64,  "T=256 (4 full chunks)"),
        ("tiny_T255",          1, 255,  2, 64,  "T=255: 第二对 chunk 缺 1"),
        ("tiny_T2562",         1, 2562, 1, 64,  "超长 T(40 chunks), 单 head"),
    ]
    for cid, B, T, H, K, desc in cases:
        db[cid] = _mk_inputs(B, T, H, K)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(csvb64.to_text(db, title="KDA delta_rule_h test cases"))
    print(f"wrote {len(db)} cases -> {a.out}")


if __name__ == "__main__":
    main()
