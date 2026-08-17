#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gate+chunk-cumsum 独立测试目录的 CSV 驱动脚本。

用法:
    python3 gen_csv.py                     # 生成 cases.csv(默认)
    python3 gen_csv.py --out data.csv      # 指定输出路径
    python3 gen_csv.py --no-bias           # 只用标准 gate(不带 dt_bias 的用例)

生成 testcases.csv 中的每个用例都包含:
    input    [B,T,H,K]   raw gate(未激活)
    A_log    [H]
    dt_bias  [H*K]       (bias 用例)
    chunk_size  标量
    scale       标量(恒为 RCP_LN2,即 HAS_SCALE=True)

行数/规模: 10 个用例沿用 level2 分级的边界覆盖思路,从 T=1 到 T=2562,
覆盖 B=1..2,H=1..3,K=32/64/128,以及 T 非 BT 倍数 / K 非 BS 倍数的部分块。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from util import csvb64  # noqa: E402

RCP_LN2 = 1.4426950216293335


def _seed() -> None:
    torch.manual_seed(20260727)


def _mk_inputs(B, T, H, K, with_bias=True, scale=RCP_LN2):
    """与 test_level2_kernel_precision._make_inputs 一致的随机分布。"""
    x = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5 - 2.0  # raw gate 均值在负区
    A_log = torch.randn(H, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H * K, dtype=torch.float32) * 0.1 if with_bias else None
    return {
        "input": x,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "chunk_size": torch.tensor([64], dtype=torch.float32),
        "scale": torch.tensor([scale], dtype=torch.float32),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "testcases.csv"))
    p.add_argument("--no-bias", action="store_true", help="不生成带 dt_bias 的用例")
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
        ("big_B2_H3_K128",     2, 100,  3, 128, "多 batch/head, K 非 32 倍数"),
        ("big_single_T193",    1, 193,  1, 32,  "T 边界 BT 非倍数 + K=32"),
        ("k32_basic",          1, 64,   1, 32,  "K=32 (BS 整数倍)"),
        ("k128t127",           1, 127,  2, 128, "K=128 + 尾 chunk 不满"),
        ("tiny_T2562",         1, 2562, 1, 64,  "超长 T(40 chunks), 单 head"),
    ]
    with_bias = not a.no_bias
    for cid, B, T, H, K, desc in cases:
        db[cid] = _mk_inputs(B, T, H, K, with_bias=with_bias)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(csvb64.to_text(db, title=f"KDA gate+chunk cumsum test cases (with_bias={with_bias})"))
    print(f"wrote {len(db)} cases -> {a.out}")


if __name__ == "__main__":
    main()