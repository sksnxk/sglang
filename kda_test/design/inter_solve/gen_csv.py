#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-3 (Inter-Solve Fused) 独立测试目录的 CSV 驱动脚本。

用法:
    python3 gen_csv.py                 # 生成 testcases.csv(默认)
    python3 gen_csv.py --out data.csv   # 指定输出路径

生成 testcases.csv 中的每个用例都包含:
    q      [B,T,H,K]   query
    k      [B,T,H,K]   key
    g      [B,T,H,K]   gate (log2 空间, 由 Kernel-1 输出)
    beta   [B,T,H]     对角项缩放 (1D)
    Akkd   [B,T,H,BC]  对角线 Akk 块 (由 Kernel-2 输出, 严格下三角)
    scale  标量 (固定为 baseline 的 0.1768 量级)

Akkd 生成方式: 内联 token_parallel 数学 (逐 token 循环, 严格下三角
gated dot product), 保持 inter_solve 目录自包含, 不依赖 token_parallel。

行数/规模: 15 个用例复用 gate_chunk_cumsum / token_parallel 的边界覆盖思路,
从 T=1 到 T=2562, 覆盖 B=1..2, H=1..3, K=32/64/128, 以及 T 非 BT(=64) 倍数 /
尾 chunk 不满的部分块。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from util import csvb64  # noqa: E402

_BT = 64
_BC = 16


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


def _seed() -> None:
    torch.manual_seed(20260728)


def _mk_inputs(B, T, H, K, scale=0.1768):
    """q/k L2-归一化(模长=1, 避免 Akkd 对角块量级过大导致前向替换爆炸);
    g = randn*0.5 - 2.0 (log2 空间, 对应 gate = -exp(A_log)*softplus(raw_gate) 的
    真实分布, 量级为 0.13~7);  beta = sigmoid(rand) (对角缩放, 恒正, 0~1).

    Akkd = Kernel-2 输出的对角线 Akk 块 [B,T,H,BC], 严格下三角
    (j < i 同 sub-chunk 内 <k[i]*beta[i], k[j]*exp2(g[i]-g[j])>).
    """
    q = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1)
    g = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5 - 2.0
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    # 内联计算 Akkd (token_parallel 数学, 逐 token 循环)
    Akkd = _compute_akkd(q, k, g, beta)
    return {
        "q": q,
        "k": k,
        "g": g,
        "beta": beta,
        "Akkd": Akkd,
        "scale": torch.tensor([scale], dtype=torch.float32),
    }


def _compute_akkd(q, k, g, beta, chunk_size=_BT, sub_chunk_size=_BC):
    """计算对角线 Akk 块 [B,T,H,BC], 与 Kernel-2 (token_parallel) 数学一致.

    Akk[i, j] = <k[i]*beta[i], k[j] * exp2(g[i]-g[j])>   (j < i, 同 sub-chunk)
    列 = j - i_ts (sub-chunk 内相对偏移), 对角线为 0.
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    qf, kf, gf, bf = q.float(), k.float(), g.float(), beta.float()
    Akkd = torch.zeros(B, T, H, BC, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                i_c, i_s = t // BT, (t % BT) // BC
                i_ts = i_c * BT + i_s * BC
                kt = kf[b, t, h] * bf[b, t, h]
                gt = gf[b, t, h]
                for j in range(i_ts, min(t, min(T, i_ts + BC))):
                    kj = kf[b, j, h]
                    gj = gf[b, j, h]
                    kgj = kj * torch.exp2(gt - gj)
                    Akkd[b, t, h, j - i_ts] = float((kt * kgj).sum())
    return Akkd


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
        ("big_B2_H3_K128",     2, 100,  3, 128, "多 batch/head, K 非 32 倍数"),
        ("big_single_T193",    1, 193,  1, 32,  "T 边界 BT 非倍数 + K=32"),
        ("k32_basic",          1, 64,   1, 32,  "K=32 (BS 整数倍)"),
        ("k128t127",           1, 127,  2, 128, "K=128 + 尾 chunk 不满"),
        ("tiny_T2562",         1, 2562, 1, 64,  "超长 T(40 chunks), 单 head"),
    ]
    for cid, B, T, H, K, desc in cases:
        db[cid] = _mk_inputs(B, T, H, K)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(csvb64.to_text(db, title="KDA inter-solve test cases"))
    print(f"wrote {len(db)} cases -> {a.out}")


if __name__ == "__main__":
    main()
