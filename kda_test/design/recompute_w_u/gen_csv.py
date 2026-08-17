#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-4 (Recompute W/U) 独立测试目录的 CSV 驱动脚本。

用法:
    python3 gen_csv.py                 # 生成 testcases.csv(默认)
    python3 gen_csv.py --out data.csv   # 指定输出路径

生成 testcases.csv 中的每个用例都包含:
    k      [B,T,H,K]   key
    v      [B,T,H,V]   value
    beta   [B,T,H]     对角项缩放 (1D)
    A      [B,T,H,BT]  Akk_inv (chunk 内 KKT 矩阵的逆, 下三角)
    gk     [B,T,H,K]   gate cumsum (log2 空间, 由 Kernel-1 输出)
    scale  标量         (固定为 K^{-0.5} 量级)

A 生成方式: 内联 inter_solve 的简化数学 (单位下三角 + 随机严格下三角扰动),
模拟 Akk_inv 的数据分布, 保持本目录自包含, 不依赖 inter_solve 目录。

行数/规模: 15 个用例复用 gate_chunk_cumsum / token_parallel / inter_solve
的边界覆盖思路, 从 T=1 到 T=2562, 覆盖 B=1..2, H=1..3, K=32/64/128,
以及 T 非 BT(=64) 倍数 / 尾 chunk 不满的部分块。
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


def _mk_Akk_inv(B, T, H, K, BT=_BT):
    """生成符合 Akk_inv 数据分布的下三角矩阵 [B, T, H, BT]。

    真实的 Akk_inv = (I - strict_tril(Akk))^{-1} @ Akk 的合并下三角逆,
    其特点是: 严格下三角 + 对角线接近 1 (因为是 I 的逆的近似)。

    本函数用 ``I + alpha * strict_tril(randn)`` 近似, 其中 alpha 控制扰动
    幅度 (这里取 0.1 使对角线占主导, 与上游 kernel 真实数据的量级一致)。
    每个 chunk 的 [BT, BT] 块独立生成, chunk 内行=token, 列=chunk 内 j 位置。
    """
    torch.manual_seed(0)  # A 的生成独立于 k/v/beta/gk 的随机流
    NT = _cdiv(T, BT)
    # 尾 chunk 不满 BT 时仍按 BT 生成 (多余的行后面会被裁掉; 越界行对应 beta=0,
    # 不会影响 w/u 的有效行)
    A = torch.zeros(B, NT * BT, H, BT, dtype=torch.float32)
    alpha = 0.1
    for b in range(B):
        for h in range(H):
            for c in range(NT):
                base = c * BT
                # 严格下三角扰动 + 单位对角
                R = torch.randn(BT, BT, dtype=torch.float32) * alpha
                R = torch.tril(R, diagonal=-1) + torch.eye(BT, dtype=torch.float32)
                A[b, base:base + BT, h] = R
    A = A[:, :T].contiguous()
    return A


def _mk_inputs(B, T, H, K, V=None, scale=None):
    """与 test_level2_kernel_precision._make_inputs 一致的随机分布。

    * q/k: L2-归一化 (模长=1);
    * v:   randn * 0.1 (小量级, 避免递推放大);
    * raw_gate: randn * 0.5 - 2.0 (对应 gate = -exp(A_log)*softplus(raw_gate+dt_bias),
      量级 0.13~7; 这里直接用作 gk = log2 空间的 gate cumsum);
    * beta: sigmoid(rand) (对角缩放, 恒正, 0~1);
    * A: 内联生成的 Akk_inv 近似 (单位下三角 + 随机扰动);
    * scale: K^{-0.5}.
    """
    if V is None:
        V = K  # 默认 K == V (与 level2 测试一致)
    if scale is None:
        scale = K ** -0.5
    torch.manual_seed(42)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1,
    )
    v = torch.randn(B, T, H, V, dtype=torch.float32) * 0.1
    raw_gate = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5 - 2.0
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    # gk = gate cumsum 的近似 (log2 空间): 直接用 raw_gate 模拟
    gk = raw_gate.clone()
    # A = Akk_inv (独立随机流, 与 k/v/beta/gk 解耦)
    A = _mk_Akk_inv(B, T, H, K, BT=_BT)
    return {
        "k": k,
        "v": v,
        "beta": beta,
        "A": A,
        "gk": gk,
        "scale": torch.tensor([scale], dtype=torch.float32),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "testcases.csv"))
    a = p.parse_args(argv)

    _seed()
    db = {}
    # (id, B, T, H, K, V, desc)  V 默认 = K
    cases = [
        ("tiny_default",       1, 128,  2, 64,  None, "baseline: 完整 2 chunks"),
        ("tiny_partial_chunk", 1, 63,   2, 64,  None, "尾 chunk 不满 BT=64"),
        ("tiny_single_head",   1, 128,  1, 64,  None, "单 head"),
        ("tiny_H3",            1, 128,  3, 64,  None, "奇数 head"),
        ("tiny_T65",           1, 65,   2, 64,  None, "T=65: 第二 chunk 只有 1 个 token"),
        ("tiny_T96",           1, 96,   2, 64,  None, "T=96 非 2 的幂"),
        ("tiny_T1",            1, 1,    2, 64,  None, "单 token"),
        ("tiny_T2",            1, 2,    2, 64,  None, "两个 token"),
        ("tiny_T100",          1, 100,  2, 64,  None, "T=100: 2 chunks(尾空 28)"),
        ("tiny_T127",          1, 127,  2, 64,  None, "T=127: 第二 chunk 缺 1"),
        ("big_B2_H3_K128",     2, 100,  3, 128, None, "多 batch/head, K 非 32 倍数"),
        ("big_single_T193",    1, 193,  1, 32,  None, "T 边界 BT 非倍数 + K=32"),
        ("k32_basic",          1, 64,   1, 32,  None, "K=32 (BS 整数倍)"),
        ("k128t127",           1, 127,  2, 128, None, "K=128 + 尾 chunk 不满"),
        ("tiny_T2562",         1, 2562, 1, 64,  None, "超长 T(40 chunks), 单 head"),
    ]
    for cid, B, T, H, K, V, desc in cases:
        db[cid] = _mk_inputs(B, T, H, K, V=V)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(csvb64.to_text(db, title="KDA recompute_w_u test cases"))
    print(f"wrote {len(db)} cases -> {a.out}")


if __name__ == "__main__":
    main()
