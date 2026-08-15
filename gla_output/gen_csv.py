#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GLA Output (Kernel 6) 独立测试目录的 CSV 驱动脚本。

用法:
    python3 gen_csv.py                     # 生成 testcases.csv(默认)
    python3 gen_csv.py --out data.csv      # 指定输出路径

生成 testcases.csv 中的每个用例都包含 Kernel 6 的全部输入张量:
    q          [B,T,H,K]   bf16  query 向量
    v          [B,T,H,V]   bf16  value 向量(下游会用 delta rule 修正为 v_new)
    raw_gate   [B,T,H,K]   fp32  未经激活的原始门控
    A_log      [H]         fp32  每 head 对数尺度
    dt_bias    [H*K]       fp32  每 head 每通道偏置
    beta       [B,T,H]     bf16  delta rule 门控 beta
    initial_state [B,H,K,V] fp32  初始压缩状态
    scale      标量 (fp32, = 1/sqrt(K))
    chunk_size 标量 (fp32, = 64)

行数/规模: 10 个用例沿用 level2 分级的边界覆盖思路, 从 T=1 到 T=127,
覆盖 H=1..3, K=V=64。所有用例 K=V=64 (chunk_delta_h kernel 要求)。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from util import csvb64  # noqa: E402

CHUNK_SIZE = 64


def _seed() -> None:
    torch.manual_seed(42)


def _make_inputs(B, T, H, K, V, seed_offset=0):
    """与 test_level2_kernel_precision._make_inputs 一致的随机分布。

    在 CPU 上生成 (DEVICE="cpu")，因为 CSV 存储需要 CPU 张量。
    """
    torch.manual_seed(42 + seed_offset)
    q = F.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16) * 0.1
    raw_gate = (torch.randn(B, T, H, K, dtype=torch.float32) * 0.5 - 2.0).to(
        torch.bfloat16
    )
    A_log = torch.randn(H, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H * K, dtype=torch.float32) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.bfloat16).sigmoid()
    initial_state = torch.randn(B, H, K, V, dtype=torch.float32) * 0.05
    return q, k, v, raw_gate, A_log, dt_bias, beta, initial_state


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "testcases.csv"
        ),
    )
    a = p.parse_args(argv)

    _seed()
    db = {}
    # (id, B, T, H, K, V, desc)
    # 与 level2 TEST_CONFIGS 一一对应（K=V=64 固定）
    cases = [
        ("tiny_default",       1, 128, 2, 64, 64, "baseline: 完整 2 chunks"),
        ("tiny_partial_chunk", 1, 63,  2, 64, 64, "尾 chunk 不满 BT=64"),
        ("tiny_single_head",   1, 128, 1, 64, 64, "单 head"),
        ("tiny_H3",            1, 128, 3, 64, 64, "奇数 head"),
        ("tiny_T65",           1, 65,  2, 64, 64, "T=65: 第二 chunk 只有 1 个 token"),
        ("tiny_T96",           1, 96,  2, 64, 64, "T=96 非 2 的幂"),
        ("tiny_T1",            1, 1,   2, 64, 64, "单 token"),
        ("tiny_T2",            1, 2,   2, 64, 64, "两个 token"),
        ("tiny_T100",          1, 100, 2, 64, 64, "T=100: 2 chunks(尾空 28)"),
        ("tiny_T127",          1, 127, 2, 64, 64, "T=127: 第二 chunk 缺 1"),
    ]
    for idx, (cid, B, T, H, K, V, desc) in enumerate(cases):
        q, k, v, raw_gate, A_log, dt_bias, beta, initial_state = _make_inputs(
            B, T, H, K, V, seed_offset=idx
        )
        scale = K ** -0.5
        db[cid] = {
            "q": q,
            "k": k,
            "v": v,
            "raw_gate": raw_gate,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "beta": beta,
            "initial_state": initial_state,
            "scale": torch.tensor([scale], dtype=torch.float32),
            "chunk_size": torch.tensor([CHUNK_SIZE], dtype=torch.float32),
        }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(
            csvb64.to_text(
                db, title="KDA GLA Output (Kernel 6) test cases (K=V=64, BF16)"
            )
        )
    print(f"wrote {len(db)} cases -> {a.out}")


if __name__ == "__main__":
    main()
