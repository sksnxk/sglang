#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CSV 编码/解码工具（base64 紧凑格式，服务于 inter-solve 测试）。

为何用 base64 而不是直接写数值?
------------------------------
inter-solve 的输入是 fp32 随机张量（q / k / g / beta / Akkd）与标量 scale，
纯文本浮点数会有格式化 / round-trip 精度 / Inf / NaN 的歧义。base64 是
torch 原生 ``.numpy().tobytes()`` 的二进制镜像，解码与原始数据逐 bit 一致，
精度对比只反映 kernel 本身的差异，不掺杂 CSV 往返噪声。

本模块提供:
  * ``tensor_to_csv(t, name)``    -> 一段 base64 CSV 文本
  * ``to_text(db, title)``        -> {caseid: {name: tensor}} -> 整个 CSV 文档
  * ``load_testcases(path)``      -> CSV 文件 -> {caseid: {name: tensor}} (CPU)

与 token_parallel/util/csvb64.py 同构, 去掉 gate 专用的 ``run_csv``。
"""

import base64
import csv
import io

import torch

_TORCH_DTYPE = torch.float32  # 存储 dtype：fp32


def tensor_to_csv(t: torch.Tensor, name: str = "t") -> str:
    """把一个 fp32 CPU 张量编码为一段单行 CSV 记录。

    列为:  tag=b64, shape(以 ``/`` 连接), numel, name, base64 字节。
    标量（shape()）编码为 ``0``。
    """
    t = t.detach().cpu().to(torch.float32).contiguous()
    buf = io.BytesIO()
    buf.write(t.numpy().tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(
        [
            "b64",
            f"{'/'.join(str(d) for d in t.shape) if len(t.shape) else '0'}",
            str(t.numel()),
            name,
            b64,
        ]
    )
    return out.getvalue()


def _from_b64_shape(b64: str, shape_s: str, n: int) -> torch.Tensor:
    """base64 + fp32 字节流解码为 CPU 张量（shape 由字符串指定）。"""
    shape = tuple(int(x) for x in shape_s.split("/")) if shape_s != "0" else ()
    raw = base64.b64decode(b64)
    arr = torch.frombuffer(bytearray(raw), dtype=_TORCH_DTYPE).reshape(shape)
    assert arr.numel() == n, f"numel mismatch {arr.numel()} != {n}"
    return arr.clone()


def to_text(db: dict, title: str = "KDA inter-solve test cases") -> str:
    """将 ``{caseid: {name: tensor}}`` 编码为整个 CSV 文档。

    第一行是分隔注释行（``#``），随后每个 case 以 ``case_<id>`` 行作为块
    头，后面跟若干数据行（每个张量一行）。
    """
    out = io.StringIO()
    out.write(f"# {title}\n")
    out.write("# 每个 case 内的张量名称: q / k / g / beta / Akkd / scale\n")
    out.write("# case_<id> 行用于分隔不同用例\n")
    for cid, tensors in db.items():
        out.write(f"case_{cid}\n")
        for name, t in tensors.items():
            if t is None:
                continue  # 空值不写数据行
            out.write(tensor_to_csv(t, name=name))
    return out.getvalue()


def parse_decoded_text(text: str) -> dict:
    """把 ``to_text`` 生成的文档解析回 ``{caseid: {name: tensor}}``。"""
    csv.field_size_limit(1 << 30)  # base64 行长, 放宽 csv 默认字段上限 (131072)
    db = {}
    cur = None
    for line in io.StringIO(text):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("case_"):
            cur = line[len("case_"):]
            db.setdefault(cur, {})
            continue
        rows = list(csv.reader(io.StringIO(line)))
        row = rows[0]
        tag, shape_s, n_s, name, b64 = row
        assert tag == "b64"
        arr = _from_b64_shape(b64, shape_s, int(n_s))
        if cur is None:
            raise ValueError("data row before any case header")
        db[cur][name] = arr
    return db


def load_testcases(csv_path: str) -> dict:
    """从磁盘加载 CSV 文件 (utf-8) 并返回 {caseid: {name: tensor}}。"""
    with open(csv_path, encoding="utf-8") as f:
        return parse_decoded_text(f.read())


if __name__ == "__main__":
    db = {
        "t1": {
            "q": torch.randn(1, 128, 2, 64),
            "k": torch.randn(1, 128, 2, 64),
            "g": torch.randn(1, 128, 2, 64),
            "beta": torch.randn(1, 128, 2),
            "Akkd": torch.randn(1, 128, 2, 16),
            "scale": torch.tensor([0.1768]),
        },
    }
    text = to_text(db)
    back = parse_decoded_text(text)
    for cid in db:
        for k in db[cid]:
            assert torch.equal(db[cid][k], back[cid][k]), (cid, k)
    print("csvb64 roundtrip OK")
