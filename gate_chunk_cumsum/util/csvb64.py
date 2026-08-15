#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CSV 数据的编码/解码工具（base64 紧凑格式，专门服务于 gate+chunk-cumsum 测试）。

为什么要 base64 而不是直接写数值?
-------------------------------
* Gate cumsum 的测试输入是 fp32 随机张量（raw gate / A_log / dt_bias）,
  直接用文本浮点数会引入额外的解析歧义（格式化、round-trip 精度、Inf/NaN 表示）。
* base64 是 torch 原生 `.numpy().tobytes()` 的二进制镜像，解码后与原始数据
  **逐 bit 一致**，因此精度对比只反映 kernel 本身,不掺杂 CSV 往返噪声。

本模块提供:
  * ``tensor_to_csv(t, name)``  -> 一段 base64 CSV 文本
  * ``load_testcases(text)``   -> dict[caseid] -> 该 case 的全部张量 (CPU)
  * ``run_csv(db, func, delete_after=True)`` -> 对每个 case 调用 ``func(**tensors)``
    并把返回值按与输入相同的张量去重策略写入同一 CSV（用于把参考结果导出成
    可复现的 "golden" 数据）。

格式约定
--------
* 文本是合法 CSV（RFC 4180 近似）：第一行为表头，后续每行对应一个张量片段。
* 每个张量的元素按行主序展平后 base64 编码，列 ``data_len`` 记录了展平后元素数，
  解码时可精确还原 shape（shape 本身也编码在列里）。
* 所有数值以 fp32 为准（torch dtype float32）。
"""

import base64
import csv
import io
import os

import torch


def tensor_to_csv(t: torch.Tensor, name: str = "t") -> str:
    """把一个 fp32 CPU 张量编码为一段单行 CSV 记录。"""
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


# 数列 dataclass 的 dtype：fp32（与 raw gate / A_log / dt_bias 的存储一致）。
_TORCH_DTYPE = torch.float32


def _from_b64_shape(b64: str, shape_s: str, n: int) -> torch.Tensor:
    """base64+f32 字节流解码为 CPU 张量（shape 由字符串指定）。"""
    shape = tuple(int(x) for x in shape_s.split("/")) if shape_s != "0" else ()
    raw = base64.b64decode(b64)
    arr = torch.frombuffer(bytearray(raw), dtype=_TORCH_DTYPE).reshape(shape)
    assert arr.numel() == n, f"numel mismatch {arr.numel()} != {n}"
    return arr.clone()


def parse_csv(text: str) -> torch.Tensor:
    """把 ``tensor_to_csv`` 生成的单行记录解析回 CPU 张量。"""
    rows = list(csv.reader(io.StringIO(text)))
    row = rows[0]
    tag, shape_s, n_s, _name, b64 = row
    assert tag == "b64", f"unsupported csv tag: {tag!r}"
    return _from_b64_shape(b64, shape_s, int(n_s))


def to_text(db: dict, title: str = "KDA gate+chunk cumsum golden cases") -> str:
    """将 ``{caseid: {name: tensor}}`` 编码为整个 CSV 文档。

    第一行是分隔行（``#`` 开头，不是合法数据）；随后每个 case 以
    ``case_<id>`` 行作为块注释，后面跟若干数据行。
    """
    out = io.StringIO()
    out.write(f"# {title}\n")
    out.write("# 每个 case 内的张量名称: input / A_log / dt_bias / output (参考结果)\n")
    out.write("# case_<id> 行用于分隔不同用例\n")
    for cid, tensors in db.items():
        out.write(f"case_{cid}\n")
        for name, t in tensors.items():
            if t is None:
                continue  # 允许 dt_bias=None:空值不写数据行
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


def run_csv(
    csv_path: str,
    func,
    device: str = "npu",
    delete_after: bool = True,
    print_out: bool = True,
) -> dict:
    """load_testcases -> 对每个 case 调用 func -> 记录 (caseid, shape, max_diff)。

    参数:
        csv_path: 输入 CSV 路径
        func: 签名 ``func(input, A_log, dt_bias=None, chunk_size=64, scale=RCP_LN2)``
             返回一个 [B,T,H,K] fp32 tensor（NPU 或 CPU）
        device: 参考实现放在哪个 device（"npu" 或 "cpu"）
        delete_after: 完成后是否删除加载的 CSV（避免把临时数据留在 repo）
        print_out: 打印每 case 的 max_diff / 是否 PASS

    返回: ``{caseid: {"max_diff": f, "pass": bool, "shape": [...]}}``
    """
    cases = load_testcases(csv_path)
    results = {}
    for cid, tensors in cases.items():
        inp = tensors["input"]
        A_log = tensors["A_log"]
        dt_bias = tensors.get("dt_bias")
        # 参考实现直接在各张量当前 device 上运行（device 保持一致即可）
        dev = inp.device if inp.is_cuda or str(inp.device).startswith("npu") else device
        out = func(
            inp, A_log,
            dt_bias=dt_bias,
            chunk_size=int(tensors.get("chunk_size", 64)),
            scale=tensors.get("scale"),
        )
        import torch  # noqa
        if getattr(torch, "npu", None) and dev.startswith("npu"):
            torch.npu.synchronize()
        # 计算与参考的 max diff
        ref = gate_cumsum_ref(
            inp, A_log, dt_bias=dt_bias,
            chunk_size=int(tensors.get("chunk_size", 64)),
            scale=tensors.get("scale"),
        )
        d = (out.float().cpu() - ref.float().cpu()).abs().max().item()
        passed = d < 1e-2
        results[cid] = {"max_diff": d, "pass": passed, "shape": list(inp.shape)}
        if print_out:
            print(f"[{cid}] shape={list(inp.shape)} max_diff={d:.3e} {'PASS' if passed else 'FAIL'}")
    if delete_after:
        os.remove(csv_path)
    return results


from src.gate_kernel import gate_cumsum_ref  # noqa: E402  (延迟导入，避免循环依赖)

if __name__ == "__main__":
    db = {
        "t1": {"input": torch.randn(1, 128, 2, 64) * 0.5 - 2.0,
               "A_log": torch.randn(2) * 0.1,
               "dt_bias": torch.randn(2 * 64) * 0.1},
    }
    text = to_text(db)
    back = parse_decoded_text(text)
    for cid in db:
        for k in db[cid]:
            assert torch.equal(db[cid][k], back[cid][k]), (cid, k)
    print("csvb64 roundtrip OK")