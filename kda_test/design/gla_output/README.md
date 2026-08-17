# gla_output 独立验证目录

KDA 第 6 个核函数（GLA Output: 跨块 + 块内组合输出）的独立、可上传验证目录。
不依赖 sglang 的 Kernel 6 本身, 只依赖 torch + triton（NPU / triton-ascend）。
但需要 sglang 的上游 K1-K5 来产生 Kernel 6 的输入。

## 目录结构

```
gla_output/
├── DESIGN.md            设计文档（数学、分核、数据流、精度/性能对比策略）
├── gen_csv.py           生成 testcases.csv（10 个用例的输入张量，base64 编码）
├── testcases.csv        （生成物）用例输入数据
├── run.py               主测试脚本：读 CSV → NPU 上跑 K1-K5 产 K6 输入 →
│                        torch_npu 元算子和 triton kernel 分别跑精度与性能，
│                        输出加速比（见下）
├── analyze_profile.py   （性能）解析 msprof 的 op_summary_*.csv，对比
│                        torch_npu 单算子拼接总时长 vs triton kernel 时长
├── per_case_profile.py  逐 case 加速比（按程序顺序分组）
├── run_cpu.sh           容器内运行脚本：source CANN + LD_LIBRARY_PATH +
│                        PYTHONPATH + AUTOLOAD=0
├── src/
│   └── gla_output_kernel.py   三种实现：CPU 参考 gla_output_ref、
│                        torch_npu 元算子 gla_output_torch、
│                        triton kernel gla_output_kernel
└── util/
    └── csvb64.py        base64 CSV 编解码（保证张量逐 bit 往返一致）
```

## run.py：精度 + 加速比测试

读 CSV 中每行用例, 对同一份数据依次:

| 步骤 | 实现 | 用途 |
|------|------|------|
| 1 | NPU 上跑 `kda_gate_chunk_cumsum` + `chunk_kda_fwd_intra` + `chunk_gated_delta_rule_fwd_h` | 产生 K6 的输入 `g/v_new/Aqk/h` |
| 2 | `gla_output_torch`（torch_npu 算子图） | 精度/性能**基准** |
| 3 | `gla_output_kernel`（triton kernel） | 被测对象 |
| 4 | `gla_output_ref`（纯 torch CPU，逐 chunk） | ground truth（`--no-ref` 可跳过） |

每个 case 输出：`torch_npu 耗时 / triton 耗时 / speedup`，以及
`max|triton-torch|`、`max|torch-ref|`、`max|triton-ref|` 的精度对。
PASS 条件：三者 `max-diff` 均 < 1e-2（默认）。退出码全过为 0。

## 快速开始

### 1. 生成测试数据

```bash
python3 gen_csv.py               # 10 个用例 (K=V=64, BF16)
```

### 2. 在容器内运行（NPU）

```bash
bash run_cpu.sh                          # 默认: 精度 + 加速比, 10 个用例
bash run_cpu.sh --repeats 50             # 更多计时轮次, 更稳的加速比
bash run_cpu.sh --max-diff 1e-3          # 更严的精度阈值
bash run_cpu.sh --no-ref                 # 跳过 CPU 参考对比, 只比 torch_npu vs triton
```

或手动：

```bash
docker exec -w /docker/zhm/0505_skill_test/sonnet/sglang/gla_output triton-ascend-env-zhm bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/torch/lib:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export PYTHONPATH=/docker/zhm/0505_skill_test/sonnet/sglang/python:$PYTHONPATH
  python3 run.py
'
```

> 必须先 source CANN 的 `set_env.sh`，设置 `LD_LIBRARY_PATH` 与
> `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；否则 torch_npu 的 DEVICE_BACKEND autoload
> 会与 triton-ascend 冲突。`PYTHONPATH` 需包含 `sglang/python` 以便
> `run.py` 调用上游 K1-K5 算子。

### 3. 冒烟（纯 CPU 参考，无 NPU）

```bash
python3 -c "from src.gla_output_kernel import gla_output_ref; import torch; print(gla_output_ref(torch.randn(1,128,2,64), torch.randn(1,128,2,64), torch.randn(1,128,2,64), torch.randn(1,128,2,64), torch.randn(1,2,2,64,64), scale=0.125).shape)"
```

## 用例说明（10 个，`gen_csv.py` 中的表）

| id | B | T | H | K=V | 覆盖点 |
|----|---|---|---|-----|--------|
| tiny_default | 1 | 128 | 2 | 64 | 完整 2 chunks |
| tiny_partial_chunk | 1 | 63 | 2 | 64 | 尾 chunk 不满 BT=64 |
| tiny_single_head | 1 | 128 | 1 | 64 | 单 head |
| tiny_H3 | 1 | 128 | 3 | 64 | 奇数 head |
| tiny_T65 | 1 | 65 | 2 | 64 | 第二 chunk 仅 1 token |
| tiny_T96 | 1 | 96 | 2 | 64 | T 非 2 的幂 |
| tiny_T1 | 1 | 1 | 2 | 64 | 单 token |
| tiny_T2 | 1 | 2 | 2 | 64 | 两个 token |
| tiny_T100 | 1 | 100 | 2 | 64 | T=100 (尾 chunk 空 28) |
| tiny_T127 | 1 | 127 | 2 | 64 | 第二 chunk 缺 1 |

## 与上游的对应关系

| 上游（sglang/kernels/ops/attention/fla/kda.py） | 本目录 |
|------------------------------------------------|--------|
| `chunk_gla_fwd_kernel_o` (triton) | `src/gla_output_kernel.py::chunk_gla_fwd_kernel_o` |
| `chunk_gla_fwd_o_gk` (python driver) | `src/gla_output_kernel.py::gla_output_kernel` |
| CPU 参考 (测试) `_cpu_gla_output` | `src/gla_output_kernel.py::gla_output_ref` |
| 性能基准 (新) torch_npu 算子图 | `src/gla_output_kernel.py::gla_output_torch` |

设计细节见 `DESIGN.md`。

## msprof 级性能测试（NPU kernel 时间）

对比 `run.py` 用 `torch.npu.synchronize()` 包住的 wall-clock 时间，msprof 记录的是
**设备侧 kernel 执行时间**（`op_summary.csv` 的 `Task Duration(us)`），不含 python/GE
调度开销：

```bash
# 1) 用 msprof 包裹 run.py（自动补 --repeats 5 --warmup 2，保证 per_case 分组正确）
bash run_cpu.sh --msprof ./prof --no-ref

# 2) 解析（两种粒度）
python3 analyze_profile.py --latest-dir ./prof               # 全量聚合
python3 analyze_profile.py --latest-dir ./prof --scope k6    # 只算 gla_output_torch 的算子
python3 per_case_profile.py --latest-dir ./prof               # 逐 case 加速比 (默认 scope=k6)
python3 per_case_profile.py --latest-dir ./prof --scope all   # 逐 case 含上游 K1-K5

# 支持额外参数:
python3 analyze_profile.py --latest-dir ./prof --metric aiv --no-pad
python3 per_case_profile.py --latest-dir ./prof --case tiny_default --triton-calls 4
```

- `analyze_profile.py` 把所有非 triton kernel 求和为「torch_npu 拼接」时间。
  `--scope k6` 只统计 `gla_output_torch` 实际发出的 `aclnn*` 算子（与 run.py 的
  wall-clock 口径一致）；`--scope all`（默认）额外包含上游 K1-K5 算子时间。
- `per_case_profile.py` 按 op_summary **行序(程序顺序)** 把整段 trace 切分成 10 个
  case 组（每组 = 该 case 的 torch_npu 算子 + 2×(warmup+repeats+1) 次 triton kernel
  调用，即 warmup=2/repeats=5 → 16 次）。`--scope k6`（默认）只统计 K6 算子，
  `--scope all` 额外含上游 K1-K5 算子时间。
- 当 `run_cpu.sh --msprof` 用了非默认的 `--repeats` / `--warmup` 时，
  `per_case_profile.py` 的 `--triton-calls` 应相应调整（公式：
  `2 * (warmup + repeats + 1)`）。

实测（910B2, triton-ascend, warmup=2/repeats=5）：10/10 PASS；wall-clock 口径下
加速比约 1.1x–1.8x；msprof 内核时间口径下（scope=k6, --no-pad）总加速比约
**20x**——torch_npu 算子图发出 13 种 aclnn* 算子共 ~794 次内核调用，triton
单 kernel 完成同样计算。差异主要来自算子图启动开销与中间张量（`qg`、
`o_cross`）的读写。
