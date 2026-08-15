# recompute_w_u 独立验证目录

KDA Kernel-4（Recompute W/U: `w=A@(k*beta*exp2(gk))`, `u=A@(v*beta)`,
`kg=k*exp2(gk_last-gk)`）的独立、可上传验证目录。不依赖 sglang 包，
只依赖 torch + torch_npu + triton（NPU / triton-ascend）。

## 目录结构

```
recompute_w_u/
├── DESIGN.md            设计文档（数学、分核、数据流、精度/性能对比策略）
├── gen_csv.py           生成 testcases.csv（15 个用例的输入张量，base64 编码）
├── testcases.csv        （生成物）用例输入数据
├── run.py               主要测试脚本：读 CSV → torch_npu 元算子和 triton kernel
│                        分别跑精度与性能，输出加速比（见下）
├── run_cpu.sh           容器内运行脚本：source CANN + LD_LIBRARY_PATH + AUTOLOAD=0
├── analyze_profile.py   （性能）解析 msprof 的 op_summary_*.csv，对比整体
│                        torch_npu 单算子拼接 vs triton kernel 总时长
├── per_case_profile.py  （性能）逐 case 解析 msprof op_summary，
│                        按程序顺序分组计算每个 case 的加速比
├── src/
│   ├── __init__.py
│   └── recompute_w_u_kernel.py   三种实现：CPU 参考 recompute_w_u_ref、
│                                   torch_npu 元算子 recompute_w_u_torch、
│                                   triton kernel recompute_w_u_triton
└── util/
    ├── __init__.py
    └── csvb64.py        base64 CSV 编解码（保证张量逐 bit 往返一致）
```

## 算子数学（Kernel-4：Recompute W/U）

给定 chunk 内的 KKT 逆矩阵 `A` (= Akk_inv, `[BT, BT]` 下三角)、Key/Value
张量与 chunk 局部 gate cumsum `gk`，重新计算解耦后的 `w/u/kg`：

```
w  = A @ (k * beta * exp2(gk))        # Key 解耦表示, 可跨 chunk 递推
u  = A @ (v * beta)                   # Value 解耦表示
kg = k * exp2(gk_last - gk)           # 时间对齐 Key (gk_last = chunk 内最后 token 的 gk)
```

- `w/u` 去掉 chunk 内的因果依赖, 可与外积形式 `w * u^T` 参与跨 chunk 递推;
- `kg` 把 chunk 内每个 token 的 Key 对齐到 chunk 末尾时间戳, 供后续
  `chunk_gla_fwd_o_gk` 使用。

### Grid 拓扑

```
Grid = (NT, B * H)
  * NT = cdiv(T, BT): chunk 个数
  * B * H: 所有 (batch, head) 对
每个 CTA 处理一个 (chunk, head), 公共加载 beta/A_inv 留寄存器,
随后串行做 V 维循环 (u) + K 维循环 (w + 可选 kg)。
```

## run.py：精度 + 加速比测试

读 CSV 中每行用例，对同一份数据跑两个实现并对比：

| 步骤 | 实现 | 用途 |
|------|------|------|
| 1 | `recompute_w_u_torch`（torch_npu 算子图） | 精度/性能**基准** |
| 2 | `recompute_w_u_triton`（triton kernel） | 被测对象 |
| 3 | `recompute_w_u_ref`（纯 torch CPU，逐 chunk） | ground truth（`--no-ref` 可跳过） |

每个 case 输出：`torch_npu 耗时 / triton 耗时 / speedup`，以及
`max|triton-torch|`（w / u / kg 分开）与 `max|torch-ref|`、`max|triton-ref|` 的精度对。
PASS 条件：三者 `max-diff` 均 < 1e-2（默认）。退出码全过为 0。

## 快速开始

### 1. 生成测试数据

```bash
python3 gen_csv.py
```

### 2. 在容器内运行（NPU）

```bash
bash run_cpu.sh                          # 默认: 精度 + 加速比, 15 个用例
bash run_cpu.sh --repeats 50             # 更多计时轮次, 更稳的加速比
bash run_cpu.sh --max-diff 1e-3          # 更严的精度阈值
bash run_cpu.sh --no-ref                 # 跳过 CPU 参考对比, 只比 torch_npu vs triton
```

或手动：

```bash
docker exec -w /docker/zhm/0505_skill_test/sonnet/sglang/recompute_w_u triton-ascend-env-zhm bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/torch/lib:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  python3 run.py
'
```

> 必须先 source CANN 的 `set_env.sh`，并设置 `LD_LIBRARY_PATH` 与
> `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；否则 torch_npu 的 DEVICE_BACKEND autoload
> 会与 triton-ascend 冲突。

### 3. 冒烟（纯 CPU 参考，无 NPU）

```bash
python3 -c "
from src.recompute_w_u_kernel import recompute_w_u_ref
import torch
k = torch.randn(1, 128, 2, 64)
v = torch.randn(1, 128, 2, 64)
beta = torch.rand(1, 128, 2).sigmoid()
A = torch.eye(64).unsqueeze(0).unsqueeze(2).expand(1, 128, 2, 64).contiguous()
gk = torch.randn(1, 128, 2, 64) * 0.5 - 2.0
w, u, kg = recompute_w_u_ref(k, v, beta, A, gk=gk)
print(w.shape, u.shape, kg.shape)
"
```

## 用例说明（15 个，`gen_csv.py` 中的表）

| id | B | T | H | K | 覆盖点 |
|----|---|---|---|---|--------|
| tiny_default | 1 | 128 | 2 | 64 | 完整 2 chunks |
| tiny_partial_chunk | 1 | 63 | 2 | 64 | 尾 chunk 不满 BT=64 |
| tiny_single_head | 1 | 128 | 1 | 64 | 单 head |
| tiny_H3 | 1 | 128 | 3 | 64 | 奇数 head |
| tiny_T65 | 1 | 65 | 2 | 64 | 第二 chunk 仅 1 token |
| tiny_T96 | 1 | 96 | 2 | 64 | T 非 2 的幂 |
| tiny_T1 | 1 | 1 | 2 | 64 | 单 token |
| tiny_T2 | 1 | 2 | 2 | 64 | 两个 token |
| tiny_T100 | 1 | 100 | 2 | 64 | T=100（尾 chunk 空 28） |
| tiny_T127 | 1 | 127 | 2 | 64 | 第二 chunk 缺 1 |
| big_B2_H3_K128 | 2 | 100 | 3 | 128 | 多 batch/head，K 非 32 倍数 |
| big_single_T193 | 1 | 193 | 1 | 32 | T 非 BT 倍数 + K=32 |
| k32_basic | 1 | 64 | 1 | 32 | K 为 BS 整数倍 |
| k128t127 | 1 | 127 | 2 | 128 | K=128 + 尾 chunk 不满 |
| tiny_T2562 | 1 | 2562 | 1 | 64 | 超长 T（40 chunks） |

## A (Akk_inv) 生成策略

`A` 是 Kernel-3 (inter_solve) 的输出。`gen_csv.py` 在生成 k/v/beta/gk 后，
**内联生成 Akk_inv 近似**（单位下三角 + alpha=0.1 随机扰动），不依赖
inter_solve 目录，保持 recompute_w_u 自包含：

```
A[chunk] = I + alpha * strict_tril(randn)   # alpha=0.1, 对角线占主导
```

这与真实 Akk_inv 的数据分布一致（Akk_inv = (I - strict_tril(Akk))^{-1} 的
合并下三角逆，对角线接近 1）。

## 与上游的对应关系

| 上游（sglang/kernels/ops/attention/fla/kda.py） | 本目录 |
|------------------------------------------------|--------|
| `recompute_w_u_fwd_kernel` | `src/recompute_w_u_kernel.py::_recompute_w_u_kernel` |
| `recompute_w_u_fwd` (python driver) | `src/recompute_w_u_kernel.py::recompute_w_u_triton` |
| CPU 参考（测试）`_cpu_recompute_w_u` | `src/recompute_w_u_kernel.py::recompute_w_u_ref` |
| 性能基准（新）torch_npu 算子图 | `src/recompute_w_u_kernel.py::recompute_w_u_torch` |

设计细节见 `DESIGN.md`。

## msprof 级性能测试（NPU kernel 时间）

对比 `run.py` 用 `torch.npu.synchronize()` 包住的 wall-clock 时间，msprof 记录的是
**设备侧 kernel 执行时间**（`op_summary.csv` 的 `Task Duration(us)`），不含 python/GE
调度开销：

```bash
# 1) 用 msprof 包裹 run.py（自动补 --repeats 5 --warmup 2，保证 per_case 分组正确）
bash run_cpu.sh --msprof ./prof --no-ref

# 2) 解析（两种粒度）
python3 analyze_profile.py --latest-dir ./prof   # 全量: torch_npu 拼接 vs triton 总时长
python3 per_case_profile.py --latest-dir ./prof  # 逐 case 加速比(按程序顺序分组)

# 支持额外参数, 如取纯 AIV 时间 / 不加每 kernel 启动开销:
python3 analyze_profile.py --latest-dir ./prof --metric aiv --no-pad
python3 per_case_profile.py --latest-dir ./prof --case tiny_default
```

- `analyze_profile.py` 把所有非 triton kernel 求和为「torch_npu 单算子拼接」时间；
- `per_case_profile.py` 按 op_summary **行序(程序顺序)** 把整段 trace 切分成 15 个
  case 组（每组 = torch 拼接 + 8 次 triton 调用，即 warmup=2/repeats=5 的调用数），
  给出每个 case 的加速比。
