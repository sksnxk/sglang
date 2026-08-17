# inter_solve 独立验证目录

KDA Kernel-3（Inter-Solve Fused：非对角线 Aqk/Akk + 对角块前向替换 + 链式求逆）
的独立、可上传验证目录。不依赖 sglang 包，只依赖 torch + torch_npu + triton
（NPU / triton-ascend）。

## 目录结构

```
inter_solve/
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
│   └── inter_solve_kernel.py   三种实现：CPU 参考 inter_solve_ref、
│                                   torch_npu 元算子 inter_solve_torch、
│                                   triton kernel inter_solve_triton
└── util/
    ├── __init__.py
    └── csvb64.py        base64 CSV 编解码（保证张量逐 bit 往返一致）
```

## 算子数学（Kernel-3：Inter-Solve Fused）

每个 chunk/head 分配一个独立 CTA，在该 chunk 内完成三阶段：

**Phase 1（非对角线块）**——6 对 (i>j) 子块的 gated dot product：

```
Akk_ij = (K_i * exp2(G_i - G_i[last])) @ (K_j * exp2(G_i[last] - G_j))^T * beta_j
Aqk_ij = (Q_i * exp2(G_i - G_i[last])) @ (K_j * exp2(G_i[last] - G_j))^T * scale
  (i, j = 0..3, i > j;  G_i[last] 是子块 i 的末尾 token 的 g 参考点)
```

**Phase 2（对角块前向替换）**——4 个 16×16 严格下三角块逐行求逆：

```
D_inv = (I - strict_tril(D))^{-1}
```

**Phase 3（链式矩阵乘合并逆）**：

```
Ai_10 = -Ai_11 @ Akk_10 @ Ai_00
Ai_21 = -Ai_22 @ Akk_21 @ Ai_11
Ai_32 = -Ai_33 @ Akk_32 @ Ai_22
Ai_20 = -Ai_22 @ (Akk_20 @ Ai_00 + Akk_21 @ Ai_10)
Ai_31 = -Ai_33 @ (Akk_31 @ Ai_11 + Akk_32 @ Ai_21)
Ai_30 = -Ai_33 @ (Akk_30 @ Ai_00 + Akk_31 @ Ai_10 + Akk_32 @ Ai_20)
```

输出的数据布局与上游一致：

- `Aqk [B,T,H,BT]`：非对角线 Aqk 块，列 = j 在 chunk 内位置；
- `Akk_inv [B,T,H,BT]`：10 个 16×16 子块合并的下三角逆，上三角为 0。

## run.py：精度 + 加速比测试

读 CSV 中每行用例，对同一份数据跑两个实现并对比：

| 步骤 | 实现 | 用途 |
|------|------|------|
| 1 | `inter_solve_torch`（torch_npu 元算子图） | 精度/性能**基准** |
| 2 | `inter_solve_triton`（triton kernel） | 被测对象 |
| 3 | `inter_solve_ref`（纯 torch CPU，逐 chunk 循环） | ground truth（`--no-ref` 可跳过） |

每个 case 输出：`torch_npu 耗时 / triton 耗时 / speedup`，以及
`max|triton-torch|`（Aqk / Akk_inv 分开）与 `max|torch-ref|`、`max|triton-ref|` 的精度对。
PASS 条件：max-diff 均 < 1e-2（默认）。退出码全过为 0。

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
docker exec -w /docker/zhm/0505_skill_test/sonnet/sglang/inter_solve triton-ascend-env-zhm bash -c '
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
python3 -c "from src.inter_solve_kernel import inter_solve_ref; import torch
q,k,g,beta=torch.randn(1,128,2,64),torch.randn(1,128,2,64),torch.randn(1,128,2,64),torch.randn(1,128,2)
Akkd=torch.randn(1,128,2,16); Aqk,Akk_inv=inter_solve_ref(q,k,g,beta,Akkd,0.1768); print(Aqk.shape, Akk_inv.shape)"
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

## Akkd 生成策略

`Akkd` 是 Kernel-2（token_parallel）的输出。`gen_csv.py` 在生成 q/k/g/beta 后，
**内联计算 Akkd**（逐 token 循环, 严格下三角 gated dot product），不依赖
token_parallel 目录，保持 inter_solve 自包含：

```
Akkd[i, j] = <k[i]*beta[i], k[j] * exp2(g[i]-g[j])>   (j < i, 同 sub-chunk)
Akkd[b, t, h, j - i_ts] = sum_d( k[i,d]*beta[i] * k[j,d] * exp2(g[i,d]-g[j,d]) )
```

## 与上游的对应关系

| 上游（sglang/kernels/ops/attention/fla/chunk_intra.py） | 本目录 |
|----------------------------------------------------------|--------|
| `chunk_kda_fwd_kernel_inter_solve_fused` | `src/inter_solve_kernel.py::_inter_solve_kernel` |
| `chunk_kda_fwd_intra` (非 FUSE_RECOMPUTE 分支) | `src/inter_solve_kernel.py::inter_solve_triton` |
| CPU 参考（测试）`_cpu_*` | `src/inter_solve_kernel.py::inter_solve_ref` |
| 性能基准（新）torch_npu 算子图 | `src/inter_solve_kernel.py::inter_solve_torch` |

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
