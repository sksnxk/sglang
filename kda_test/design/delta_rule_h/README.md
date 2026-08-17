# delta_rule_h 独立验证目录

KDA Kernel-5（Delta Rule H：跨 chunk 状态递推）的独立、可上传验证目录。
不依赖 sglang 包，只依赖 torch + torch_npu + triton（NPU / triton-ascend）。

## 目录结构

```
delta_rule_h/
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
│   └── delta_rule_h_kernel.py   三种实现：CPU 参考 delta_rule_h_ref、
│                                   torch_npu 元算子 delta_rule_h_torch、
│                                   triton kernel delta_rule_h_triton
└── util/
    ├── __init__.py
    └── csvb64.py        base64 CSV 编解码（保证张量逐 bit 往返一致）
```

## 算子数学（Kernel-5：Delta Rule H）

每个 (batch, head, V-tile) 分配一个独立 CTA，在该序列的所有 NT 个 chunk 上串行递推：

```
for c in 0..NT-1:
    ① Save snapshot:    h[c] = state                          # 供 Kernel-6 读取
    ② Delta Rule:       v_new = u - w @ state^T               # 残差 = 原值 - 历史预测
    ③ Save v_new:       v_new[c] = v_new                     # 供 Kernel-6 读取
    ④ Per-channel decay: state *= exp2(gk_last)              # log2 空间逐 channel 衰减
    ⑤ State update:     state += k^T @ v_new                  # 外积累加
Epilogue: 写回 final state 到 initial_state (in-place)
```

输出数据布局与上游一致：

- `h [B, NT, H, V, K]`：每 chunk 起始状态快照（Kernel-6 读 `h[t]` 作为 chunk t 的起始状态）；
- `v_new [B, T, H, V]`：Delta Rule 残差 value（供 Kernel-6 output 使用）；
- `initial_state [N, H, V, K]`：in-place 更新为最终状态（供下一 batch 复用）。

只保留 `USE_GK + USE_EXP2 + INPLACE_UPDATE + SAVE_NEW_VALUE + USE_INITIAL_STATE` 路径，
删除上游的 VARLEN / `USE_G`（标量 gate）/ `USE_EXP2=False` 等分支。

## run.py：精度 + 加速比测试

读 CSV 中每行用例，对同一份数据跑两个实现并对比：

| 步骤 | 实现 | 用途 |
|------|------|------|
| 1 | `delta_rule_h_torch`（torch_npu 元算子图） | 精度/性能**基准** |
| 2 | `delta_rule_h_triton`（triton kernel） | 被测对象 |
| 3 | `delta_rule_h_ref`（纯 torch CPU，逐 chunk 串行） | ground truth（`--no-ref` 可跳过） |

每个 case 输出：`torch_npu 耗时 / triton 耗时 / speedup`，以及
`max|triton_h - torch_h|`、`max|triton_v_new - torch_v_new|` 的精度对，
可选 vs CPU 参考的 `max|torch - ref|`、`max|triton - ref|`。
PASS 条件：max-diff 均 < 1e-2（默认）。退出码全过为 0。

**注意**：triton kernel 与 torch_npu 都 in-place 更新 `initial_state`，故 `run.py`
每次跑前都 `clone` 一份 `initial_state`，避免污染下次基准 / 参考。

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
docker exec -w /docker/zhm/0505_skill_test/sonnet/sglang/delta_rule_h triton-ascend-env-zhm bash -c '
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
python3 -c "from src.delta_rule_h_kernel import delta_rule_h_ref; import torch
B,T,H,K=1,128,2,64; k=torch.randn(B,T,H,K); w=torch.randn(B,T,H,K)*0.1
u=torch.randn(B,T,H,K)*0.1; gk=torch.randn(B,T,H,K)*0.5-2.0
is_=torch.randn(B,H,K,K)*0.05; idx=torch.arange(B,dtype=torch.int32)
h,vn=delta_rule_h_ref(k,w,u,gk,is_,idx); print(h.shape, vn.shape)"
```

## 用例说明（15 个，`gen_csv.py` 中的表）

**重要约束**：所有用例固定 `K=V=64` —— 上游 `chunk_delta_h` kernel 的 flat 1D
store 假定行步长 K=64，在 K≠64 时会产生错误结果（与 `test_level2_kernel_precision.py`
的约定一致）。T 与 B/H 的变化覆盖各种边界情形。

| id | B | T | H | K(=V) | 覆盖点 |
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
| big_B2_H3 | 2 | 100 | 3 | 64 | 多 batch/head |
| big_B2_T193 | 2 | 193 | 2 | 64 | 多 batch + T 非 BT 倍数 |
| big_T256 | 1 | 256 | 2 | 64 | T=256（4 full chunks） |
| tiny_T255 | 1 | 255 | 2 | 64 | T=255：第二对 chunk 缺 1 |
| tiny_T2562 | 1 | 2562 | 1 | 64 | 超长 T（40 chunks） |

## 与上游的对应关系

| 上游（sglang/kernels/ops/attention/fla/chunk_delta_h.py） | 本目录 |
|----------------------------------------------------------|--------|
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `src/delta_rule_h_kernel.py::_delta_rule_h_kernel` |
| `chunk_gated_delta_rule_fwd_h` (python driver) | `src/delta_rule_h_kernel.py::delta_rule_h_triton` |
| CPU 参考（测试）`_cpu_delta_rule_h` | `src/delta_rule_h_kernel.py::delta_rule_h_ref` |
| 性能基准（新）torch_npu 算子图 | `src/delta_rule_h_kernel.py::delta_rule_h_torch` |

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
python3 per_case_profile.py --latest-dir ./prof  --case tiny_default
```

- `analyze_profile.py` 把所有非 triton kernel 求和为「torch_npu 单算子拼接」时间；
- `per_case_profile.py` 按 op_summary **行序(程序顺序)** 把整段 trace 切分成 15 个
  case 组（每组 = torch 拼接 + 8 次 triton 调用，即 warmup=2/repeats=5 的调用数），
  给出每个 case 的加速比。
