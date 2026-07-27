# chunk_kda_fwd 算子 910B2 性能验证任务文档

## 1. 算子和环境概览

### 目标算子
- **文件**: `python/sglang/kernels/ops/attention/fla/kda.py`
- **函数**: `chunk_kda_fwd` (line 1023)
- **功能**: KDA (Kernel Dynamic Attention) Prefill 分块计算，是 Kimi Linear 模型的核心注意力算子
- **调用链**:
  ```
  chunk_kda() → chunk_kda_fwd()
    ├── kda_gate_chunk_cumsum() 或 chunk_local_cumsum()  # Gate 激活 + cumsum
    ├── chunk_kda_fwd_intra()                               # 块内 KKT + QK 计算
    │     ├── chunk_kda_fwd_kernel_intra_sub_chunk          # (safe_gate 路径)
    │     │   或 chunk_kda_fwd_intra_token_parallel         # (默认路径)
    │     └── chunk_kda_fwd_kernel_inter_solve_fused         # 块间求解
    ├── chunk_gated_delta_rule_fwd_h()                      # 递归状态更新
    └── chunk_gla_fwd_o_gk()                                # 最终输出计算
  ```
- **实现方式**: 纯 Triton 语言编写，所有子 kernel 均为 `@triton.jit` + `@triton.autotune`

### 验证环境

| 项目 | 当前容器 | 验证容器 `triton-ascend-env-zy` |
|------|---------|-------------------------------|
| 硬件 | Ascend 910B2 × 8 | Ascend 910B2 × 8 |
| torch | 2.10.0+cpu | 2.9.0+cpu |
| torch_npu | 2.10.0.post2 | 2.9.0.post2 |
| triton | 3.7.0 | 3.2.0 (原生 Ascend backend) |
| CANN | 未配置 | 8.5.0 (需 source set_env.sh) |
| tsl | 未安装 | **不需要** (Triton 3.2.0 自带 Ascend backend) |

## 2. 环境启动命令

验证容器 `triton-ascend-env-zy` 的完整启动命令:

```bash
# 进入容器
docker exec -it triton-ascend-env-zy bash

# 每次新 shell 必须执行以下环境初始化:
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_npu/lib:$LD_LIBRARY_PATH

# 设置 Python path
export PYTHONPATH=/docker/zhm/0505_skill_test/sonnet/sglang/python:$PYTHONPATH

# 验证环境
python3 -c "
import torch; import torch_npu
print('NPU:', torch.npu.is_available(), torch.npu.get_device_name(0))
import triton
print('Triton:', triton.__version__, 'backend:', triton.runtime.driver.active.get_current_target().backend)
print('FLA device:', end=' ')
from sglang.kernels.ops.attention.fla.utils import get_available_device
print(get_available_device())
"
```

## 3. 可行性分析

### 环境验证结果

| 检查项 | 结果 | 备注 |
|--------|------|------|
| torch_npu 可用 | PASS | NPU device count: 8 |
| Triton Active Driver | PASS | `<ascend.NPUDriver>` |
| Triton Backend | PASS | `npu` |
| FLA utils 平台检测 | PASS | `device=npu, device_platform=npu` (无需修改代码) |
| 简单 Triton kernel (add) | PASS | NPU 上运行正确 |
| 缺失依赖 | 需安装 | pybase64, IPython, msgspec |

### 关键发现: Triton 3.2.0 已原生支持 Ascend

- **不需要 `tsl` 模块**。Triton 3.2.0 自带 `triton/backends/ascend/driver.py`，后端为 `npu`
- **不需要修改 FLA utils**。`_check_platform()` 返回 `npu`（非 nvidia/amd/intel/musa），但不会阻塞代码路径
- **需要固定 LD_LIBRARY_PATH**，包含 torch/lib 和 torch_npu/lib

## 4. 现有测试用例

### 4.1 单元测试: `test/registered/attention/test_kda_kernels.py`

| 测试类 | 测试内容 | 关联 chunk_kda_fwd |
|--------|---------|-------------------|
| `TestKDAFusedSigmoidGatingRecurrent` | 测试 fused_recurrent_kda 和 fused_sigmoid_gating_delta_rule_update 的一致性 | 间接 (decode 路径) |
| `TestKDAGateChunkCumsum` | 测试 `kda_gate_chunk_cumsum` 子 kernel | **直接** (prefill 子组件) |
| `TestKDAChunkExponentDomain` | 测试 `chunk_kda()` 与 naive 递归实现的正确性 | **直接** (prefill 端到端) |
| `TestKDAPackedDecode` | 测试 packed decode 路径 | 不直接 (decode 路径) |

**关键测试**: `TestKDAChunkExponentDomain.test_chunk_prefill_matches_natural_exp_recurrence` (line 288) 是唯一直接测试 `chunk_kda()` → `chunk_kda_fwd()` 正确性的用例。

**已修改**: 所有 `@unittest.skipIf` 已更新为支持 NPU，硬编码 `device="cuda"` 已替换为 `get_device()`。详见第 4.3 节。

### 4.2 Benchmark: `benchmark/bench_linear_attention/`

| 文件 | 内容 |
|------|------|
| `bench_kda_decode.py` | KDA Packed Decode 性能对比 (decode, 非 prefill) |
| `bench_kda_prefill_cutedsl.py` | Triton KDA vs CuTeDSL KDA (prefill, Blackwell SM100) |
| `bench_cutedsl_kda_decode.py` | CuTeDSL KDA decode |

### 4.3 测试文件 NPU 兼容性修改

文件: `test/registered/attention/test_kda_kernels.py`

**修改内容**:

1. 导入 `is_npu`:
   ```python
   from sglang.srt.utils.common import get_device, is_npu
   ```

2. 4 个 `@unittest.skipIf` 更新:

   | 测试类 | 原条件 | 新条件 |
   |--------|-------|--------|
   | `TestKDAFusedSigmoidGatingRecurrent` | `cuda or xpu` | `cuda or xpu or npu` |
   | `TestKDAGateChunkCumsum` | `cuda` | `cuda or npu` |
   | `TestKDAChunkExponentDomain` | `cuda` | `cuda or npu` |
   | `TestKDAPackedDecode` | `cuda` | `cuda or npu` |

3. `TestKDAGateChunkCumsum._run_case` 中 4 处 `device="cuda"` → `device=get_device()`

其他测试类已经在使用 `get_device()` 或 `self.device`，无需修改。

## 5. 逐层验证结果

### 验证配置

| 参数 | 值 |
|------|-----|
| B, T, H, K, V | 1, 128/256, 2/4, 64, 64 |
| chunk_size | 64 |
| dtype | bfloat16 |

### 阶段 1: chunk_local_cumsum — PASS

```
Input:  [1, 256, 4, 64]
Output: [1, 256, 4, 64]
Autotune: BS=16, num_warps=8, num_stages=4
Time: 0.243ms
```

### 阶段 2: kda_gate_chunk_cumsum — PASS

```
Input:  [1, 256, 4, 64]
Output: [1, 256, 4, 64], dtype=float32
Autotune: BS=32, num_warps=2, num_stages=2
Time: 0.183ms
```

### 阶段 3: chunk_kda_fwd_intra — PARTIAL FAIL

`chunk_kda_fwd_intra` 内部包含 2 个 kernel:

| 子 kernel | 文件 | 结果 |
|-----------|------|------|
| `chunk_kda_fwd_intra_token_parallel` | `chunk_intra_token_parallel.py` | PASS (0.185ms) |
| `chunk_kda_fwd_kernel_inter_solve_fused` | `chunk_intra.py:47` | **FAIL** |

**失败根因**: `chunk_kda_fwd_kernel_inter_solve_fused` 在 Ascend BiShengHIR 编译器阶段崩溃:

```
loc("kernel.ttadapter.mlir":2:1): error: Failed to run BiShengHIR pipeline
loc("kernel.ttadapter.mlir":274:22): error: 'scf.if' op along control flow edge
  from Region #1 to parent results: source type #0 'memref<16x16xf32, #hivm.address_space<ub>>'
  should match input type #0 'memref<16x16xf32, #hivm.address_space<gm>>'
loc("kernel.ttadapter.mlir":821:14): error: 'hivm.hir.vcast' op Unsupported op for finding the root alloc.
```

**错误本质**: 
1. `scf.if` 分支的不同内存空间 (UB vs GM) 无法统一
2. `hivm.hir.vcast` 操作不被 BiShengHIR 编译器支持

**影响范围**: 该 kernel 是整个 prefill 路径的核心（块间求解 + 可选 recompute 融合），失败意味着 `chunk_kda_fwd` 无法完整运行。

### 阶段 4-6: 未执行

由于阶段 3 的 `inter_solve_fused` 编译失败，`chunk_kda_fwd` 的完整调用链无法走通。后续的 `chunk_gated_delta_rule_fwd_h` 和 `chunk_gla_fwd_o_gk` 未测试。

## 6. 阻塞问题分析

### 当前阻塞: `chunk_kda_fwd_kernel_inter_solve_fused` 编译失败

**Triton kernel 源码** (`chunk_intra.py:47`):
- 使用 `@triton.autotune` (BK=[32,64], BV=64, num_warps=[1,2,4])
- 使用 `@triton.heuristics` (IS_VARLEN 条件编译)
- 使用 `tl.make_block_ptr` 进行块内存访问
- 使用 `tl.dot` 进行矩阵乘法
- 包含 `scf.if` 条件分支 (FUSE_RECOMPUTE / FUSE_DIAGONAL 控制)

**BiShengHIR 编译器错误**:
1. `'hivm.hir.vcast' op Unsupported op` — 编译器不支持 `vcast` 操作
2. `scf.if` memory space mismatch — 分支间不同内存空间 (UB vs GM) 类型不兼容

### 可能的解决方向

1. **绕过 inter_solve_fused 的 autotune 配置**: 尝试禁用某些 config 组合，看是否有特定 config 能通过编译
2. **禁用 fused 路径**: 强制 `fuse_diagonal=False, fuse_recompute=False`，避免 `scf.if` 分支
3. **升级编译器**: 检查是否有更新版本的 CANN/BiShengHIR 修复了 `vcast` 和 memory space 问题
4. **改写 kernel**: 拆分 `inter_solve_fused` 为更简单的 kernel，避免 Trition→MLIR 编译器的边界情况

## 7. 执行检查清单

- [x] 4.1 进入 triton-ascend-env-zy 容器
- [x] 4.2 验证 tsl + Triton + NPU 环境 (Triton 3.2.0 自带 Ascend backend，无需 tsl)
- [x] 4.3 FLA utils 平台检测 (无需修改，正确识别 npu)
- [x] 5.1 环境验证通过
- [x] 5.2.1 chunk_local_cumsum 在 NPU 上运行成功
- [x] 5.2.2 kda_gate_chunk_cumsum 在 NPU 上运行成功
- [ ] 5.2.3 chunk_kda_fwd_intra 在 NPU 上运行成功 — **BLOCKED**: `inter_solve_fused` BiShengHIR 编译失败
- [ ] 5.2.4 chunk_gated_delta_rule_fwd_h 在 NPU 上运行成功
- [ ] 5.2.5 chunk_gla_fwd_o_gk 在 NPU 上运行成功
- [ ] 5.2.6 完整 chunk_kda_fwd 在 NPU 上运行成功
- [ ] 5.3 正确性测试通过 (与 naive 实现对比)
- [ ] 5.4 性能数据采集完成
- [ ] 5.5 对比分析报告完成
- [x] 测试文件 NPU 兼容性修改: `test_kda_kernels.py` 的 skipIf + device 硬编码已修复

## 8. 下一步

解决 `chunk_kda_fwd_kernel_inter_solve_fused` 的 BiShengHIR 编译问题是关键。建议优先尝试:

1. 使用 `safe_gate=True` 路径（走 `chunk_kda_fwd_kernel_intra_sub_chunk` 而非 `chunk_kda_fwd_intra_token_parallel`），观察是否改变编译行为
2. 尝试不同的 autotune config 组合（仅 BK=64, num_warps=1 等最简配置）
3. 检查 CANN 8.5.0 是否有已知的 `vcast` 支持限制

## 9. 手动测试命令

### 9.1 完整 chunk_kda_fwd 端到端测试

```bash
# 进入验证容器并初始化环境
docker exec -it triton-ascend-env-zy bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_npu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/docker/zhm/0505_skill_test/sonnet/sglang/python:$PYTHONPATH

# 运行测试脚本
python3 << 'PYEOF'
import sys
sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")
import torch
import torch_npu

from sglang.kernels.ops.attention.fla.kda import chunk_kda

# ---- 测试配置 ----
device = "npu"
dtype = torch.bfloat16
H, K, V = 2, 64, 64          # heads, key_dim, value_dim
T = 129                       # 总 tokens (129 = 2*64 + 1, 跨 3 个 chunk)
lengths = [129]               # 单序列
B = len(lengths)               # batch = 序列数

# ---- 构造输入 ----
torch.manual_seed(42)
shape = (1, T, H, K)           # (B=1, T, H, K)
q = torch.nn.functional.normalize(
    torch.randn(shape, dtype=torch.float32, device=device), dim=-1
).to(dtype)
k = torch.nn.functional.normalize(
    torch.randn(shape, dtype=torch.float32, device=device), dim=-1
).to(dtype)
v = torch.randn(1, T, H, V, dtype=dtype, device=device) * 0.1

# Gate: 使用 A_log 激活路径 (fuse_gate=True)
raw_gate = (torch.randn(1, T, H, K, dtype=torch.float32, device=device) * 0.5 - 2.0).to(dtype)
A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.1
dt_bias = torch.randn(H * K, dtype=torch.float32, device=device) * 0.1

# Beta: sigmoid 激活
beta = torch.rand(1, T, H, dtype=dtype, device=device).sigmoid()

# Initial state
initial_state = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.05
initial_state_indices = torch.arange(B, dtype=torch.int32, device=device)

print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
print(f"g: {raw_gate.shape}, beta: {beta.shape}")
print(f"initial_state: {initial_state.shape}")

# ---- 运行 chunk_kda (内部调用 chunk_kda_fwd) ----
print("\nRunning chunk_kda (A_log path, fuse_gate=True)...")
try:
    output = chunk_kda(
        q=q, k=k, v=v,
        g=raw_gate,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    torch.npu.synchronize()
    print(f"PASS: output shape = {output.shape}")
    print(f"output mean = {output.float().mean().item():.6f}, std = {output.float().std().item():.6f}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
PYEOF
```

### 9.2 无 A_log 路径 (pre-activated gate + cumsum)

```bash
# 如果 A_log 路径因 inter_solve_fused 失败，尝试直接传入已激活的 gate
python3 << 'PYEOF'
import sys
sys.path.insert(0, "/docker/zhm/0505_skill_test/sonnet/sglang/python")
import torch
import torch_npu
from sglang.kernels.ops.attention.fla.kda import chunk_kda

device = "npu"
dtype = torch.bfloat16
H, K, V, T = 2, 64, 64, 128

torch.manual_seed(42)
q = torch.nn.functional.normalize(
    torch.randn(1, T, H, K, dtype=torch.float32, device=device), dim=-1
).to(dtype)
k = torch.nn.functional.normalize(
    torch.randn(1, T, H, K, dtype=torch.float32, device=device), dim=-1
).to(dtype)
v = torch.randn(1, T, H, V, dtype=dtype, device=device) * 0.1
# 直接传入已激活的 gate (A_log=None 路径)
g = torch.randn(1, T, H, K, dtype=dtype, device=device) * 0.5
beta = torch.rand(1, T, H, dtype=dtype, device=device).sigmoid()
initial_state = torch.randn(1, H, K, V, dtype=torch.float32, device=device) * 0.05
initial_state_indices = torch.tensor([0], dtype=torch.int32, device=device)

print("Running chunk_kda (A_log=None, pre-activated gate)...")
try:
    output = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        A_log=None,  # skip gate activation
    )
    torch.npu.synchronize()
    print(f"PASS: output shape = {output.shape}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
PYEOF
```

### 9.3 运行现有单元测试 (修改后支持 NPU 的版本)

```bash
# 运行单个测试类
python3 -m pytest test/registered/attention/test_kda_kernels.py::TestKDAGateChunkCumsum -v 2>&1

# 运行 KDA 端到端测试 (注: 当前因 inter_solve_fused 编译失败, 预期会报错)
python3 -m pytest test/registered/attention/test_kda_kernels.py::TestKDAChunkExponentDomain -v 2>&1
```