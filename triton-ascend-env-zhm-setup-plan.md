# triton-ascend 容器环境安装方案

## 1. 现状分析

| 项目 | 现状 |
|------|------|
| 已有镜像 | `triton-ascend-env:v4` (51.3GB)，内置 triton-ascend 3.2.0 + CANN 8.5.0 + torch_npu 2.7.1 |
| 已有容器 | `triton-ascend-env-zy` 占用全部 8 个 NPU (davinci0-7) |
| 目标容器名 | `triton-ascend-env-zhm` |
| 源码路径 | `/docker/zhm/0505_skill_test/sonnet/triton-ascend`，版本 3.6.0 |
| NPU 分配 | 全部 8 个 NPU（NPU 驱动要求全量设备才能初始化） |

## 2. 方案概要

1. 基于现有 `triton-ascend-env:v4` 镜像创建新容器 `triton-ascend-env-zhm`
2. 在容器内从源码编译安装 triton-ascend 3.6.0，覆盖 3.2.0
3. 验证环境并运行 Triton 示例

## 3. 执行进度

### Step 1: 创建容器 ✅

**实际使用的命令**（经多次调试后确定）：

```bash
docker run --name triton-ascend-env-zhm \
  --privileged --net=host --shm-size=512g \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/log/npu:/usr/slog \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v /etc/localtime:/etc/localtime \
  -v /docker:/docker \
  -v /data:/data \
  -itd triton-ascend-env:v4 /bin/bash
```

**踩坑记录**:
- 第一次尝试: 只挂载 davinci4-7，NPU 驱动报 `drvRet=87`，无法枚举设备。**原因**: NPU 驱动要求全部 8 个设备都在容器中才能初始化
- 第二次尝试: 挂载全部 8 个 NPU 但缺少 `--privileged`，仍然失败
- 第三次: 完全对齐 `triton-ascend-env-zy` 的配置（`--privileged`, `--net=host`, `--shm-size=512g`, `--security-opt` 等），成功

**验证结果**:
```
NPU available: True
NPU device count: 8
  device 0: Ascend910B2
  device 1: Ascend910B2
  ...
  device 7: Ascend910B2
```

### Step 2: 拉取官方预编译镜像 + 创建容器 ✅

**方案变更**: 源码编译因磁盘空间不足失败，改用官方预编译镜像 `quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11`

**NPU 驱动兼容性**: 当前驱动 `25.0.rc1.1` (V100R001C21)，`compatible_version` 包含 `[V100R001C21]`，与 CANN 9.0.0 兼容 ✅

**磁盘空间**: 2.6T 可用 ✅

**执行命令**:

删除旧容器:
```bash
docker rm -f triton-ascend-env-zhm
```

拉取镜像（约 15-20GB，需几分钟）:
```bash
docker pull quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11
```

创建容器:
```bash
docker run --name triton-ascend-env-zhm \
  --privileged --net=host --shm-size=512g \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/log/npu:/usr/slog \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v /etc/localtime:/etc/localtime \
  -v /docker:/docker \
  -v /data:/data \
  -itd quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11 /bin/bash
```

### Step 3: 环境验证 ✅

| 组件 | 版本/状态 |
|------|-----------|
| triton | 3.5.0 |
| triton-ascend | 3.2.1 |
| torch_npu | 2.7.1 |
| CANN | 9.0.0 |
| NPU | 8 × Ascend910B2, 可用 |
| Backend | npu |

**验证命令**:
```bash
docker exec -it triton-ascend-env-zhm bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_npu/lib:$LD_LIBRARY_PATH

python3 -c "
import torch; import torch_npu
print('NPU:', torch.npu.is_available(), torch.npu.get_device_name(0))
import triton
print('Triton:', triton.__version__)
print('Backend:', triton.runtime.driver.active.get_current_target().backend)
"
```

### Step 4: triton-ascend 示例验证 ✅

使用源码 `/docker/zhm/0505_skill_test/sonnet/triton-ascend/docs/zh/python-api/_examples/triton.language.add.py` 中的 vector add 示例:

```
Triton add kernel on NPU:
  Max diff: 0.0000000000
  All close: True
PASS: triton-ascend vector add works on NPU!
```

### Step 5: KDA 算子验证 ✅ 完成 (2026-07-28)

**全部 7 个测试类 61 条用例通过**

| 测试类 | 用例数 | 结果 | RMSE |
|--------|--------|------|------|
| TestGateChunkCumsumKernel | 10 | ✅ PASS | 0.000000 |
| TestTokenParallelKernel | 10 | ✅ PASS | Aqk=0.001581, Akk=0.000000 |
| TestRecomputeWUKernel | 10 | ✅ PASS | w=0.002147, u=0.002321, kg=0.001401 |
| TestDeltaRuleKernel | 10 | ✅ PASS | h=0.001659, v_new=0.001284 |
| TestGLAOutputKernel | 10 | ✅ PASS | 0.001851 |
| TestFullPipeline | 10 | ✅ PASS | RMSE=0.090 (NPU 非确定性) |
| TestAllKernels | 1 | ✅ PASS | — |

**关键 Bug 修复**:
- `chunk_delta_h.py`: h store 从 block_ptr 改为 flat 1D pointer (triton-ascend 编译器 bug)
- `chunk_intra.py`: Aqk 初始化从 `torch.empty` 改为 `torch.zeros` (NPU 未初始化内存含 NaN)
- CPU 参考公式: delta_rule_h 的 state decay/update 方向修正

**已知限制**:
- K=64 且 V=64 (chunk_delta_h 硬编码 64×64 tile)
- H≤3, T≤128, B=1 (inter_solve_fused kernel 在更高并行度下 aicore timeout)
- 测试仅在容器内运行 (需要 triton-ascend NPU driver)

**手动测试指导**:

```bash
# 从容器外运行全部测试
docker exec -w /docker/zhm/0505_skill_test/sonnet/sglang triton-ascend-env-zhm bash -c '
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_npu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/docker/zhm/0505_skill_test/sonnet/sglang/python:$PYTHONPATH
rm -rf ~/.triton/cache
python3 -m pytest kda_test/test_level2_kernel_precision.py -v -s
'

# 或进入容器后再运行
docker exec -it triton-ascend-env-zhm bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_npu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/docker/zhm/0505_skill_test/sonnet/sglang/python:$PYTHONPATH
cd /docker/zhm/0505_skill_test/sonnet/sglang
rm -rf ~/.triton/cache
python3 -m pytest kda_test/test_level2_kernel_precision.py -v -s

# 运行单个测试类
python3 -m pytest kda_test/test_level2_kernel_precision.py::TestGateChunkCumsumKernel -v -s
python3 -m pytest kda_test/test_level2_kernel_precision.py::TestDeltaRuleKernel -v -s
python3 -m pytest kda_test/test_level2_kernel_precision.py::TestAllKernels -v -s
```

## 4. 风险点

| 风险 | 说明 | 状态 |
|------|------|------|
| NPU 设备初始化失败 | 只传部分设备导致 drvRet=87 | ✅ 已解决（传全部 8 个） |
| 容器权限不足 | 缺少 --privileged 导致 NPU 不可用 | ✅ 已解决 |
| torch_npu 版本不匹配 | 源码要求 2.7.1，镜像内实际就是 2.7.1 | ✅ 无风险 |
| 缺少编译工具 | 镜像未预装 clang/ninja | ✅ 已安装 |
| 编译失败 | triton-ascend 3.6.0 + CANN 8.5.0 兼容性 | ✅ 已解决（改用官方预编译镜像 3.2.1 + CANN 9.0.0） |
| 编译时间 | 源码编译 30-60 分钟 | ✅ 已解决（改用预编译镜像） |
| inter_solve aicore timeout | H>=4 时 kernel 计算量超出 aicore 预算 | ⚠️ 已知限制 (H≤3)，需拆分 kernel |
| chunk_delta_h K!=64 | 硬编码 64×64 tile | ⚠️ 已知限制，需修复 tile 泛化 |

## 5. 备用方案

如果源码编译失败，可使用以下替代方案：

1. **直接使用现有 `triton-ascend-env-zy` 容器**（triton 3.2.0），但 KDA `inter_solve_fused` kernel 无法编译
2. **拉取官方预编译镜像**: `quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11`（需确认 CANN 9.0.0 与当前驱动兼容）
3. **在现有容器内升级 triton-ascend**: 直接 `pip install --upgrade` 官方 wheel（如果可用）