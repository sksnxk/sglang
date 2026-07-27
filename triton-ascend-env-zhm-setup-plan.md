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

### Step 2: 编译安装 triton-ascend 3.6.0 🔄 进行中

**环境准备**:
- 安装 clang-15 + lld-15: `apt-get install -y clang-15 lld-15`
- 安装 ninja: `pip install ninja`
- 设置 clang 软链接: `update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100`

**实际环境版本**:
| 组件 | 版本 |
|------|------|
| torch | 2.7.1 |
| torch_npu | 2.7.1 |
| CANN | 8.5.0 |
| clang | 15.0.7 |
| cmake | 3.22.1 |
| ninja | 1.13.0 |
| pybind11 | 3.0.1 |

**编译命令**:
```bash
docker exec triton-ascend-env-zhm bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_npu/lib:$LD_LIBRARY_PATH
  cd /docker/zhm/0505_skill_test/sonnet/triton-ascend
  pip install -e .
'
```

**状态**: 后台编译中，预计 30-60 分钟...

### Step 3: 验证 ⏳ 待编译完成

编译完成后执行:
1. 验证 triton 版本升级到 3.6.0
2. 验证 Triton NPU backend 正常
3. 运行简单 Triton kernel 示例 (add)
4. 运行 KDA 相关 kernel 测试

## 4. 风险点

| 风险 | 说明 | 状态 |
|------|------|------|
| NPU 设备初始化失败 | 只传部分设备导致 drvRet=87 | ✅ 已解决（传全部 8 个） |
| 容器权限不足 | 缺少 --privileged 导致 NPU 不可用 | ✅ 已解决 |
| torch_npu 版本不匹配 | 源码要求 2.7.1，镜像内实际就是 2.7.1 | ✅ 无风险 |
| 缺少编译工具 | 镜像未预装 clang/ninja | ✅ 已安装 |
| 编译失败 | triton-ascend 3.6.0 + CANN 8.5.0 兼容性 | ⏳ 待确认 |
| 编译时间 | 源码编译 30-60 分钟 | ⏳ 进行中 |

## 5. 备用方案

如果源码编译失败，可使用以下替代方案：

1. **直接使用现有 `triton-ascend-env-zy` 容器**（triton 3.2.0），但 KDA `inter_solve_fused` kernel 无法编译
2. **拉取官方预编译镜像**: `quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11`（需确认 CANN 9.0.0 与当前驱动兼容）
3. **在现有容器内升级 triton-ascend**: 直接 `pip install --upgrade` 官方 wheel（如果可用）