#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# 在 triton-ascend-env-zhm 容器内, 以「先 source CANN 环境 + 设置 NPU 相关环境变量」
# 的方式运行 inter-solve (非对角 Aqk/Akk + 对角块前向替换 + 链式求逆) 的精度&性能测试。
#
# 用法:
#   bash run_cpu.sh [CSV]           # 默认 testcases.csv, 跑 triton(NPU)+ref
#   bash run_cpu.sh testcases.csv --max-diff 1e-3
#
# 关键点:
#   * 必须先 source SET_ENV 和设置 LD_LIBRARY_PATH / TORCH_DEVICE_BACKEND_AUTOLOAD=0,
#     否则 torch_npu 无法加载 (torch 的 DEVICE_BACKEND auto-load 会与 torch_npu 冲突)。
#   * 本脚本可直接在 docker 容器内执行; 从宿主机冒烟时可用:
#       docker exec -w <dir> triton-ascend-env-zhm bash run_cpu.sh ...

set -e

# 容器内 CANN 环境
: "${SET_ENV:=/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [ -f "$SET_ENV" ]; then
  # shellcheck disable=SC1090
  source "$SET_ENV"
fi

# triton-ascend 需要 torch/torch_npu 的动态库在 LD_LIBRARY_PATH
# (镜像里 python3.11 site-packages 路径; 按实际环境调整)
TORCH_LIB=$(python3 -c 'import os,sysconfig; p=os.path.join(sysconfig.get_paths()["purelib"],"torch","lib"); print(p)' 2>/dev/null || true)
TORCHNPU_LIB=$(python3 -c 'import os,sysconfig; p=os.path.join(sysconfig.get_paths()["purelib"],"torch_npu","lib"); print(p)' 2>/dev/null || true)
for lib in "$TORCH_LIB" "$TORCHNPU_LIB"; do
  if [ -d "$lib" ]; then
    export LD_LIBRARY_PATH="$lib:$LD_LIBRARY_PATH"
  fi
done
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

cd "$(dirname "$0")"

if [ ! -f testcases.csv ]; then
  echo "没有 testcases.csv, 先生成…"
  python3 gen_csv.py
fi

# 需要 msprof 级性能数据(op_summary)时, 用 msprof 包裹 run.py:
#
#   bash run_cpu.sh --msprof ./prof [附加 run.py 参数]   # →
#      msprof --output=./prof python3 run.py [args]      # (未显式给 --repeats/--warmup 时自动用 5/2)
#   python3 analyze_profile.py --latest-dir ./prof       # 全量: torch_npu 拼接 vs triton
#   python3 per_case_profile.py --latest-dir ./prof      # 逐 case 加速比(按行序分组)
if [ "$1" = "--msprof" ]; then
  output="${2:-./prof}"
  shift 2 || true
  PROFILE_ARGS=()
  for a in "$@"; do PROFILE_ARGS+=("$a"); done
  if [[ " $* " != *"--repeats"* ]]; then PROFILE_ARGS+=(--repeats 5); fi
  if [[ " $* " != *"--warmup"* ]]; then PROFILE_ARGS+=(--warmup 2); fi
  echo "[msprof] msprof --output=$output --application=\"python3 run.py ${PROFILE_ARGS[*]}\""
  msprof --output="$output" --application="python3 run.py ${PROFILE_ARGS[*]}"
  echo "msprof 数据已写入 $output; 解析:"
  echo "  python3 analyze_profile.py --latest-dir $output"
  echo "  python3 per_case_profile.py --latest-dir $output"
  exit $?
fi

python3 run.py "$@"
