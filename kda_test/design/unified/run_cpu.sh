#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# 在 triton-ascend-env-zhm 容器内运行 KDA 6 算子统一 bench。
#
# 用法:
#   bash run_cpu.sh                              # 默认: 模式 A 正确性
#   bash run_cpu.sh --limit 3                    # 冒烟: 只跑前 3 个 case
#   bash run_cpu.sh --msprof ./prof              # 模式 B: msprof 采集
#   bash run_cpu.sh --msprof ./prof --repeats 5 --warmup 2
#
# 关键点:
#   * 必须先 source CANN set_env.sh 并设置 LD_LIBRARY_PATH / TORCH_DEVICE_BACKEND_AUTOLOAD=0,
#     否则 torch_npu 无法加载。
#   * 本脚本必须运行在容器 triton-ascend-env-zhm 内；从宿主机:
#       docker exec -w <dir> triton-ascend-env-zhm bash run_cpu.sh ...

set -e

# 容器内 CANN 环境
: "${SET_ENV:=/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [ -f "$SET_ENV" ]; then
  # shellcheck disable=SC1090
  source "$SET_ENV"
fi

# triton-ascend 需要 torch/torch_npu 的动态库在 LD_LIBRARY_PATH
TORCH_LIB=$(python3 -c 'import os,sysconfig; p=os.path.join(sysconfig.get_paths()["purelib"],"torch","lib"); print(p)' 2>/dev/null || true)
TORCHNPU_LIB=$(python3 -c 'import os,sysconfig; p=os.path.join(sysconfig.get_paths()["purelib"],"torch_npu","lib"); print(p)' 2>/dev/null || true)
for lib in "$TORCH_LIB" "$TORCHNPU_LIB"; do
  if [ -d "$lib" ]; then
    export LD_LIBRARY_PATH="$lib:$LD_LIBRARY_PATH"
  fi
done
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

cd "$(dirname "$0")"

# 用例表 (cases_meta.json) 若不存在, 先生成
if [ ! -f cases_meta.json ]; then
  echo "没有 cases_meta.json, 先生成…"
  python3 gen_cases.py
fi

if [ "$1" = "--msprof" ]; then
  output="${2:-./prof}"
  shift 2 || true
  PROFILE_ARGS=()
  for a in "$@"; do PROFILE_ARGS+=("$a"); done
  if [[ " $* " != *"--repeats"* ]]; then PROFILE_ARGS+=(--repeats 5); fi
  if [[ " $* " != *"--warmup"* ]]; then PROFILE_ARGS+=(--warmup 2); fi
  echo "[msprof] msprof --output=$output --application=\"python3 bench.py --msprof ${PROFILE_ARGS[*]}\""
  msprof --output="$output" --application="python3 bench.py --msprof ${PROFILE_ARGS[*]}"
  echo "msprof 数据已写入 $output; 解析:"
  echo "  python3 analyze_profile.py --latest-dir $output"
  echo "  python3 per_case_profile.py --latest-dir $output"
  exit $?
fi

python3 bench.py "$@"
