#!/usr/bin/env bash
# 统一训练入口脚本（支持预处理开关）
#
# 用法：
#   bash run_train.sh
#   bash run_train.sh Obama
#   bash run_train.sh Obama --preprocess on
#   bash run_train.sh Obama --preprocess off
#
# 默认：
#   ID=Obama
#   preprocess=off

set -e

########################################
# 0. conda 初始化
########################################
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/root/miniconda3/etc/profile.d/conda.sh"
fi

########################################
# 1. 定位项目根目录
########################################
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

########################################
# 2. 参数解析
########################################
ID="Obama"
DO_PREPROCESS="off"

# 第一个参数如果不是 flag，当作 ID
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
    ID="$1"
    shift
fi

# 解析 flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preprocess)
            DO_PREPROCESS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

DATASET_DIR="dataset/${ID}"

HEAD_CFG="${DATASET_DIR}/HeadNeRF_config.txt"
TORSO_CFG="${DATASET_DIR}/TorsoNeRF_config.txt"

LOG_DIR_HEAD="${DATASET_DIR}/logs/${ID}_head"
LOG_DIR_TORSO="${DATASET_DIR}/logs/${ID}_com"

mkdir -p "$LOG_DIR_HEAD" "$LOG_DIR_TORSO"

########################################
# 3. 打印配置
########################################
echo "[run_train.sh] ID            = ${ID}"
echo "[run_train.sh] PREPROCESS    = ${DO_PREPROCESS}"
echo "[run_train.sh] DATASET_DIR   = ${DATASET_DIR}"
echo "[run_train.sh] HeadNeRF CFG  = ${HEAD_CFG}"
echo "[run_train.sh] TorsoNeRF CFG = ${TORSO_CFG}"
echo

########################################
# 4. 可选：预处理流程
########################################
if [[ "$DO_PREPROCESS" == "on" ]]; then
    echo "========== 预处理 Step0 (adnerf_ds) =========="
    conda run -n adnerf_ds python data_util/process_data.py \
        --id "$ID" --step 0 \
        2>&1 | tee "${DATASET_DIR}/logs/preprocess_step0.log"
    echo

    echo "========== 预处理 Step1~7 (adnerf_pre) =========="
    for s in 1 2 3 4 5 6 7; do
        echo "---- Step ${s} ----"
        conda run -n adnerf_pre python data_util/process_data.py \
            --id "$ID" --step "$s" \
            2>&1 | tee "${DATASET_DIR}/logs/preprocess_step${s}.log"
    done
    echo
else
    echo "[run_train.sh] 跳过预处理步骤（--preprocess off）"
    echo
fi

########################################
# 5. 配置文件检查
########################################
if [ ! -f "$HEAD_CFG" ]; then
    echo "!!! 找不到 HeadNeRF 配置文件：$HEAD_CFG"
    exit 1
fi

if [ ! -f "$TORSO_CFG" ]; then
    echo "!!! 找不到 TorsoNeRF 配置文件：$TORSO_CFG"
    exit 1
fi

########################################
# 6. 训练 HeadNeRF（adnerf）
########################################
echo "========== 训练 HeadNeRF =========="
conda run -n adnerf python NeRFs/HeadNeRF/run_nerf.py \
    --config "$HEAD_CFG" \
    2>&1 | tee "${LOG_DIR_HEAD}/train_head.log"
echo

########################################
# 7. 同步 HeadNeRF checkpoint → TorsoNeRF
########################################
echo "========== 同步 HeadNeRF checkpoint =========="
HEAD_CKPT_DIR="${LOG_DIR_HEAD}/checkpoints"
TORSO_CKPT_DIR="${LOG_DIR_TORSO}/checkpoints"
mkdir -p "$TORSO_CKPT_DIR"

if [ -d "$HEAD_CKPT_DIR" ]; then
    LATEST_HEAD_CKPT="$(ls -t "$HEAD_CKPT_DIR" 2>/dev/null | head -n 1 || true)"
    if [ -n "$LATEST_HEAD_CKPT" ]; then
        cp -f "${HEAD_CKPT_DIR}/${LATEST_HEAD_CKPT}" "${TORSO_CKPT_DIR}/"
        echo "[run_train.sh] 已复制 ${LATEST_HEAD_CKPT}"
    else
        echo "!!! 未找到 head checkpoint"
    fi
else
    echo "!!! 未找到 head checkpoints 目录"
fi
echo

########################################
# 8. 训练 TorsoNeRF（adnerf）
########################################
echo "========== 训练 TorsoNeRF =========="
conda run -n adnerf python NeRFs/TorsoNeRF/run_nerf.py \
    --config "$TORSO_CFG" \
    2>&1 | tee "${LOG_DIR_TORSO}/train_torso.log"
echo

echo "[run_train.sh] 全流程完成。"
