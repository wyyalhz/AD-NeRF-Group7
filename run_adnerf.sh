#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "用法: $0 <train|infer> [参数...]"
    exit 1
fi

MODE="$1"
shift

ID="Obama"
AUD_FILE=""
TEST_SIZE=""
PREPROCESS="off"   # 新增：训练预处理开关

while [[ $# -gt 0 ]]; do
    case "$1" in
        --id)
            ID="$2"; shift 2;;
        --aud_file)
            AUD_FILE="$2"; shift 2;;
        --test_size)
            TEST_SIZE="$2"; shift 2;;
        --preprocess)
            PREPROCESS="$2"; shift 2;;
        *)
            echo "未知参数: $1"
            exit 1;;
    esac
done

echo "[run_adnerf] MODE       = ${MODE}"
echo "[run_adnerf] ID         = ${ID}"
echo "[run_adnerf] AUD_FILE   = ${AUD_FILE}"
echo "[run_adnerf] TEST_SIZE  = ${TEST_SIZE}"
echo "[run_adnerf] PREPROCESS = ${PREPROCESS}"
echo

case "$MODE" in
    train)
        echo "[run_adnerf] 调用 run_train.sh 开始训练 AD-NeRF..."
        echo "[run_adnerf] -> id = ${ID}"
        echo "[run_adnerf] -> preprocess = ${PREPROCESS}"
        echo

        # 直接按新接口调用（你已改过 run_train.sh 支持 --preprocess）
        bash run_train.sh "$ID" --preprocess "$PREPROCESS"

        echo
        echo "[run_adnerf] 训练流程结束。"
        ;;

    infer)
        if [ -z "$AUD_FILE" ]; then
            AUD_FILE="dataset/${ID}/aud.wav"
        fi
        if [ -z "$TEST_SIZE" ]; then
            TEST_SIZE=300
        fi

        echo "[run_adnerf] 调用 run_infer.sh 进行推理..."
        echo "[run_adnerf] -> id       = ${ID}"
        echo "[run_adnerf] -> aud_file  = ${AUD_FILE}"
        echo "[run_adnerf] -> test_size = ${TEST_SIZE}"
        echo

        bash run_infer.sh "$ID" "$AUD_FILE" "$TEST_SIZE"

        echo
        echo "[run_adnerf] 推理流程结束。"
        ;;

    *)
        echo "未知模式: ${MODE}（只能是 train 或 infer）"
        exit 1;;
esac
