#!/usr/bin/env bash
# 统一推理 / 渲染入口脚本（支持 wav -> npy 自动预处理，支持可变 ID）
#
# 推荐用法（新）：
#   1) 输入 wav（自动生成同名 npy、输出同名 mp4）
#      bash run_infer.sh Obama dataset/Obama/other.wav 300
#
#   2) 输入 npy（直接推理、输出同名 mp4；wav 用同名 wav）
#      bash run_infer.sh Obama dataset/Obama/other.npy 300
#
# 兼容旧用法（老）：
#      bash run_infer.sh dataset/Obama/other.wav 300
#      bash run_infer.sh dataset/Obama/other.npy 300
#
# 规则（名字一致）：
#   other.wav  -> other.npy  -> other.mp4
#
# 输出：
#   dataset/<ID>/render_out/<name>.mp4

set -e

########################################
# 1. 让 conda 命令可用
########################################
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/root/miniconda3/etc/profile.d/conda.sh"
fi

########################################
# 2. 基本路径配置
########################################
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

########################################
# 3. 参数解析（兼容新旧两种调用）
########################################
# 新用法：run_infer.sh <ID> <IN_AUDIO> <TEST_SIZE>
# 旧用法：run_infer.sh <IN_AUDIO> <TEST_SIZE>

ID_DEFAULT="Obama"
TEST_SIZE_DEFAULT=300

# 判断第一个参数是不是像路径（含 / 或 以 dataset 开头）：
# 如果像路径，则认为使用旧用法：第一个参数是 IN_AUDIO
if [[ "${1:-}" == */* || "${1:-}" == dataset/* ]]; then
    ID="$ID_DEFAULT"
    IN_AUDIO="${1:-}"
    TEST_SIZE="${2:-$TEST_SIZE_DEFAULT}"
else
    # 新用法
    ID="${1:-$ID_DEFAULT}"
    IN_AUDIO="${2:-}"
    TEST_SIZE="${3:-$TEST_SIZE_DEFAULT}"
fi

DATASET_DIR="dataset/${ID}"
TORSO_TEST_CFG="${DATASET_DIR}/TorsoNeRFTest_config.txt"

# 若 IN_AUDIO 为空，则默认用 dataset/<ID>/aud.wav
if [ -z "$IN_AUDIO" ]; then
    IN_AUDIO="${DATASET_DIR}/aud.wav"
fi

########################################
# 4. 推导 NAME / WAV / NPY / MP4
########################################
IN_BASENAME="$(basename "$IN_AUDIO")"

if [[ "$IN_BASENAME" == *.wav ]]; then
    NAME="${IN_BASENAME%.wav}"
    INPUT_TYPE="wav"
elif [[ "$IN_BASENAME" == *.npy ]]; then
    NAME="${IN_BASENAME%.npy}"
    INPUT_TYPE="npy"
else
    echo "!!! 不支持的输入格式: $IN_AUDIO"
    echo "    请传 .wav 或 .npy"
    exit 1
fi

WAV_FILE="${DATASET_DIR}/${NAME}.wav"
NPY_FILE="${DATASET_DIR}/${NAME}.npy"

OUT_ROOT="${DATASET_DIR}/render_out"
OUT_MP4="${OUT_ROOT}/${NAME}.mp4"
mkdir -p "$OUT_ROOT"

########################################
# 5. 打印关键信息
########################################
echo "[run_infer.sh] 使用 ID           : ${ID}"
echo "[run_infer.sh] Torso 配置文件     : ${TORSO_TEST_CFG}"
echo "[run_infer.sh] 输入文件          : ${IN_AUDIO} (type=${INPUT_TYPE})"
echo "[run_infer.sh] 目标 wav           : ${WAV_FILE}"
echo "[run_infer.sh] 目标 npy           : ${NPY_FILE}"
echo "[run_infer.sh] test_size          : ${TEST_SIZE}"
echo "[run_infer.sh] 输出视频           : ${OUT_MP4}"
echo

########################################
# 6. 预处理：确保 wav(16k) + 生成同名 npy
########################################

# 6.1 输入 wav：统一生成 dataset/<ID>/<NAME>.wav（16k）
#     输入 npy：复制到 dataset/<ID>/<NAME>.npy，并要求同名 wav 存在用于合成 mp4
if [[ "$INPUT_TYPE" == "wav" ]]; then
    if [ ! -f "$IN_AUDIO" ]; then
        echo "!!! 输入 wav 不存在: $IN_AUDIO"
        exit 1
    fi

    if command -v ffmpeg >/dev/null 2>&1; then
        echo "[run_infer.sh] 使用 ffmpeg 将输入 wav 转为 16k: ${WAV_FILE}"
        TMP_WAV="${WAV_FILE}.tmp16k.wav"
        ffmpeg -y -i "$IN_AUDIO" -f wav -ar 16000 "$TMP_WAV"
        mv -f "$TMP_WAV" "$WAV_FILE"
    else
        echo "!!! 未安装 ffmpeg，无法保证 16k 采样率。直接复制 wav：${WAV_FILE}"
        cp -f "$IN_AUDIO" "$WAV_FILE"
    fi

elif [[ "$INPUT_TYPE" == "npy" ]]; then
    if [ ! -f "$IN_AUDIO" ]; then
        echo "!!! 输入 npy 不存在: $IN_AUDIO"
        exit 1
    fi

    echo "[run_infer.sh] 复制/覆盖 npy 到: ${NPY_FILE}"
    cp -f "$IN_AUDIO" "$NPY_FILE"

    if [ ! -f "$WAV_FILE" ]; then
        echo "!!! 输入是 npy，但缺少同名 wav 用于合成：${WAV_FILE}"
        echo "    请提供 ${NAME}.wav 放在 ${DATASET_DIR}/ 下，或改为传 wav 作为输入。"
        exit 1
    fi
fi

# 6.2 生成同名 npy（如果不存在）
if [ ! -f "$NPY_FILE" ]; then
    echo "[run_infer.sh] 未找到 npy，开始从 wav 提取 DeepSpeech 特征生成 npy..."

    TMP_DIR="${DATASET_DIR}/_ds_tmp_${NAME}"
    rm -rf "$TMP_DIR"
    mkdir -p "$TMP_DIR"

    cp -f "$WAV_FILE" "${TMP_DIR}/aud.wav"

    # 提取必须在 adnerf_ds 环境中跑
    conda run -n adnerf_ds python data_util/deepspeech_features/extract_ds_features.py --input "${TMP_DIR}"

    if [ ! -f "${TMP_DIR}/aud.npy" ]; then
        echo "!!! 提取失败：未生成 ${TMP_DIR}/aud.npy"
        exit 1
    fi

    mv -f "${TMP_DIR}/aud.npy" "$NPY_FILE"
    rm -rf "$TMP_DIR"

    echo "[run_infer.sh] 已生成 npy：${NPY_FILE}"
else
    echo "[run_infer.sh] 已存在 npy，跳过特征提取：${NPY_FILE}"
fi

########################################
# 7. NeRF 渲染（生成无声 .avi）
########################################
# 渲染在 adnerf 环境跑
conda run -n adnerf python NeRFs/TorsoNeRF/run_nerf.py \
  --config "$TORSO_TEST_CFG" \
  --aud_file "$NPY_FILE" \
  --test_size "$TEST_SIZE"

echo "[run_infer.sh] NeRF 渲染完成，开始查找最近生成的 .avi 文件..."

########################################
# 8. 查找最近生成的 avi 文件
########################################
AVI_PATH=$(find "$DATASET_DIR" -maxdepth 6 -name "*.avi" -mmin -10 | head -n 1 || true)

if [ -z "$AVI_PATH" ]; then
    echo "!!! 没找到最近生成的 .avi 文件，请检查 run_nerf 输出路径。"
    exit 1
fi

echo "[run_infer.sh] 找到 .avi 文件: $AVI_PATH"

########################################
# 9. ffmpeg 合成音视频（输出 NAME.mp4）
########################################
if command -v ffmpeg >/dev/null 2>&1; then
    if [ ! -f "$WAV_FILE" ]; then
        echo "!!! 没找到 wav：${WAV_FILE}，无法合成 mp4。"
        exit 1
    fi

    echo "[run_infer.sh] 使用 ffmpeg 合成音视频 -> ${OUT_MP4}"
    ffmpeg -y \
        -i "$AVI_PATH" \
        -i "$WAV_FILE" \
        -c:v libx264 \
        -c:a aac \
        -shortest \
        "$OUT_MP4" >/dev/null 2>&1

    echo "[run_infer.sh] 合成完成，输出视频: $OUT_MP4"
else
    echo "!!! 系统中未安装 ffmpeg，无法自动合成音视频。"
    echo "    你可以手动执行："
    echo "    ffmpeg -i ${AVI_PATH} -i ${WAV_FILE} -c:v libx264 -c:a aac ${OUT_MP4}"
fi

echo
echo "[run_infer.sh] 推理 / 渲染流程结束。"
