#!/bin/bash
#SBATCH --job-name=pe-c4-fixed-v2
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/c4_fixed_v2/%j_c4.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/c4_fixed_v2/%j_c4.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# C4评估 - 修复Padding Leakage版本

set -e

MODEL_PATH=${1}
PE_TYPE=${2}
SIGMA=${3:-700.0}
SEED=${4:-42}

if [ -z "${MODEL_PATH}" ] || [ -z "${PE_TYPE}" ]; then
    echo "Usage: sbatch scripts/run_c4_fixed_v2.sh <model_path> <pe_type> [sigma] [seed]"
    exit 1
fi

CODE_DIR="/home/qijunrong/03-proj/PE"
DATA_DIR="/data/qijunrong/03-proj/PE"

mkdir -p ${DATA_DIR}/logs/c4_fixed_v2 ${DATA_DIR}/results/c4_fixed_v2

export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="${CODE_DIR}/eval_fewshot_c4_fixed_v2.py"

MODEL_SIZE="300M"
BASE_LEN=512
TEST_LENGTHS=(1024 2048 4096)

# 小样本+适中步数
FEW_SHOT_K=1000
FEW_SHOT_STEPS=100
FEW_SHOT_LR=5e-6

MODEL_NAME=$(basename $(dirname ${MODEL_PATH}))
RESULT_FILE="${DATA_DIR}/results/c4_fixed_v2/${MODEL_NAME}_v2.json"

echo "=================================================="
echo "C4 Evaluation V2 (Padding Leakage Fixed)"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "PE Type: ${PE_TYPE}"
echo "Features:"
echo "  - group_texts (no padding)"
echo "  - per-length adaptation"
echo "  - ignore_index=50256"
echo "=================================================="

${PYTHON_BIN} ${SCRIPT} \
    --model_path ${MODEL_PATH} \
    --model_size ${MODEL_SIZE} \
    --dataset_path ${DATA_DIR} \
    --local_tokenizer_path "${DATA_DIR}/wikitext/tokenizer" \
    --few_shot_k ${FEW_SHOT_K} \
    --few_shot_steps ${FEW_SHOT_STEPS} \
    --few_shot_lr ${FEW_SHOT_LR} \
    --base_len ${BASE_LEN} \
    --test_lengths ${TEST_LENGTHS[@]} \
    --pe_type ${PE_TYPE} \
    --sigma ${SIGMA} \
    --rope_scaling_threshold 7 \
    --decay_func gaussian \
    --seed ${SEED} \
    --output_file ${RESULT_FILE}

echo ""
echo "=================================================="
echo "Evaluation complete!"
echo "Result: ${RESULT_FILE}"
echo "=================================================="
