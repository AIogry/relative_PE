#!/bin/bash
#SBATCH --job-name=pe-fixed-eval
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/fixed/%j_fixed.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/fixed/%j_fixed.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# 修复版评估 - 使用C4数据，大样本量

set -e

MODEL_PATH=${1}
PE_TYPE=${2}
SIGMA=${3:-700.0}
SEED=${4:-42}

if [ -z "${MODEL_PATH}" ] || [ -z "${PE_TYPE}" ]; then
    echo "Usage: sbatch scripts/run_fixed_evaluation.sh <model_path> <pe_type> [sigma] [seed]"
    exit 1
fi

CODE_DIR="/home/qijunrong/03-proj/PE"
DATA_DIR="/data/qijunrong/03-proj/PE"

mkdir -p ${DATA_DIR}/logs/fixed ${DATA_DIR}/results/fixed

export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="${CODE_DIR}/eval_fewshot_extrap_fixed.py"

MODEL_SIZE="300M"
BASE_LEN=512
TEST_LENGTHS=(1024 2048 4096)

# 【关键修改】大样本量+更多步数
FEW_SHOT_K=2000
FEW_SHOT_STEPS=300
FEW_SHOT_LR=5e-6

MODEL_NAME=$(basename $(dirname ${MODEL_PATH}))
RESULT_FILE="${DATA_DIR}/results/fixed/${MODEL_NAME}_K${FEW_SHOT_K}_fixed.json"

echo "=================================================="
echo "Fixed Evaluation (C4 Domain, Large K)"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "PE Type: ${PE_TYPE}"
echo "Few-shot K: ${FEW_SHOT_K} (from C4 train)"
echo "Test data: C4 validation (same domain)"
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
