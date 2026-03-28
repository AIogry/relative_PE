#!/bin/bash
#SBATCH --job-name=pe-wikitext-eval
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/wikitext/%j_wikitext.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/wikitext/%j_wikitext.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# 使用WikiText-103进行评估（与C4不同领域，避免数据污染）

set -e

MODEL_PATH=${1}
PE_TYPE=${2}
SIGMA=${3:-700.0}
SEED=${4:-42}

if [ -z "${MODEL_PATH}" ] || [ -z "${PE_TYPE}" ]; then
    echo "Usage: sbatch scripts/run_wikitext_evaluation.sh <model_path> <pe_type> [sigma] [seed]"
    exit 1
fi

CODE_DIR="/home/qijunrong/03-proj/PE"
DATA_DIR="/data/qijunrong/03-proj/PE"

mkdir -p ${DATA_DIR}/logs/wikitext ${DATA_DIR}/results/wikitext

export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="${CODE_DIR}/eval_fewshot_wikitext.py"

MODEL_SIZE="300M"
BASE_LEN=512
TEST_LENGTHS=(1024 2048 4096)

# 【关键】使用WikiText进行few-shot和测试
FEW_SHOT_K=5000
FEW_SHOT_STEPS=300
FEW_SHOT_LR=5e-6

MODEL_NAME=$(basename $(dirname ${MODEL_PATH}))
RESULT_FILE="${DATA_DIR}/results/wikitext/${MODEL_NAME}_wikitext_K${FEW_SHOT_K}.json"

echo "=================================================="
echo "WikiText-103 Evaluation (Cross-Domain)"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "PE Type: ${PE_TYPE}"
echo "Data: WikiText-103 (different domain from C4)"
echo "Few-shot K: ${FEW_SHOT_K} (from WikiText train)"
echo "Test: WikiText validation"
echo "=================================================="

${PYTHON_BIN} ${SCRIPT} \
    --model_path ${MODEL_PATH} \
    --model_size ${MODEL_SIZE} \
    --wikitext_path "${DATA_DIR}/wikitext/raw" \
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
