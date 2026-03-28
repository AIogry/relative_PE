#!/bin/bash
#SBATCH --job-name=pe-arxiv-eval
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/arxiv/%j_arxiv.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/arxiv/%j_arxiv.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# ArXiv评估脚本

set -e

MODEL_PATH=${1}
PE_TYPE=${2}
SIGMA=${3:-700.0}
ADAPT_MODE=${4:-"base"}  # "base" or "extrap"
SEED=${5:-42}

if [ -z "${MODEL_PATH}" ] || [ -z "${PE_TYPE}" ]; then
    echo "Usage: sbatch scripts/run_arxiv_evaluation.sh <model_path> <pe_type> [sigma] [adapt_mode] [seed]"
    echo "  adapt_mode: 'base' (adapt on 512) or 'extrap' (adapt on each length)"
    exit 1
fi

CODE_DIR="/home/qijunrong/03-proj/PE"
DATA_DIR="/data/qijunrong/03-proj/PE"

mkdir -p ${DATA_DIR}/logs/arxiv ${DATA_DIR}/results/arxiv

export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="${CODE_DIR}/eval_fewshot_arxiv.py"

MODEL_SIZE="300M"
BASE_LEN=512
TEST_LENGTHS=(1024 2048 4096)

FEW_SHOT_K=256
FEW_SHOT_STEPS=50
FEW_SHOT_LR=5e-6

MODEL_NAME=$(basename $(dirname ${MODEL_PATH}))
ADAPT_SUFFIX="_${ADAPT_MODE}adapt"
RESULT_FILE="${DATA_DIR}/results/arxiv/${MODEL_NAME}_arxiv_K${FEW_SHOT_K}${ADAPT_SUFFIX}.json"

if [ "${ADAPT_MODE}" == "extrap" ]; then
    ADAPT_FLAG="--adapt_on_extrap"
    echo "Mode: Adapt on EACH extrap length"
else
    ADAPT_FLAG=""
    echo "Mode: Adapt on BASE length (512)"
fi

echo "=================================================="
echo "ArXiv Evaluation"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "PE Type: ${PE_TYPE}"
echo "Adapt Mode: ${ADAPT_MODE}"
echo "=================================================="

${PYTHON_BIN} ${SCRIPT} \
    --model_path ${MODEL_PATH} \
    --model_size ${MODEL_SIZE} \
    --arxiv_train_path "${DATA_DIR}/arxiv_data/arxiv_train" \
    --arxiv_val_path "${DATA_DIR}/arxiv_data/arxiv_validation" \
    --local_tokenizer_path "${DATA_DIR}/wikitext/tokenizer" \
    --few_shot_k ${FEW_SHOT_K} \
    --few_shot_steps ${FEW_SHOT_STEPS} \
    --few_shot_lr ${FEW_SHOT_LR} \
    ${ADAPT_FLAG} \
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
