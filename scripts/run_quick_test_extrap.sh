#!/bin/bash
#SBATCH --job-name=pe-test-extrap
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/test/%j_test_extrap.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/test/%j_test_extrap.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=15:00

# 快速测试外推评估

set -e

# 解析参数
MODEL_PATH=${1}
PE_TYPE=${2:-"hipe_yarn"}
SIGMA=${3:-700.0}
SEED=${4:-6198}

if [ -z "${MODEL_PATH}" ]; then
    echo "Usage: sbatch scripts/run_quick_test_extrap.sh <model_path> [pe_type] [sigma] [seed]"
    echo "Example:"
    echo "  sbatch scripts/run_quick_test_extrap.sh \\"
    echo "    /data/qijunrong/03-proj/PE/checkpoints/test/TEST_20M_hipe_yarn_s42/model_final.pt \\"
    echo "    hipe_yarn 700.0 42"
    exit 1
fi

CODE_DIR="/home/qijunrong/03-proj/PE"
DATA_DIR="/data/qijunrong/03-proj/PE"

ARXIV_PATH="${DATA_DIR}/arxiv_data/arxiv_validation"
TOKENIZER_PATH="${DATA_DIR}/wikitext/tokenizer"
RESULTS_ROOT="${DATA_DIR}/results/test"

mkdir -p ${RESULTS_ROOT}

export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="${CODE_DIR}/eval_fewshot_extrap.py"

echo "=================================================="
echo "QUICK TEST - Extrapolation Evaluation"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "PE Type: ${PE_TYPE}"
echo "=================================================="

${PYTHON_BIN} ${SCRIPT} \
    --model_path ${MODEL_PATH} \
    --model_size "20M" \
    --arxiv_data_path ${ARXIV_PATH} \
    --local_tokenizer_path ${TOKENIZER_PATH} \
    --few_shot_k 50 \
    --few_shot_steps 10 \
    --few_shot_lr 1e-5 \
    --base_len 512 \
    --test_lengths 1024 2048 \
    --pe_type ${PE_TYPE} \
    --sigma ${SIGMA} \
    --rope_scaling_threshold 7 \
    --decay_func gaussian \
    --seed ${SEED} \
    --output_file "${RESULTS_ROOT}/test_extrap_result.json" \
    # --eval_batches 5  # 快速测试也用全部数据

echo "=================================================="
echo "Test evaluation complete!"
echo "=================================================="
