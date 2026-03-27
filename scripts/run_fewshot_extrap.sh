#!/bin/bash
#SBATCH --job-name=pe-fewshot-extrap
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/extrap/%j_%x.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/extrap/%j_%x.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00

# Few-Shot外推评估脚本
# 使用说明: sbatch scripts/run_fewshot_extrap.sh <model_path> <pe_type> [sigma] [seed]

set -e

# 解析参数
MODEL_PATH=${1}
PE_TYPE=${2}
SIGMA=${3:-700.0}
SEED=${4:-6198}

if [ -z "${MODEL_PATH}" ] || [ -z "${PE_TYPE}" ]; then
    echo "Usage: sbatch scripts/run_fewshot_extrap.sh <model_path> <pe_type> [sigma] [seed]"
    echo "Example: sbatch scripts/run_fewshot_extrap.sh /data/qijunrong/03-proj/PE/checkpoints/pretrain_c4/300M_hipe_yarn_L512_sig700.0_s42/model_final.pt hipe_yarn 700.0 42"
    exit 1
fi

# 路径配置
CODE_DIR="/home/qijunrong/03-proj/PE"        # 代码目录
DATA_DIR="/data/qijunrong/03-proj/PE"        # 数据目录

ARXIV_PATH="${DATA_DIR}/arxiv_data/arxiv_validation"
TOKENIZER_PATH="${DATA_DIR}/wikitext/tokenizer"
RESULTS_ROOT="${DATA_DIR}/results/fewshot_extrap"
LOG_DIR="${DATA_DIR}/logs/extrap"

mkdir -p ${RESULTS_ROOT} ${LOG_DIR}

# 环境配置
export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="${CODE_DIR}/eval_fewshot_extrap.py"

# 实验配置
MODEL_SIZE="300M"
BASE_LEN=512
FEW_SHOT_K=128        # 减少适应样本，防止过拟合
FEW_SHOT_STEPS=50     # 约0.5 epoch
FEW_SHOT_LR=5e-6      # 更低学习率，保守调整
TEST_LENGTHS=(1024 2048 4096)

# 从模型路径提取模型名称
MODEL_NAME=$(basename $(dirname ${MODEL_PATH}))
RESULT_FILE="${RESULTS_ROOT}/${MODEL_NAME}_extrap.json"

echo "=================================================="
echo "Few-Shot Extrapolation Evaluation"
echo "=================================================="
echo "Code Dir: ${CODE_DIR}"
echo "Data Dir: ${DATA_DIR}"
echo "Model: ${MODEL_PATH}"
echo "PE Type: ${PE_TYPE}"
echo "Sigma: ${SIGMA}"
echo "Base Length: ${BASE_LEN}"
echo "Test Lengths: ${TEST_LENGTHS[@]}"
echo "Few-shot K: ${FEW_SHOT_K}, Steps: ${FEW_SHOT_STEPS}"
echo "Output: ${RESULT_FILE}"
echo "=================================================="

${PYTHON_BIN} ${SCRIPT} \
    --model_path ${MODEL_PATH} \
    --model_size ${MODEL_SIZE} \
    --arxiv_data_path ${ARXIV_PATH} \
    --local_tokenizer_path ${TOKENIZER_PATH} \
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
    --output_file ${RESULT_FILE} \
    # --eval_batches 100  # 默认使用全部测试数据

echo "Evaluation complete: ${RESULT_FILE}"
