#!/bin/bash
#SBATCH --job-name=pe-quick-test
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/test/%j_quick_test.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/test/%j_quick_test.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=30:00

# 快速测试脚本 - 验证代码能否跑通
# 使用小模型(20M)、少量数据、短训练

set -e

# 解析参数
PE_TYPE=${1:-"hipe_yarn"}  # 默认测试新方案
SIGMA=${2:-700.0}
SEED=${3:-6198}

# 路径配置
CODE_DIR="/home/qijunrong/03-proj/PE"
DATA_DIR="/data/qijunrong/03-proj/PE"

DATA_PATH="${DATA_DIR}"
TOKENIZER_PATH="${DATA_DIR}/wikitext/tokenizer"
CHECKPOINT_ROOT="${DATA_DIR}/checkpoints/test"
WANDB_DIR="${DATA_DIR}/wandb"
LOG_DIR="${DATA_DIR}/logs/test"

mkdir -p ${CHECKPOINT_ROOT} ${WANDB_DIR} ${LOG_DIR}

# 环境配置
export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED=1

PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="${CODE_DIR}/train_hipe_c4_pretrain.py"

# 测试配置（小规模）
MODEL_SIZE="20M"           # 小模型，快速训练
SEQ_LEN=512
TRAIN_SIZE=1000            # 只用1000条数据
VAL_SIZE=100
MAX_TOKENS=10000000        # 10M tokens (约200步)
GLOBAL_BS=32
MICRO_BS=8
LR=3e-4
EVAL_INTERVAL=50
SAVE_INTERVAL=100

echo "=================================================="
echo "QUICK TEST - Code Validation"
echo "=================================================="
echo "Code Dir: ${CODE_DIR}"
echo "Data Dir: ${DATA_DIR}"
echo "PE Type: ${PE_TYPE}"
echo "Sigma: ${SIGMA}"
echo "Model: ${MODEL_SIZE} (small for testing)"
echo "Data: ${TRAIN_SIZE} samples"
echo "Expected steps: ~200"
echo "=================================================="

# 构建运行ID
RUN_ID="TEST_${MODEL_SIZE}_${PE_TYPE}_s${SEED}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"

echo "Starting test run..."
echo "Output: ${OUTPUT_DIR}"

${PYTHON_BIN} ${SCRIPT} \
    --output_dir ${OUTPUT_DIR} \
    --run_id ${RUN_ID} \
    --dataset_path ${DATA_PATH} \
    --local_tokenizer_path ${TOKENIZER_PATH} \
    --model_size ${MODEL_SIZE} \
    --seq_len ${SEQ_LEN} \
    --train_size ${TRAIN_SIZE} \
    --val_size ${VAL_SIZE} \
    --max_tokens ${MAX_TOKENS} \
    --global_batch_size ${GLOBAL_BS} \
    --micro_batch_size ${MICRO_BS} \
    --lr ${LR} \
    --pe_type ${PE_TYPE} \
    --sigma ${SIGMA} \
    --rope_scaling_threshold 7 \
    --decay_func gaussian \
    --eval_interval ${EVAL_INTERVAL} \
    --save_interval ${SAVE_INTERVAL} \
    --seed ${SEED} \
    --wandb_dir ${WANDB_DIR} \
    --wandb_mode offline

echo ""
echo "=================================================="
echo "TEST PASSED! Training completed successfully."
echo "Output: ${OUTPUT_DIR}"
echo "=================================================="

# 如果测试成功，提示可以运行正式实验
echo ""
echo "Next steps:"
echo "1. Check the output: tail -f ${LOG_DIR}/*.out"
echo "2. If successful, run full experiment:"
echo "   bash scripts/run_full_experiment_suite.sh"
