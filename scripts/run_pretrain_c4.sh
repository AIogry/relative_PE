#!/bin/bash
#SBATCH --job-name=pe-c4-pretrain
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/pretrain/%j_%x.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/pretrain/%j_%x.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=96:00:00

# 预训练脚本：C4数据集
# 使用说明: sbatch scripts/run_pretrain_c4.sh <pe_type> [sigma] [seed]

set -e

# 解析参数
PE_TYPE=${1:-"hipe"}  # rope, hipe, rope_yarn, hipe_yarn
SIGMA=${2:-700.0}
SEED=${3:-6198}

# 路径配置
CODE_DIR="/home/qijunrong/03-proj/PE"        # 代码目录
DATA_DIR="/data/qijunrong/03-proj/PE"        # 数据目录

DATA_PATH="${DATA_DIR}"                      # 包含c4_30M_train/val
TOKENIZER_PATH="${DATA_DIR}/wikitext/tokenizer"
CHECKPOINT_ROOT="${DATA_DIR}/checkpoints/pretrain_c4"
WANDB_DIR="${DATA_DIR}/wandb"
LOG_DIR="${DATA_DIR}/logs/pretrain"

mkdir -p ${CHECKPOINT_ROOT} ${WANDB_DIR} ${LOG_DIR}

# 环境配置
export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="${CODE_DIR}/train_hipe_c4_pretrain.py"

# 实验配置
MODEL_SIZE="300M"
SEQ_LEN=512
MAX_TOKENS=1000000000  # 1B tokens
GLOBAL_BS=64
MICRO_BS=16
LR=3e-4
EVAL_INTERVAL=500
SAVE_INTERVAL=2000

# 构建运行ID
RUN_ID="${MODEL_SIZE}_${PE_TYPE}_L${SEQ_LEN}_sig${SIGMA}_s${SEED}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"

echo "=================================================="
echo "C4 Pretraining Experiment"
echo "=================================================="
echo "Code Dir: ${CODE_DIR}"
echo "Data Dir: ${DATA_DIR}"
echo "PE Type: ${PE_TYPE}"
echo "Sigma: ${SIGMA}"
echo "Model: ${MODEL_SIZE}"
echo "Seq Len: ${SEQ_LEN}"
echo "Output: ${OUTPUT_DIR}"
echo "Seed: ${SEED}"
echo "=================================================="

# 运行训练
${PYTHON_BIN} ${SCRIPT} \
    --output_dir ${OUTPUT_DIR} \
    --run_id ${RUN_ID} \
    --dataset_path ${DATA_PATH} \
    --local_tokenizer_path ${TOKENIZER_PATH} \
    --model_size ${MODEL_SIZE} \
    --seq_len ${SEQ_LEN} \
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

echo "Training complete: ${RUN_ID}"
