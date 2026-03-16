#!/bin/bash
#SBATCH --job-name=olmo-60m-ft-baseline
#SBATCH --ntasks=1
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/%x_%j.out
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# ================= 环境配置 =================

export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 
PYTHON_EXE="/home/qijunrong/anaconda3/bin/python"

# ================= 路径配置 =================
# 1. 基础模型路径 (依然是用之前的 Baseline 2k 模型)
PRETRAINED_CKPT="/data/qijunrong/03-proj/PE/checkpoints/olmo-60m-Baseline-RoPE-flash-len2048-1.5B/final_model.pt"

# 2. 数据路径
DATA_PATH="/data/qijunrong/03-proj/PE"

# 3. 配置文件
CONFIG_PATH="./configs/olmo_60m.yaml"

# ================= 微调配置 =================
# 注意：这里是 Baseline 的微调，不加 Scaled 参数
RUN_NAME="olmo-60m-Stage2-FT-8k-Baseline-Standard"
TARGET_SEQ_LEN=8192

# 训练步数 (保持一致)
MAX_STEPS=1000
GLOBAL_BS=128
MICRO_BS=4 

# ================= 启动微调 (Standard RoPE) =================
echo "===================================================================="
echo "Starting Stage 2 Fine-tuning (Baseline Standard RoPE)"
echo "From 2k -> 8k"
echo "GPU: 1"
echo "===================================================================="

$PYTHON_EXE train.py \
    --config $CONFIG_PATH \
    --run_name "$RUN_NAME" \
    --finetune_from "$PRETRAINED_CKPT" \
    --dataset_path "$DATA_PATH" \
    --position_embedding rope \
    --train_max_sequence_length $TARGET_SEQ_LEN \
    --val_max_sequence_length $TARGET_SEQ_LEN \
    --train_size 1000000 \
    --val_size 2000 \
    --batch_size $GLOBAL_BS \
    --micro_batch_size $MICRO_BS \
    --max_steps $MAX_STEPS \
    --save_interval 500 \
    --eval_interval 100

echo "Finished Fine-tuning Baseline: $RUN_NAME"