#!/bin/bash
#SBATCH --job-name=olmo-60m-ft-8k
#SBATCH --ntasks=1
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/%x_%j.out
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# ================= 环境配置 =================

# 关键环境路径
export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 

# 指定 Python 解释器
PYTHON_EXE="/home/qijunrong/anaconda3/bin/python"

# ================= 路径与参数配置 =================
# 1. 基础模型路径
PRETRAINED_CKPT="/data/qijunrong/03-proj/PE/checkpoints/olmo-60m-ScaledRoPE-flash-len2048-1.5B/final_model.pt"

# 2. 数据集根目录 (train.py 会自动寻找下级目录)
DATA_PATH="/data/qijunrong/03-proj/PE"

# 3. 配置文件
CONFIG_PATH="./configs/olmo_60m.yaml"

# 4. 微调超参数
RUN_NAME="olmo-60m-Stage2-FT-8k-Sigma85"
TARGET_SEQ_LEN=8192
SIGMA=85.0
DECAY_FUNC="exp"

# 5. 训练步数与批次 (8k 长度非常吃显存，MicroBS 设为 4 或 2)
MAX_STEPS=1000
GLOBAL_BS=128
MICRO_BS=4 

# ================= 启动微调 =================
echo "===================================================================="
echo "Starting Stage 2 Fine-tuning (2k -> 8k)"
echo "Base Model: $PRETRAINED_CKPT"
echo "Target Length: $TARGET_SEQ_LEN"
echo "Method: Scaled RoPE (Sigma=$SIGMA, $DECAY_FUNC)"
echo "Steps: $MAX_STEPS (Rapid Adaptation)"
echo "===================================================================="

$PYTHON_EXE train.py \
    --config $CONFIG_PATH \
    --run_name "$RUN_NAME" \
    --finetune_from "$PRETRAINED_CKPT" \
    --dataset_path "$DATA_PATH" \
    --position_embedding rope \
    --use_scaled_rope1 \
    --scaled_rope_sigma $SIGMA \
    --decay_func "$DECAY_FUNC" \
    --train_max_sequence_length $TARGET_SEQ_LEN \
    --val_max_sequence_length $TARGET_SEQ_LEN \
    --train_size 1000000 \
    --val_size 2000 \
    --batch_size $GLOBAL_BS \
    --micro_batch_size $MICRO_BS \
    --max_steps $MAX_STEPS \
    --save_interval 500 \
    --eval_interval 100

echo "Finished Fine-tuning: $RUN_NAME"
echo "===================================================================="