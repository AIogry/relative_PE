#!/bin/bash -x

#SBATCH --job-name=olmo-60m-baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/%x_%j.out
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mem=128G

export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ================= 配置区域 =================
# 1. 保持与实验组完全一致的 Batch Size
GLOBAL_BS=128
MICRO_BS=8

# 2. 保持一致的数据量
TRAIN_SIZE=6000000
SEQ_LEN=2048

# 3. 步数计算 (重要修正)
# 5,000,000 / 128 = 39,062.5
# 为了防止跑到 40000 步时报 "IndexError/StopIteration" (数据耗尽)，
# 我们这里设置为 39100，这足以跑完一轮数据。
MAX_STEPS=6000

run_name="olmo-60m-Baseline-RoPE-flash-len2048-1B"

echo "===================================================================="
echo "Starting Baseline Run: $run_name"
echo "Method: Standard RoPE (No Scaling)"
echo "Batch Size: Global=$GLOBAL_BS, Micro=$MICRO_BS"
echo "Steps: $MAX_STEPS (Matches 5000k data)"
echo "===================================================================="

/home/qijunrong/anaconda3/bin/python train.py \
    --config ./configs/olmo_60m.yaml \
    --run_name "$run_name" \
    --position_embedding rope \
    --train_max_sequence_length $SEQ_LEN \
    --train_size $TRAIN_SIZE \
    --val_size 10000 \
    --batch_size $GLOBAL_BS \
    --micro_batch_size $MICRO_BS \
    --max_steps $MAX_STEPS \
    --save_interval 5000 \
    --log_interval 200 \
    --eval_interval 1000 \
    --seed 6198

# 注意：我去掉了以下参数，使其变为纯 RoPE：
# --use_scaled_rope1 
# --scaled_rope_sigma ...
# --decay_func ...

echo "Finished run: $run_name"