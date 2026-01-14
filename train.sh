#!/bin/bash -x

#SBATCH --job-name=olmo-60m-one-sigma
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




# echo "Baseline"
# 2025-12-4 测试了长度为2048、训练数据量为100万、训练步长
# 2026-1-13 test len = 2048, token = 50w, max_step=60000
# /home/qijunrong/anaconda3/bin/python train.py \
#    --config ./configs/olmo_60m.yaml \
#    --run_name "olmo-60m-RoPEbaseline-singlegpu-len2048-50w" \
#    --position_embedding rope \
#    --train_max_sequence_length 2048 \
#    --train_size 500000 \
#    --val_size 10000 \
#    --batch_size 8 \
#    --micro_batch_size 4 \
#    --max_steps 60000 \
#    --save_interval 5000 \
#    --log_interval 200 \
#    --seed 6198




# 1. 训练数据量调整为 50W (验证小样本下的 Inductive Bias)
# TRAIN_SIZE=500000
# 2026-1-13 to 2026-1-14 sigma = 85.0, order = 1.5, 1.1, 1.7, sequence_num = 500k
# declare -a SIGMAS=(85.0)
# declare -a ORDERS=(1.5 1.1 1.7)
#for sigma in "${SIGMAS[@]}"; do
#   for order in "${ORDERS[@]}"; do
    
#        run_name="olmo-60m-ScaledRoPE-segmented-s${sigma}-o${order}-len2048-50w"
        
#        echo "===================================================================="
#        echo "Starting run: $run_name"
#        echo "Configuration: decay_func=segmented, sigma=$sigma, decay_order=$order"
#        echo "Data Size: $TRAIN_SIZE tokens"
#        echo "===================================================================="

#        /home/qijunrong/anaconda3/bin/python train.py \
#            --config ./configs/olmo_60m.yaml \
#            --position_embedding rope \
#            --use_scaled_rope1 \
#            --scaled_rope_sigma $sigma \
#            --decay_func "segmented" \
#            --decay_order $order \
#            --run_name "$run_name" \
#            --train_max_sequence_length 2048 \
#            --train_size $TRAIN_SIZE \
#            --val_size 10000 \
#            --batch_size 8 \
#            --micro_batch_size 4 \
#            --max_steps 60000 \
#            --save_interval 5000 \
#            --log_interval 200 \
#            --seed 6198

#        echo "Finished run: $run_name"
#        echo "--------------------------------------------------------------------"
#        echo ""
#    done
#done


# 2025-12-4 测试了declare -a SIGMAS=(30.0 60.0 70.0 80.0 100.0)
# 2026-1-13 测试segment SIGMAS=(80.0 85.0 90.0) ORDERS=(2.0 4.0 8.0 16.0)
# declare -a SIGMAS=(80.0 85.0 90.0)
# declare -a ORDERS=(2.0 4.0 8.0 16.0)
# for sigma in "${SIGMAS[@]}"; do
#    for order in "${ORDERS[@]}"; do
    
#        run_name="olmo-60m-ScaledRoPE-segmented-s${sigma}-o${order}-len2048-50w"
        
#        echo "===================================================================="
#        echo "Starting run: $run_name"
#        echo "Configuration: decay_func=segmented, sigma=$sigma, decay_order=$order"
#        echo "Data Size: $TRAIN_SIZE tokens"
#        echo "===================================================================="

#        /home/qijunrong/anaconda3/bin/python train.py \
#            --config ./configs/olmo_60m.yaml \
#            --position_embedding rope \
#            --use_scaled_rope1 \
#            --scaled_rope_sigma $sigma \
#            --decay_func "segmented" \
#            --decay_order $order \
#            --run_name "$run_name" \
#            --train_max_sequence_length 2048 \
#            --train_size $TRAIN_SIZE \
#            --val_size 10000 \
#            --batch_size 8 \
#            --micro_batch_size 4 \
#            --max_steps 60000 \
#            --save_interval 5000 \
#            --log_interval 200 \
#            --seed 6198

#        echo "Finished run: $run_name"
#        echo "--------------------------------------------------------------------"
#        echo ""
#    done
#done



# test exp, sigma = 85.0, len = 2048, every 1000step
# ================= 配置区域 =================

SIGMA=85.0
DECAY_FUNC="exp"
SEQ_LEN=2048

# single RTX 5090
# Global Batch Size: 128
GLOBAL_BS=128
# Micro Batch Size: 8
MICRO_BS=8

# debug
MAX_STEPS=40000
TRAIN_SIZE=5000000
# TRAIN_SIZE=5000000

# run_name="olmo-60m-ScaledRoPE-${DECAY_FUNC}-s${SIGMA}-len${SEQ_LEN}-1B_tokens-BS${GLOBAL_BS}"
run_name="olmo-60m-ScaledRoPE-flash-len2048-1B"

echo "===================================================================="
echo "Starting run: $run_name"
echo "Method: $DECAY_FUNC decay, Sigma: $SIGMA"
echo "Batch Size: Global=$GLOBAL_BS, Micro=$MICRO_BS"
echo "Steps: $MAX_STEPS (Approx 1B Tokens)"
echo "===================================================================="

/home/qijunrong/anaconda3/bin/python train.py \
    --config ./configs/olmo_60m.yaml \
    --run_name "$run_name" \
    --position_embedding rope \
    --use_scaled_rope1 \
    --scaled_rope_sigma $SIGMA \
    --decay_func "$DECAY_FUNC" \
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

echo "Finished run: $run_name"
