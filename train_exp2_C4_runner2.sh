#!/bin/bash

#SBATCH --job-name=c4-biograd-search
#SBATCH --output=./logs/c4_biograd_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
# [关键] 使用 C4 版本的 Python 脚本
SCRIPT="train_exp2_C4_layerwise2.py" 

# === 路径配置 ===
CHECKPOINT_ROOT="/data/qijunrong/03-proj/PE/checkpoints_c4_biograd"
C4_DATA_ROOT="/data/qijunrong/03-proj/PE"
LOCAL_TOKENIZER="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

mkdir -p $CHECKPOINT_ROOT
mkdir -p ./logs

# === 全局配置 ===
GLOBAL_BS=64
SEED=6198
MAX_TOKENS=200000000 # 1亿 Token

# C4 数据量控制 (与 Baseline 保持一致)
TRAIN_SAMPLES=2000000
VAL_SAMPLES=10000

# === DEBUG 开关 ===
DEBUG_STEPS="" 
# DEBUG_STEPS=100

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS"
fi

# ============================================================
# [关键] 定义参数搜索空间
# ============================================================

# 1. Sigma 列表 (建议覆盖从小到大的范围)
# SIGMAS=(50.0 100.0 200.0 300.0 500.0)
# SIGMAS=(700.0 1000.0 800.0 1200.0 600.0)
SIGMAS=(200.0 500.0 700.0 1000.0 100.0 50.0)

# 2. Threshold 列表 (Bio-Gradient 开始的层数)
THRESHOLDS=(3)  # 2和4先不跑

# 3. 模型与长度列表
MODELS=("20M" "60M") # "20M")
LENGTHS=(2048 1024 512)

# ============================================================
# 函数：执行单次 Bio-Gradient 实验
# ============================================================
run_experiment() {
    local M_SIZE=$1
    local SEQ_LEN=$2
    local SIGMA=$3
    local THR=$4
    
    # --- 动态计算 Micro Batch Size ---
    local CUR_MICRO_BS=16
    if [ "$M_SIZE" == "60M" ]; then CUR_MICRO_BS=8; fi
    if [ "$SEQ_LEN" -ge 2048 ]; then CUR_MICRO_BS=$((CUR_MICRO_BS / 2)); fi

    # 生成 Run ID (包含关键参数)
    local RUN_ID="c4_biograd_${M_SIZE}_L${SEQ_LEN}_sig${SIGMA}_thr${THR}"
    if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
    
    # 输出目录 (带时间戳防止覆盖)
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}_${TIMESTAMP}"

    echo "----------------------------------------------------------------"
    echo ">>> [START] Model: $M_SIZE | Len: $SEQ_LEN | Sigma: $SIGMA | Thr: $THR"
    echo ">>> Dataset: C4 | Train Samples: $TRAIN_SAMPLES"
    echo ">>> Micro BS: $CUR_MICRO_BS | Output: $OUTPUT_DIR"
    
    $PYTHON_BIN $SCRIPT \
        --output_dir $OUTPUT_DIR \
        --run_id $RUN_ID \
        --model_size $M_SIZE \
        --dataset_path $C4_DATA_ROOT \
        --local_tokenizer_path $LOCAL_TOKENIZER \
        --seq_len $SEQ_LEN \
        --global_batch_size $GLOBAL_BS \
        --micro_batch_size $CUR_MICRO_BS \
        --train_size $TRAIN_SAMPLES \
        --val_size $VAL_SAMPLES \
        --use_scaled_rope \
        --sigma $SIGMA \
        --rope_scaling_threshold $THR \
        $LIMIT_ARGS \
        --seed $SEED

    if [ $? -ne 0 ]; then
        echo ">>> [ERROR] Run Failed: $RUN_ID"
    else
        echo ">>> [SUCCESS] Run Finished: $RUN_ID"
    fi
}

# ============================================================
# 主循环：遍历所有组合
# ============================================================
echo "========================================================"
echo ">>> [BATCH START] Bio-Gradient Hyperparameter Search on C4"
echo "========================================================"

for M_SIZE in "${MODELS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do
        for SIGMA in "${SIGMAS[@]}"; do
            for THR in "${THRESHOLDS[@]}"; do
                
                # 执行实验
                run_experiment $M_SIZE $SEQ_LEN $SIGMA $THR
                
            done
        done
    done
done

echo ">>> All C4 Bio-Gradient Experiments Completed."