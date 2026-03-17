#!/bin/bash

#SBATCH --job-name=exp2-varlensize
#SBATCH --output=./logs/exp2_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_layerwise.py"

# === 路径配置 ===
CHECKPOINT_ROOT="/data/qijunrong/03-proj/PE/checkpoints_variable_len"
LOCAL_DATA="/data/qijunrong/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

mkdir -p $CHECKPOINT_ROOT
mkdir -p ./logs

# === 全局配置 ===
GLOBAL_BS=64
SEED=6198
MAX_TOKENS=100000000 # 保持 1亿 Token 总量

# Debug 开关
DEBUG_STEPS="" 
# DEBUG_STEPS=100

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG] Running for $DEBUG_STEPS steps only."
    TRAIN_ARGS="--max_train_steps $DEBUG_STEPS"
else
    TRAIN_ARGS="--max_tokens $MAX_TOKENS"
fi

# ============================================================
# 定义参数空间
# ============================================================

# 1. Sigma 列表
# SIGMAS=(80.0 1.0 10.0 1000.0 0.5 700.0 70.0)   用来测试60m len=2048的
SIGMAS=(200.0 300.0 500.0 700.0 1000.0)
# SIGMAS=(100.0 500.0)

# 2. Threshold 列表
THRESHOLDS=(2 3 4)

# ============================================================
# 函数：执行单次训练
# ============================================================
run_experiment() {
    local M_SIZE=$1
    local SEQ_LEN=$2
    local SIGMA=$3
    local THR=$4
    
    # --- 自动调整 Micro Batch Size 防止 OOM ---
    # 逻辑：序列变长或模型变大时，减小 Micro BS
    local CUR_MICRO_BS=16  # 默认基准 (20M, 512/1024)

    # 1. 如果是 60M 模型，基准减半 (因为 d_model 翻倍，显存压力大)
    if [ "$M_SIZE" == "60M" ]; then
        CUR_MICRO_BS=8
    fi

    # 2. 如果序列长度 >= 2048，基准再减半
    if [ "$SEQ_LEN" -ge 2048 ]; then
        CUR_MICRO_BS=$((CUR_MICRO_BS / 2))
    fi

    # 生成 Run ID
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local RUN_ID="${M_SIZE}_L${SEQ_LEN}_sig${SIGMA}_thr${THR}"
    local OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}_${TIMESTAMP}"

    echo "----------------------------------------------------------------"
    echo ">>> [START] Model: $M_SIZE | Len: $SEQ_LEN | Sigma: $SIGMA | Thr: $THR"
    echo ">>> Micro BS: $CUR_MICRO_BS | Output: $OUTPUT_DIR"
    
    $PYTHON_BIN $SCRIPT \
        --output_dir $OUTPUT_DIR \
        --run_id $RUN_ID \
        --model_size $M_SIZE \
        --local_data_path $LOCAL_DATA \
        --local_tokenizer_path $LOCAL_TOKENIZER \
        --seq_len $SEQ_LEN \
        --global_batch_size $GLOBAL_BS \
        --micro_batch_size $CUR_MICRO_BS \
        --seed $SEED \
        --use_scaled_rope \
        --sigma $SIGMA \
        --rope_scaling_threshold $THR \
        $TRAIN_ARGS

    if [ $? -ne 0 ]; then
        echo ">>> [ERROR] Run Failed!"
    fi
}



# ============================================================
# 阶段 1: 20M 模型实验 (Length 1024, 2048)
# ============================================================
# MODEL="20M"
# LENGTHS=(1024)

# echo ">>> Starting PHASE 1: 20M Model Experiments..."

# for len in "${LENGTHS[@]}"; do
    # 仅跑 Bio-Gradient
#    for sigma in "${SIGMAS[@]}"; do
#        for thr in "${THRESHOLDS[@]}"; do
#            run_experiment $MODEL $len $sigma $thr
#        done
#    done
# done



echo ">>> All Phase 1 & 2 Experiments Finished."