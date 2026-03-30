#!/bin/bash

#SBATCH --job-name=exp2-c4-revlocal-60m
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/revlocal_60M/tmp_%j.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/revlocal_60M/tmp_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G

export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED=1
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_c4_60m_syn.py" 

ROOT_DIR="/data/qijunrong/03-proj/PE"
LOG_DIR="${ROOT_DIR}/logs/revlocal_60M/$(date +%Y%m%d)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp2/revlocal_60M"
WANDB_DIR="${ROOT_DIR}/wandb/offline/revlocal_60M"
C4_DATA_ROOT="${ROOT_DIR}"
LOCAL_TOKENIZER="${ROOT_DIR}/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR

# === 日志重定向与清理机制 ===
JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/revlocal_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/revlocal_${JOB_ID}.err"

exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f /data/qijunrong/03-proj/PE/logs/revlocal_60M/tmp_${JOB_ID}.out
    rm -f /data/qijunrong/03-proj/PE/logs/revlocal_60M/tmp_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"

# === 训练常量配置 ===
MODELS=("60M")
GLOBAL_BS=64
SEEDS=(6198)

# C4 配置: 300M tokens
MAX_TOKENS=300000000
TRAIN_SAMPLES=2000000
VAL_SAMPLES=10000

# === DEBUG 配置 ===
DEBUG_STEPS="" 
# DEBUG_STEPS=200

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS --lr 6e-4"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS --lr 6e-4"
fi

# === 调参网格 (Grid Search) ===
LENGTHS=(1024)
WINDOW_SIZES=(128)
SIGMAS=(200.0 300.0)

# 关键配置：num_local_layers=-4 表示后4层使用局部注意力 (60M共8层，即层4,5,6,7)
NUM_LOCAL_LAYERS=-4

echo ">>> [Config] Using REVERSE local attention: last 4 layers (layers 4-7) will use local attention"

# =========================================================
# [核心修改]：根据实验类型动态调整 MBS
# =========================================================
get_mbs() {
    local m_size=$1
    local seq_len=$2
    local exp_type=$3
    
    # 基础MBS配置
    local mbs=16
    if [ "$m_size" == "60M" ]; then mbs=16; fi
    
    # 长序列统一下调
    if [ "$seq_len" -ge 2048 ]; then mbs=$((mbs / 2)); fi

    # 局部注意力特殊处理（关闭 FlashAttention，显存开销大）
    # 注意：Reverse Local 同样需要关闭 FlashAttention
    if [[ "$exp_type" == "G2L_RoPE" ]] || [[ "$exp_type" == "G2L_HIPE" ]]; then
        if [ "$seq_len" -ge 2048 ]; then
            mbs=4
        fi
    fi

    echo $mbs
}

run_exp() {
    local TYPE=$1; local M_SIZE=$2; local SEQ_LEN=$3; local EXTRA_ARGS=$4
    
    local CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN $TYPE)
    
    # 清理参数名，生成友好的 RUN_ID
    local CLEAN_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--//g' | sed 's/ /_/g' | sed 's/_local_window_size_/W/g' | sed 's/_num_local_layers_//g' | sed 's/_use_scaled_rope//g' | sed 's/_sigma_/S/g')
    
    local RUN_ID="revlocal_${TYPE}_L${SEQ_LEN}_${CLEAN_ARGS}"
    local OUT_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
    
    echo ">>> Running: [${TYPE}] Model=${M_SIZE} | Len=${SEQ_LEN} | MBS=${CUR_MICRO_BS} | Args: ${EXTRA_ARGS}"
    
    $PYTHON_BIN $SCRIPT \
        --output_dir $OUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
        --dataset_path $C4_DATA_ROOT --local_tokenizer_path $LOCAL_TOKENIZER \
        --wandb_dir $WANDB_DIR --wandb_mode $WANDB_MODE \
        --train_size $TRAIN_SAMPLES --val_size $VAL_SAMPLES \
        --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
        --seed ${SEEDS[0]} $LIMIT_ARGS $EXTRA_ARGS
}

echo ">>> Starting 60M Reverse Local Attention Experiments..."

for M in "${MODELS[@]}"; do
    for L in "${LENGTHS[@]}"; do
        # 1. Baseline: 纯 RoPE 全局注意力 (已有，可注释掉或保留用于对比)
        # run_exp "Base" $M $L ""
        
        # 2. 【新实验A】Global-to-Local RoPE: 前4层全局+标准RoPE，后4层局部+标准RoPE
        # num_local_layers=-4 表示后4层使用局部注意力
        for W in "${WINDOW_SIZES[@]}"; do
            run_exp "G2L_RoPE" $M $L "--local_window_size $W --num_local_layers $NUM_LOCAL_LAYERS"
        done
        
        # 3. 【新实验B】Global-to-Local HIPE: 前4层全局+标准RoPE，后4层局部+HIPE
        # 注意：HIPE 的 threshold 设为3，确保前4层(0,1,2,3)用标准RoPE，后4层(4,5,6,7)用HIPE
        # 同时 num_local_layers=-4 确保后4层用局部注意力
        for W in "${WINDOW_SIZES[@]}"; do
            for S in "${SIGMAS[@]}"; do
                run_exp "G2L_HIPE" $M $L "--local_window_size $W --num_local_layers $NUM_LOCAL_LAYERS --use_scaled_rope --sigma $S --rope_scaling_threshold 3"
            done
        done
        
    done
done

echo ">>> All Reverse Local Attention Experiments Completed."
