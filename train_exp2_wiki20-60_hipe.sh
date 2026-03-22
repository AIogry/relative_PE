#!/bin/bash

#SBATCH --job-name=exp2-wiki20-60-hipe
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/exp2_wiki20-60/tmp_hipe_%j.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/exp2_wiki20-60/tmp_hipe_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# === 环境配置（与base.sh对齐） ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
export WANDB_MODE="offline"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_wiki20-60.py" 

# === 路径配置（与base.sh严格对应） ===
ROOT_DIR="/data/qijunrong/03-proj/PE"
# 日志目录：按日期分层，与base.sh保持一致
LOG_DIR="${ROOT_DIR}/logs/exp2_wiki20-60/$(date +%Y%m%d)"
# Checkpoint根目录：与base.sh同层级（base/hipe）
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp2/wiki20-60/hipe"
# Wandb离线目录（与base.sh共享）
WANDB_DIR="${ROOT_DIR}/wandb/offline/exp2_wiki20-60"

# 数据/Tokenizer路径（完全复用base.sh的定义）
LOCAL_DATA="/data/qijunrong/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

# 创建所有必要目录（容错处理，与base.sh一致）
mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR

# === 日志重定向（与base.sh一致） ===
JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/hipe_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/hipe_${JOB_ID}.err"
exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

# === 清理临时日志（与base.sh一致） ===
function cleanup {
    rm -f /data/qijunrong/03-proj/PE/logs/exp2_wiki20-60/tmp_hipe_${JOB_ID}.out
    rm -f /data/qijunrong/03-proj/PE/logs/exp2_wiki20-60/tmp_hipe_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"

# === 全局配置（与base.sh对齐） ===
GLOBAL_BS=64
SEEDS=(6198 1024 7 568 3427)     # 5个随机种子（复用base的种子）
MAX_TOKENS=100000000 # 1亿 Token（与base一致）

# === DEBUG 配置（与base.sh一致） ===
DEBUG_STEPS="" 
# DEBUG_STEPS=100  # <--- 取消注释开启调试

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS"
fi

# ============================================================
# 定义参数空间（保留HIPE核心参数，仅调整格式）
# ============================================================
# 1. Sigma 列表
SIGMAS=(50.0 100.0 200.0 500.0 700.0 1000.0)
# 2. Threshold 列表
THRESHOLDS=(3)
# 3. 模型/序列长度（与base.sh完全对齐）
MODELS=("20M" "60M")
LENGTHS=(512 1024 2048)

# ============================================================
# 核心函数：计算Micro Batch Size（复用base.sh的逻辑+兼容HIPE）
# 参数：
# $1: 模型大小 (20M/60M)
# $2: 序列长度 (512/1024/2048)
# ============================================================
get_mbs() {
    local m_size=$1
    local seq_len=$2
    
    # 基础MBS配置（与base.sh完全一致）
    local mbs=32
    if [ "$m_size" == "60M" ]; then mbs=16; fi
    if [ "$seq_len" -ge 2048 ]; then mbs=$((mbs / 2)); fi

    echo $mbs
}

# ============================================================
# 核心函数：执行单次HIPE实验（复用base.sh风格）
# ============================================================
run_experiment() {
    local M_SIZE=$1
    local SEQ_LEN=$2
    local SIGMA=$3
    local THR=$4
    local SEED=$5
    
    # --- 自动调整 Micro Batch Size（复用base的get_mbs函数）---
    CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)

    # --- 生成RUN_ID/输出目录（与base.sh风格对齐，移除TIMESTAMP）---
    RUN_ID="hipe_${M_SIZE}_L${SEQ_LEN}_sig${SIGMA}_thr${THR}"
    OUTPUT_DIR="${CHECKPOINT_ROOT}/${M_SIZE}/${RUN_ID}/seed_${SEED}"

    # --- 打印实验信息（与base.sh格式一致）---
    echo -e "\n>>> [HIPE] Model: $M_SIZE | Len: $SEQ_LEN | Sigma: $SIGMA | Thr: $THR | MBS: $CUR_MICRO_BS | SEED: $SEED"
    echo ">>> Output Dir: $OUTPUT_DIR"
    
    # --- 执行训练命令（参数格式与base.sh对齐）---
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
        $LIMIT_ARGS

    # --- 错误检查（与base.sh一致）---
    if [ $? -ne 0 ]; then
        echo ">>> [ERROR] HIPE实验失败！Model: $M_SIZE, Len: $SEQ_LEN, Sigma: $SIGMA, SEED: $SEED"
    fi
}

# ============================================================
# 批量执行HIPE实验（分阶段，与base.sh风格对齐）
# ============================================================
echo -e "\n>>> [BATCH START] Running HIPE experiments..."

# 阶段1: 20M 模型
echo -e "\n>>> PHASE 1: 20M Model Experiments..."
for SEED in "${SEEDS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do
        for SIGMA in "${SIGMAS[@]}"; do
            for THR in "${THRESHOLDS[@]}"; do
                run_experiment "20M" $SEQ_LEN $SIGMA $THR $SEED
            done
        done
    done
done

# 阶段2: 60M 模型
echo -e "\n>>> PHASE 2: 60M Model Experiments..."
for SEED in "${SEEDS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do
        for SIGMA in "${SIGMAS[@]}"; do
            for THR in "${THRESHOLDS[@]}"; do
                run_experiment "60M" $SEQ_LEN $SIGMA $THR $SEED
            done
        done
    done
done

echo -e "\n>>> All HIPE Experiments Completed."