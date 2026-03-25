#!/bin/bash

#SBATCH --job-name=exp2-c4-300M-extrap-base
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/exp2_c4_300M/tmp_extrap_base_%j.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/exp2_c4_300M/tmp_extrap_base_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
export WANDB_MODE="offline"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_c4full_extrap.py" 

# === 路径配置 ===
ROOT_DIR="/data/qijunrong/03-proj/PE"
LOG_DIR="${ROOT_DIR}/logs/exp2_c4_300M/$(date +%Y%m%d)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp2/c4_300M/base_extrap"
WANDB_DIR="${ROOT_DIR}/wandb/offline/exp2_c4_300M"

C4_DATA_ROOT="${ROOT_DIR}"
ARXIV_VAL_ROOT="${ROOT_DIR}/arxiv_data/arxiv_validation" # 新增 Arxiv 验证集路径
LOCAL_TOKENIZER="${ROOT_DIR}/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR

JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/exp2_base_extrap_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/exp2_base_extrap_${JOB_ID}.err"

exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f /data/qijunrong/03-proj/PE/logs/exp2_c4_300M/tmp_extrap_base_${JOB_ID}.out
    rm -f /data/qijunrong/03-proj/PE/logs/exp2_c4_300M/tmp_extrap_base_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"

# === 全局配置 ===
GLOBAL_BS=64
SEEDS=(6198)

# C4 专享大数据量配置
MAX_TOKENS=1000000000 # 1B token
TRAIN_SAMPLES=5000000 # 5M samples
VAL_SAMPLES=10000

# 外推验证配置
SAVE_INTERVAL=1000
EXTRAP_INTERVAL=2000

# === DEBUG 配置 ===
DEBUG_STEPS="" 
# DEBUG_STEPS=200
if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS --lr 3e-4"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS --lr 3e-4"
fi

MODELS=("300M")
# [核心修改] 严格锁定为 512，外推交给 Python 脚本的验证逻辑
LENGTHS=(512) 

get_mbs() {
    echo 16 # 512 长度下 300M 模型 MBS 设为 16
}

run_baseline_experiment() {
    local baseline_type=$1
    local extra_args=$2
    
    echo -e "\n>>> [BATCH START] Running $baseline_type on C4 for Extrapolation..."
    
    for SEED in "${SEEDS[@]}"; do
        for M_SIZE in "${MODELS[@]}"; do
            for SEQ_LEN in "${LENGTHS[@]}"; do
                CUR_MICRO_BS=$(get_mbs)
                RUN_ID="c4_extrap_${baseline_type}_${M_SIZE}_L${SEQ_LEN}"
                OUTPUT_DIR="${CHECKPOINT_ROOT}/${M_SIZE}/${RUN_ID}/seed_${SEED}"
                
                echo ">>> [$baseline_type] Model: $M_SIZE | Len: $SEQ_LEN | MBS: $CUR_MICRO_BS | SEED: $SEED"
                
                $PYTHON_BIN $SCRIPT \
                    --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
                    --dataset_path $C4_DATA_ROOT --arxiv_val_path $ARXIV_VAL_ROOT --local_tokenizer_path $LOCAL_TOKENIZER \
                    --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
                    --train_size $TRAIN_SAMPLES --val_size $VAL_SAMPLES \
                    --save_interval $SAVE_INTERVAL --extrap_eval_interval $EXTRAP_INTERVAL \
                    --wandb_dir $WANDB_DIR --wandb_mode $WANDB_MODE \
                    $extra_args \
                    $LIMIT_ARGS --seed $SEED
                
                if [ $? -ne 0 ]; then
                    echo ">>> [ERROR] $baseline_type实验失败！Model: $M_SIZE, Len: $SEQ_LEN, SEED: $SEED"
                fi
            done
        done
    done
}

# 运行纯 RoPE 训练，Python 代码中每 2000 步会为其挂载 YaRN 测外推
run_baseline_experiment "rope" ""

echo -e "\n>>> All C4 300M Extrap Baselines Completed."