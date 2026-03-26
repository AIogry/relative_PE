#!/bin/bash

#SBATCH --job-name=exp2-c4-300M-learnable-sigma
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/exp2_c4_300M/tmp_learnable_sigma_%j.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/exp2_c4_300M/tmp_learnable_sigma_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED=1
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_c4_300m_learnable_sigma.py" 

# === 路径配置 ===
ROOT_DIR="/data/qijunrong/03-proj/PE"
LOG_DIR="${ROOT_DIR}/logs/exp2_c4_300M/$(date +%Y%m%d)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp2/c4_300M/learnable_sigma"
WANDB_DIR="${ROOT_DIR}/wandb/offline/exp2_c4_300M"

C4_DATA_ROOT="${ROOT_DIR}"
LOCAL_TOKENIZER="${ROOT_DIR}/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR

JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/learnable_sigma_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/learnable_sigma_${JOB_ID}.err"

exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f /data/qijunrong/03-proj/PE/logs/exp2_c4_300M/tmp_learnable_sigma_${JOB_ID}.out
    rm -f /data/qijunrong/03-proj/PE/logs/exp2_c4_300M/tmp_learnable_sigma_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"

# === 全局配置 ===
GLOBAL_BS=64
SEEDS=(6198) # 1024 7 568 3427)

# C4 专享大数据量配置
MAX_TOKENS=1000000000 # 1B token
TRAIN_SAMPLES=5000000 # 10M samples
VAL_SAMPLES=10000
SAVE_INTERVAL=100

# === DEBUG 配置 ===
DEBUG_STEPS="" 
# DEBUG_STEPS=50

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS --lr 3e-4"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS --lr 3e-4"
fi

MODELS=("300M")
THRESHOLDS=(7 9)
# 一一对应的length和sigma初值
LENGTHS=(512 1024) # 2048)  # 512 1024
SIGMAS=(200 500) # 700) # 200 500

# ============================================================
# 核心函数1：计算Micro Batch Size (300M 专属)
# ============================================================
get_mbs() {
    local m_size=$1
    local seq_len=$2
    
    local mbs=8
    if [ "$seq_len" -ge 2048 ]; then mbs=4; fi      # 普通的都可以运行

    echo $mbs
}

# ============================================================
# 核心函数2：运行可学习Sigma实验
# ============================================================
run_learnable_sigma_experiment() {
    echo -e "\n>>> [BATCH START] Running Learnable Sigma on C4..."
    
    for SEED in "${SEEDS[@]}"; do
        for M_SIZE in "${MODELS[@]}"; do
            # 一一对应length和sigma
            for i in "${!LENGTHS[@]}"; do
                SEQ_LEN=${LENGTHS[$i]}
                SIGMA=${SIGMAS[$i]}
                
                # 对每个threshold运行实验
                for THR in "${THRESHOLDS[@]}"; do
                    CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
                    RUN_ID="c4_learnable_sigma_${M_SIZE}_L${SEQ_LEN}_sigma${SIGMA}_thr${THR}"
                    OUTPUT_DIR="${CHECKPOINT_ROOT}/${M_SIZE}/${RUN_ID}/seed_${SEED}"
                    
                    echo ">>> [Learnable Sigma] Model: $M_SIZE | Len: $SEQ_LEN | MBS: $CUR_MICRO_BS | Sigma: $SIGMA | Thr: $THR | SEED: $SEED"
                    
                    $PYTHON_BIN $SCRIPT \
                        --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
                        --dataset_path $C4_DATA_ROOT --local_tokenizer_path $LOCAL_TOKENIZER \
                        --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
                        --train_size $TRAIN_SAMPLES --val_size $VAL_SAMPLES \
                        --wandb_dir $WANDB_DIR --wandb_mode $WANDB_MODE \
                        --use_scaled_rope --sigma $SIGMA --learnable_sigma \
                        --rope_scaling_threshold $THR \
                        --save_interval $SAVE_INTERVAL \
                        $LIMIT_ARGS --seed $SEED
                    if [ $? -ne 0 ]; then
                        echo ">>> [ERROR] Learnable Sigma实验失败！Model: $M_SIZE, Len: $SEQ_LEN, Sigma: $SIGMA, Thr: $THR, SEED: $SEED"
                    fi
                done

            done
        done
    done
}

run_learnable_sigma_experiment

echo -e "\n>>> All C4 300M Learnable Sigma Experiments Completed."