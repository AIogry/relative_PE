#!/bin/bash

#SBATCH --job-name=exp3-sst2-rope-finetune
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/exp3/sst2/300m/tmp_sst2_rope_%j.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/exp3/sst2/300m/tmp_sst2_rope_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="finetune_sst2.py"

# === 路径配置 ===
ROOT_DIR="/data/qijunrong/03-proj/PE"
LOG_DIR="${ROOT_DIR}/logs/exp3/sst2/300m/$(date +%Y%m%d)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp3/sst2/300m/base"
WANDB_DIR="${ROOT_DIR}/wandb/offline/sst2"
SST2_DATA_PATH="${ROOT_DIR}/sst2_data"

# 模型路径配置
ROPE_512_PATH="${ROOT_DIR}/checkpoints_exp2/c4_300M/base/300M/c4_baseline_rope_300M_L512/seed_6198/model.pt"
ROPE_2048_PATH="${ROOT_DIR}/checkpoints_exp2/c4_300M/base/300M/c4_baseline_rope_300M_L2048/seed_6198/model.pt"

LOCAL_TOKENIZER="${ROOT_DIR}/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR

JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/sst2_rope_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/sst2_rope_${JOB_ID}.err"

exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f ${ROOT_DIR}/logs/exp3/sst2/tmp_sst2_rope_${JOB_ID}.out
    rm -f ${ROOT_DIR}/logs/exp3/sst2/tmp_sst2_rope_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"

# === 全局配置 ===
MODEL_SIZE="300M"
SEEDS=(6198)
NUM_EPOCHS=5          # SST-2通常5个epoch足够
MAX_LENGTH=128
TRAIN_BS=16
EVAL_BS=64
LR=5e-4
CLASSIFIER_LR=1e-3
LORA_LR=5e-4
EVAL_INTERVAL=100     # SST-2训练更快，评估间隔可以更大
SAVE_INTERVAL=500
GRAD_ACCUM_STEPS=2
EARLY_STOPPING_PATIENCE=5  # 默认早停耐心，实际会根据shot动态调整

# Few-shot 设置: -1=full, 其他为具体样本数
# 包含更多shot设置以观察性能随数据量变化
SHOT_SETTINGS=(-1 100 500 1000 2000 5000 10000)

# === LoRA 配置 ===
LORA_RANKS=(8)
LORA_ALPHA=32

# === DEBUG 配置 ===
DEBUG_SHOT=""
# DEBUG_SHOT=100

if [ -n "$DEBUG_SHOT" ]; then
    echo ">>> [DEBUG MODE] Shot: $DEBUG_SHOT"
    SHOT_SETTINGS=($DEBUG_SHOT)
fi

FAILED_COUNT=0

# ============================================================
# 核心函数：RoPE + LoRA (推荐)
# ============================================================
run_rope_lora() {
    local base_model=$1
    local seq_len=$2
    local few_shot=$3
    local lora_rank=$4
    local seed=$5
    
    local shot_str=$([ "$few_shot" -lt 0 ] && echo "full" || echo "shot${few_shot}")
    local run_id="rope_${seq_len}_lora${lora_rank}_${shot_str}_seed${seed}"
    local output_dir="${CHECKPOINT_ROOT}/${seq_len}/lora${lora_rank}/${shot_str}/seed_${seed}"
    
    # 动态早停：full数据集需要更多耐心
    local patience=$EARLY_STOPPING_PATIENCE
    if [ "$few_shot" -lt 0 ]; then
        patience=15  # full数据集增加到15
    elif [ "$few_shot" -ge 5000 ]; then
        patience=10  # 大数据集也增加
    fi
    echo ">>> Early stopping patience: $patience"
    
    echo -e "\n>>> [RoPE + LoRA - SST-2] Seq: $seq_len | Rank: $lora_rank | Shot: $shot_str | Seed: $seed"
    
    $PYTHON_BIN $SCRIPT \
        --base_model_path $base_model \
        --model_size $MODEL_SIZE \
        --local_tokenizer_path $LOCAL_TOKENIZER \
        --output_dir $output_dir \
        --run_name $run_id \
        --few_shot $few_shot \
        --use_lora \
        --lora_rank $lora_rank \
        --lora_alpha $LORA_ALPHA \
        --lora_lr $LORA_LR \
        --num_epochs $NUM_EPOCHS \
        --max_length $MAX_LENGTH \
        --train_batch_size $TRAIN_BS \
        --eval_batch_size $EVAL_BS \
        --lr $LR \
        --classifier_lr $CLASSIFIER_LR \
        --seed $seed \
        --eval_interval $EVAL_INTERVAL \
        --save_interval $SAVE_INTERVAL \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --early_stopping_patience $patience \
        --wandb_mode $WANDB_MODE \
        --wandb_dir $WANDB_DIR \
        --sst2_data_path $SST2_DATA_PATH
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ">>> [ERROR] RoPE + LoRA failed! Seq: $seq_len, Rank: $lora_rank, Shot: $shot_str, Seed: $seed"
        ((FAILED_COUNT++))
        return 1
    fi
    return 0
}

# ============================================================
# 批量运行函数
# ============================================================
run_all_rope_experiments() {
    echo -e "\n=========================================="
    echo ">>> Running RoPE SST-2 Experiments"
    echo "=========================================="
    
    for seed in "${SEEDS[@]}"; do
        for shot in "${SHOT_SETTINGS[@]}"; do
            
            # === 512 长度模型 ===
            if [ -f "$ROPE_512_PATH" ]; then
                for rank in "${LORA_RANKS[@]}"; do
                    run_rope_lora $ROPE_512_PATH 512 $shot $rank $seed
                done
            else
                echo ">>> WARNING: RoPE 512 model not found at $ROPE_512_PATH"
            fi
            
            # === 2048 长度模型 ===
            if [ -f "$ROPE_2048_PATH" ]; then
                for rank in "${LORA_RANKS[@]}"; do
                    run_rope_lora $ROPE_2048_PATH 2048 $shot $rank $seed
                done
            else
                echo ">>> WARNING: RoPE 2048 model not found at $ROPE_2048_PATH"
            fi
            
        done
    done
}

# ============================================================
# 主程序
# ============================================================

# 检查模型是否存在
if [ ! -f "$ROPE_512_PATH" ] && [ ! -f "$ROPE_2048_PATH" ]; then
    echo ">>> WARNING: No RoPE models found!"
    exit 1
fi

EXPERIMENT_MODE="${1:-lora}"

echo ">>> Starting RoPE SST-2 Fine-tuning"
echo ">>> Mode: $EXPERIMENT_MODE"
echo ">>> Seeds: ${SEEDS[@]}"
echo ">>> Shot settings: ${SHOT_SETTINGS[@]}"

case "$EXPERIMENT_MODE" in
    "all"|"lora")
        run_all_rope_experiments
        ;;
    *)
        echo "Usage: sbatch $0 {all|lora}"
        echo ""
        echo "Examples:"
        echo "  sbatch $0              # 运行LoRA实验（默认）"
        echo "  DEBUG_SHOT=500 sbatch $0  # Debug模式"
        exit 1
        ;;
esac

echo -e "\n=========================================="
echo ">>> All RoPE SST-2 experiments completed at $(date)"
echo ">>> Results saved to: $CHECKPOINT_ROOT"
if [ $FAILED_COUNT -gt 0 ]; then
    echo ">>> WARNING: $FAILED_COUNT experiment(s) failed!"
    exit 1
else
    echo ">>> All experiments completed successfully!"
    exit 0
fi
