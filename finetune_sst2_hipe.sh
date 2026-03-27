#!/bin/bash

#SBATCH --job-name=exp3-sst2-hipe-finetune
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/exp3/sst2/300m/tmp_sst2_hipe_%j.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/exp3/sst2/300m/tmp_sst2_hipe_%j.err
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
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp3/sst2/300m/hipe"
WANDB_DIR="${ROOT_DIR}/wandb/offline/sst2"
SST2_DATA_PATH="${ROOT_DIR}/sst2_data"

# HIPE 模型路径 (learnable sigma models)
# 路径中包含 thrX 标记，表示 threshold (如 thr7 表示 0-7层是RoPE, 8-15层是ScaledRoPE)
HIPE_512_PATH="${ROOT_DIR}/checkpoints_exp2/c4_300M/learnable_sigma/300M/c4_learnable_sigma_300M_L512_sigma200_thr7/seed_6198/model.pt"
HIPE_2048_PATH="${ROOT_DIR}/checkpoints_exp2/c4_300M/learnable_sigma/300M/c4_learnable_sigma_300M_L2048_sigma700_thr7/seed_6198/model.pt"

LOCAL_TOKENIZER="${ROOT_DIR}/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR

JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/sst2_hipe_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/sst2_hipe_${JOB_ID}.err"

exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f ${ROOT_DIR}/logs/exp3/sst2/tmp_sst2_hipe_${JOB_ID}.out
    rm -f ${ROOT_DIR}/logs/exp3/sst2/tmp_sst2_hipe_${JOB_ID}.err
}
trap cleanup EXIT

# === 从模型路径自动提取 threshold ===
extract_threshold() {
    local model_path=$1
    # 从路径中提取 thrX (如 thr7 -> 7)
    if [[ $model_path =~ thr([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "-1"  # 没有找到thr标记，所有层都是ScaledRoPE
    fi
}

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"

# === 全局配置 ===
MODEL_SIZE="300M"
SEEDS=(6198)
NUM_EPOCHS=5
MAX_LENGTH=128
TRAIN_BS=16
EVAL_BS=64
LR=5e-4
CLASSIFIER_LR=1e-3
LORA_LR=5e-4
SIGMA_LR=1e-3
EVAL_INTERVAL=100
SAVE_INTERVAL=500
GRAD_ACCUM_STEPS=2
EARLY_STOPPING_PATIENCE=5

# Few-shot 设置: -1=full, 其他为具体样本数
SHOT_SETTINGS=(-1 100 500 1000 2000 5000 10000)

# === LoRA 配置 ===
LORA_RANKS=(8)
LORA_ALPHA=32

# === HIPE 配置 ===
SIGMA_512=200
SIGMA_2048=700
DECAY_FUNC="gaussian"

# === DEBUG 配置 ===
DEBUG_SHOT=""
# DEBUG_SHOT=100

if [ -n "$DEBUG_SHOT" ]; then
    echo ">>> [DEBUG MODE] Shot: $DEBUG_SHOT"
    SHOT_SETTINGS=($DEBUG_SHOT)
fi

FAILED_COUNT=0

# ============================================================
# 核心函数1：HIPE 可学习 sigma
# ============================================================
run_hipe_learnable() {
    local base_model=$1
    local seq_len=$2
    local sigma=$3
    local few_shot=$4
    local seed=$5
    
    # 自动提取 threshold
    local threshold=$(extract_threshold "$base_model")
    echo ">>> Detected threshold: $threshold (layers 0-$threshold use fixed RoPE, layers $((threshold+1))-15 use ScaledRoPE)"
    
    # 动态早停：full数据集需要更多耐心
    local patience=$EARLY_STOPPING_PATIENCE
    if [ "$few_shot" -lt 0 ]; then
        patience=15  # full数据集增加到15
    elif [ "$few_shot" -ge 5000 ]; then
        patience=10  # 大数据集也增加
    fi
    echo ">>> Early stopping patience: $patience"
    
    local shot_str=$([ "$few_shot" -lt 0 ] && echo "full" || echo "shot${few_shot}")
    local run_id="hipe_${seq_len}_learnable_sigma${sigma}_thr${threshold}_${shot_str}_seed${seed}"
    local output_dir="${CHECKPOINT_ROOT}/${seq_len}/learnable_thr${threshold}/${shot_str}/seed_${seed}"
    
    echo -e "\n>>> [HIPE Learnable - SST-2] Seq: $seq_len | Sigma: $sigma | Threshold: $threshold | Shot: $shot_str | Seed: $seed"
    
    $PYTHON_BIN $SCRIPT \
        --base_model_path $base_model \
        --model_size $MODEL_SIZE \
        --local_tokenizer_path $LOCAL_TOKENIZER \
        --output_dir $output_dir \
        --run_name $run_id \
        --use_scaled_rope \
        --sigma $sigma \
        --rope_scaling_threshold $threshold \
        --decay_func $DECAY_FUNC \
        --learnable_sigma \
        --sigma_lr $SIGMA_LR \
        --few_shot $few_shot \
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
        echo ">>> [ERROR] HIPE learnable failed! Seq: $seq_len, Shot: $shot_str, Seed: $seed"
        ((FAILED_COUNT++))
        return 1
    fi
    return 0
}

# ============================================================
# 核心函数2：HIPE + LoRA
# ============================================================
run_hipe_lora() {
    local base_model=$1
    local seq_len=$2
    local sigma=$3
    local few_shot=$4
    local lora_rank=$5
    local seed=$6
    
    # 自动提取 threshold
    local threshold=$(extract_threshold "$base_model")
    echo ">>> Detected threshold: $threshold (layers 0-$threshold use fixed RoPE, layers $((threshold+1))-15 use ScaledRoPE)"
    
    # 动态早停：full数据集需要更多耐心
    local patience=$EARLY_STOPPING_PATIENCE
    if [ "$few_shot" -lt 0 ]; then
        patience=15  # full数据集增加到15
    elif [ "$few_shot" -ge 5000 ]; then
        patience=10  # 大数据集也增加
    fi
    echo ">>> Early stopping patience: $patience"
    
    local shot_str=$([ "$few_shot" -lt 0 ] && echo "full" || echo "shot${few_shot}")
    local run_id="hipe_${seq_len}_lora${lora_rank}_sigma${sigma}_thr${threshold}_${shot_str}_seed${seed}"
    local output_dir="${CHECKPOINT_ROOT}/${seq_len}/lora${lora_rank}_thr${threshold}/${shot_str}/seed_${seed}"
    
    echo -e "\n>>> [HIPE + LoRA - SST-2] Seq: $seq_len | Rank: $lora_rank | Sigma: $sigma | Threshold: $threshold | Shot: $shot_str | Seed: $seed"
    
    $PYTHON_BIN $SCRIPT \
        --base_model_path $base_model \
        --model_size $MODEL_SIZE \
        --local_tokenizer_path $LOCAL_TOKENIZER \
        --output_dir $output_dir \
        --run_name $run_id \
        --use_scaled_rope \
        --sigma $sigma \
        --rope_scaling_threshold $threshold \
        --decay_func $DECAY_FUNC \
        --learnable_sigma \
        --sigma_lr $SIGMA_LR \
        --use_lora \
        --lora_rank $lora_rank \
        --lora_alpha $LORA_ALPHA \
        --lora_lr $LORA_LR \
        --few_shot $few_shot \
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
        echo ">>> [ERROR] HIPE + LoRA failed! Seq: $seq_len, Rank: $lora_rank, Shot: $shot_str, Seed: $seed"
        ((FAILED_COUNT++))
        return 1
    fi
    return 0
}

# ============================================================
# 批量运行函数
# ============================================================
run_all_hipe_experiments() {
    echo -e "\n=========================================="
    echo ">>> Running HIPE SST-2 Experiments"
    echo "=========================================="
    
    for seed in "${SEEDS[@]}"; do
        for shot in "${SHOT_SETTINGS[@]}"; do
            
            # === 512 长度模型 ===
            if [ -f "$HIPE_512_PATH" ]; then
                run_hipe_learnable $HIPE_512_PATH 512 $SIGMA_512 $shot $seed
                
                for rank in "${LORA_RANKS[@]}"; do
                    run_hipe_lora $HIPE_512_PATH 512 $SIGMA_512 $shot $rank $seed
                done
            else
                echo ">>> WARNING: HIPE 512 model not found at $HIPE_512_PATH"
            fi
            
            # === 2048 长度模型 ===
            if [ -f "$HIPE_2048_PATH" ]; then
                run_hipe_learnable $HIPE_2048_PATH 2048 $SIGMA_2048 $shot $seed
                
                for rank in "${LORA_RANKS[@]}"; do
                    run_hipe_lora $HIPE_2048_PATH 2048 $SIGMA_2048 $shot $rank $seed
                done
            else
                echo ">>> WARNING: HIPE 2048 model not found at $HIPE_2048_PATH"
            fi
            
        done
    done
}

# ============================================================
# 主程序
# ============================================================

if [ ! -f "$HIPE_512_PATH" ] && [ ! -f "$HIPE_2048_PATH" ]; then
    echo ">>> WARNING: No HIPE models found!"
    exit 1
fi

EXPERIMENT_MODE="${1:-learnable}"

echo ">>> Starting HIPE SST-2 Fine-tuning"
echo ">>> Mode: $EXPERIMENT_MODE"
echo ">>> Seeds: ${SEEDS[@]}"
echo ">>> Shot settings: ${SHOT_SETTINGS[@]}"

case "$EXPERIMENT_MODE" in
    "all")
        run_all_hipe_experiments
        ;;
    "learnable")
        echo ">>> Running learnable sigma experiments"
        for shot in "${SHOT_SETTINGS[@]}"; do
            run_hipe_learnable $HIPE_512_PATH 512 $SIGMA_512 $shot 6198
            [ -f "$HIPE_2048_PATH" ] && run_hipe_learnable $HIPE_2048_PATH 2048 $SIGMA_2048 $shot 6198
        done
        ;;
    "lora")
        echo ">>> Running HIPE + LoRA experiments"
        for shot in "${SHOT_SETTINGS[@]}"; do
            for rank in "${LORA_RANKS[@]}"; do
                run_hipe_lora $HIPE_512_PATH 512 $SIGMA_512 $shot $rank 6198
                [ -f "$HIPE_2048_PATH" ] && run_hipe_lora $HIPE_2048_PATH 2048 $SIGMA_2048 $shot $rank 6198
            done
        done
        ;;
    *)
        echo "Usage: sbatch $0 {all|learnable|lora}"
        echo ""
        echo "Examples:"
        echo "  sbatch $0              # 运行learnable实验（默认）"
        echo "  sbatch $0 lora         # 运行LoRA实验"
        echo "  DEBUG_SHOT=500 sbatch $0 learnable  # Debug模式"
        exit 1
        ;;
esac

echo -e "\n=========================================="
echo ">>> All HIPE SST-2 experiments completed at $(date)"
echo ">>> Results saved to: $CHECKPOINT_ROOT"
if [ $FAILED_COUNT -gt 0 ]; then
    echo ">>> WARNING: $FAILED_COUNT experiment(s) failed!"
    exit 1
else
    echo ">>> All experiments completed successfully!"
    exit 0
fi
