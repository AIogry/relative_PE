#!/bin/bash

#SBATCH --job-name=exp3-sst2-hipe-full
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/exp3/sst2/300m/tmp_sst2_hipe_full_%j.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/exp3/sst2/300m/tmp_sst2_hipe_full_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ============================================================
# SST-2 Full Dataset Fine-tuning with Multiple Seeds (HIPE)
# 使用3个seeds运行全量数据实验，确保结果稳定性
# ============================================================

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

# HIPE 模型路径 (learnable sigma models with threshold)
HIPE_512_PATH="${ROOT_DIR}/checkpoints_exp2/c4_300M/learnable_sigma/300M/c4_learnable_sigma_300M_L512_sigma200_thr7/seed_6198/model.pt"
HIPE_2048_PATH="${ROOT_DIR}/checkpoints_exp2/c4_300M/learnable_sigma/300M/c4_learnable_sigma_300M_L2048_sigma700_thr7/seed_6198/model.pt"

LOCAL_TOKENIZER="${ROOT_DIR}/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR

JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/sst2_hipe_full_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/sst2_hipe_full_${JOB_ID}.err"

exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f ${ROOT_DIR}/logs/exp3/sst2/tmp_sst2_hipe_full_${JOB_ID}.out
    rm -f ${ROOT_DIR}/logs/exp3/sst2/tmp_sst2_hipe_full_${JOB_ID}.err
}
trap cleanup EXIT

# === 从模型路径自动提取 threshold ===
extract_threshold() {
    local model_path=$1
    if [[ $model_path =~ thr([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "7"  # 默认threshold
    fi
}

echo ">>> ===================================================================="
echo ">>> SST-2 Full Dataset Fine-tuning (HIPE) with 3 Seeds"
echo ">>> ===================================================================="
echo ">>> Experiment started at $(date)"
echo ">>> SLURM Job ID: ${JOB_ID}"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> ===================================================================="

# === 全局配置 ===
MODEL_SIZE="300M"
NUM_EPOCHS=15              # 更多epochs
MAX_LENGTH=128
TRAIN_BS=16
EVAL_BS=64
LR=5e-4
CLASSIFIER_LR=1e-3
LORA_LR=5e-4
SIGMA_LR=1e-3
SAVE_INTERVAL=2000
GRAD_ACCUM_STEPS=2

# === Full Dataset 专用配置 ===
FEW_SHOT=-1
EVAL_INTERVAL_SAMPLES=1000
EARLY_STOPPING_PATIENCE=30

# === LoRA 配置 ===
LORA_RANK=8
LORA_ALPHA=32

# === HIPE 配置 ===
SIGMA_512=200
SIGMA_2048=700
DECAY_FUNC="gaussian"

# === 3个Seeds配置 ===
SEEDS=(6198 7309 8420)
echo ">>> Running with ${#SEEDS[@]} seeds: ${SEEDS[@]}"

FAILED_COUNT=0

# ============================================================
# 核心函数：HIPE + LoRA (Full Dataset)
# ============================================================
run_hipe_full() {
    local base_model=$1
    local seq_len=$2
    local sigma=$3
    local seed=$4
    
    # 自动提取 threshold
    local threshold=$(extract_threshold "$base_model")
    
    local run_id="hipe_${seq_len}_lora${LORA_RANK}_sigma${sigma}_thr${threshold}_full_seed${seed}"
    local output_dir="${CHECKPOINT_ROOT}/${seq_len}/lora${LORA_RANK}_thr${threshold}/full_multiseed/seed_${seed}"
    
    echo ""
    echo ">>> ------------------------------------------------------------------------"
    echo ">>> [HIPE Full] Seq: $seq_len | Sigma: $sigma | Threshold: $threshold | Seed: $seed"
    echo ">>> ------------------------------------------------------------------------"
    echo ">>> Config:"
    echo "     - Train samples: 67,349 (full SST-2 train set)"
    echo "     - HIPE: Layers 0-$threshold use fixed RoPE, layers $((threshold+1))-15 use ScaledRoPE"
    echo "     - Eval interval: every $EVAL_INTERVAL_SAMPLES samples"
    echo "     - Early stopping patience: $EARLY_STOPPING_PATIENCE evals"
    echo "     - Max epochs: $NUM_EPOCHS"
    echo "     - Output: $output_dir"
    echo ""
    
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
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_lr $LORA_LR \
        --few_shot $FEW_SHOT \
        --num_epochs $NUM_EPOCHS \
        --max_length $MAX_LENGTH \
        --train_batch_size $TRAIN_BS \
        --eval_batch_size $EVAL_BS \
        --lr $LR \
        --classifier_lr $CLASSIFIER_LR \
        --seed $seed \
        --eval_interval_samples $EVAL_INTERVAL_SAMPLES \
        --save_interval $SAVE_INTERVAL \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --wandb_mode offline \
        --wandb_dir $WANDB_DIR \
        --sst2_data_path $SST2_DATA_PATH
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ">>> [ERROR] HIPE Full failed! Seq: $seq_len, Seed: $seed"
        ((FAILED_COUNT++))
        return 1
    fi
    
    echo ">>> [SUCCESS] HIPE Full completed! Seq: $seq_len, Seed: $seed"
    return 0
}

# ============================================================
# 主程序
# ============================================================

# 检查模型是否存在
if [ ! -f "$HIPE_512_PATH" ] && [ ! -f "$HIPE_2048_PATH" ]; then
    echo ">>> [ERROR] No HIPE models found!"
    echo "  Expected at:"
    echo "    - $HIPE_512_PATH"
    echo "    - $HIPE_2048_PATH"
    exit 1
fi

echo ""
echo ">>> Starting HIPE Full Dataset Experiments (${#SEEDS[@]} seeds × 2 seq lengths)"
echo ""

# 运行所有seeds
for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> ======================================================================"
    echo ">>> SEED: $seed ($((${#SEEDS[@]} - $(echo ${SEEDS[@]} | tr ' ' '\n' | grep -n $seed | cut -d: -f1) + 1))/${#SEEDS[@]})"
    echo ">>> ======================================================================"
    
    # 512长度模型
    if [ -f "$HIPE_512_PATH" ]; then
        run_hipe_full $HIPE_512_PATH 512 $SIGMA_512 $seed
    else
        echo ">>> [WARNING] HIPE 512 model not found at $HIPE_512_PATH"
    fi
    
    # 2048长度模型
    if [ -f "$HIPE_2048_PATH" ]; then
        run_hipe_full $HIPE_2048_PATH 2048 $SIGMA_2048 $seed
    else
        echo ">>> [WARNING] HIPE 2048 model not found at $HIPE_2048_PATH"
    fi
done

echo ""
echo ">>> ===================================================================="
echo ">>> All HIPE Full Dataset experiments completed at $(date)"
echo ">>> Results saved to: ${CHECKPOINT_ROOT}/*/lora${LORA_RANK}_thr*/full_multiseed/"
echo ">>> ===================================================================="

if [ $FAILED_COUNT -gt 0 ]; then
    echo ">>> [WARNING] $FAILED_COUNT experiment(s) failed!"
    exit 1
else
    echo ">>> All experiments completed successfully!"
    echo ""
    echo ">>> To analyze results across seeds, run:"
    echo "    python analyze_sample_efficiency.py --result_dir ${CHECKPOINT_ROOT}"
    exit 0
fi
