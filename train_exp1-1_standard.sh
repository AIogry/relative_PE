#!/bin/bash -x

#SBATCH --job-name=olmo-exp1
#SBATCH --output=./logs/exp1_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
# === 环境设置 ===
export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp1_induction.py"
OUTPUT_DIR="./results_exp1/standard" 

mkdir -p $OUTPUT_DIR
mkdir -p ./logs

echo "Starting Experiment 1 Batch (High Steps & Simplified Task)..."

# 通用设置：降低难度，增加步数，增加 Batch Size
# 注意：steps 3000 是为了确保能观察到 Grokking
COMMON_ARGS="--vocab_size 50 --num_pairs 4 --steps 20000 --batch_size 512 --seq_len 64"

# ==========================================
# 1. Baseline (Standard RoPE)
# ==========================================
echo "Running Baseline..."
$PYTHON_BIN $SCRIPT \
    --output_dir $OUTPUT_DIR \
    --run_id "baseline_standard" \
    $COMMON_ARGS


# ==========================================
# 3. (可选) 测试 Exponential Decay
# ==========================================
# 如果你想对比 exp decay，可以解开下面的注释

declare -a EXP_SIGMAS=(0.5 0.1 1.0 10.0 50.0 100.0 500.0 0.01)
for sigma in "${EXP_SIGMAS[@]}"; do
    run_id="scaled_exp_sigma_${sigma}"
    echo "Running Scaled RoPE (Exp Decay) with Sigma: $sigma"
    $PYTHON_BIN $SCRIPT \
        --output_dir $OUTPUT_DIR \
        --run_id "$run_id" \
        --use_scaled_rope \
        --sigma $sigma \
        --decay_func "exp" \
        $COMMON_ARGS
done


echo "Experiment 1 Batch Finished. Data saved to $OUTPUT_DIR"