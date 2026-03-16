#!/bin/bash -x

#SBATCH --job-name=viz_prep_fast
#SBATCH --output=./logs/viz_prep_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"

# 🔴 关键：必须使用包含 torch.save 的 viz 脚本
SCRIPT="train_exp1_viz.py"

# 输出根目录
OUTPUT_ROOT="./checkpoints_viz"
mkdir -p $OUTPUT_ROOT
mkdir -p ./logs

echo ">>> Starting Visualization Prep (Fast Mode)..."

# === 通用参数 (参考了 exp1_standard 的高性能配置) ===
# Batch Size 512能更好地利用显存，加速收敛
# 步数 20000 保证绝对收敛 (Visual Pattern 会非常清晰)
COMMON_ARGS="--vocab_size 50 --num_pairs 4 --steps 20000 --batch_size 512 --seq_len 64 --model_size 20M"

# ==========================================
# 2. Bio-Gradient (Ours)
# ==========================================
# 选用效果最明显的参数: Sigma=200, Threshold=3
SIGMA=200.0
THR=3
RUN_ID="viz_gradient_sig${SIGMA}_thr${THR}"
OUT_DIR="$OUTPUT_ROOT/bio_gradient"
mkdir -p $OUT_DIR

echo ">>> Running Bio-Gradient (Sigma=$SIGMA, Thr=$THR)..."
$PYTHON_BIN $SCRIPT \
    --output_dir $OUT_DIR \
    --run_id $RUN_ID \
    --use_scaled_rope \
    --sigma $SIGMA \
    --rope_scaling_threshold $THR \
    $COMMON_ARGS

echo ">>> All Visualization Models Prepared!"


# ==========================================
# 1. Baseline (Standard RoPE)
# ==========================================
RUN_ID="viz_baseline"
OUT_DIR="$OUTPUT_ROOT/baseline"
mkdir -p $OUT_DIR

echo ">>> Running Baseline..."
$PYTHON_BIN $SCRIPT \
    --output_dir $OUT_DIR \
    --run_id $RUN_ID \
    $COMMON_ARGS \
    # Standard RoPE (默认配置)

