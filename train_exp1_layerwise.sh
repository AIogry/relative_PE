#!/bin/bash

#SBATCH --job-name=olmo-layer
#SBATCH --output=./logs/layer_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp1_layerwise.py"
RESULTS_DIR="./results_layerwise"

mkdir -p $RESULTS_DIR/block
mkdir -p $RESULTS_DIR/induction
mkdir -p ./logs

# =======================================================
# 实验 A: Block Copy 任务 (测试精度)
# 预期: Bio-Gradient 应该比 Uniform Sigma=0.5 强，接近 Baseline
# =======================================================
echo ">>> Running Block Copy Experiments..."

BLOCK_ARGS="--task block --block_size 5 --vocab_size 100 --num_pairs 4 --steps 20000 --batch_size 256 --seq_len 128"

# 1. Baseline
$PYTHON_BIN $SCRIPT \
    --output_dir $RESULTS_DIR/block \
    --run_id "baseline_block" \
    $BLOCK_ARGS

# 2. Uniform Scaled (Sigma=0.5) - [参照组: 预期 Acc 较低]
$PYTHON_BIN $SCRIPT \
    --output_dir $RESULTS_DIR/block \
    --run_id "uniform_0.5_block" \
    --use_scaled_rope --sigma 0.5 \
    $BLOCK_ARGS

# 3. Bio-Gradient (Thr=2, Sigma=0.5) - [主角: 预期 Acc 高]
# Layer 0-2: Standard (负责精准复制)
# Layer 3-7: Sigma 0.5 (负责泛化)
$PYTHON_BIN $SCRIPT \
    --output_dir $RESULTS_DIR/block \
    --run_id "gradient_block" \
    --use_scaled_rope --sigma 0.5 \
    --rope_scaling_threshold 2 \
    $BLOCK_ARGS


# =======================================================
# 实验 B: Standard Induction 任务 (测试泛化)
# 预期: Bio-Gradient 应该保持高 Acc
# =======================================================
echo ">>> Running Standard Induction Experiments..."

INDUCT_ARGS="--task standard --vocab_size 100 --num_pairs 4 --steps 10000 --batch_size 256 --seq_len 128"

# 1. Baseline
$PYTHON_BIN $SCRIPT \
    --output_dir $RESULTS_DIR/induction \
    --run_id "baseline_induct" \
    $INDUCT_ARGS

# 2. Bio-Gradient (Thr=2, Sigma=0.5)
$PYTHON_BIN $SCRIPT \
    --output_dir $RESULTS_DIR/induction \
    --run_id "gradient_induct" \
    --use_scaled_rope --sigma 0.5 \
    --rope_scaling_threshold 2 \
    $INDUCT_ARGS

echo "Layer-wise Experiments Finished."