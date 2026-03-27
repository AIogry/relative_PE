#!/bin/bash
#SBATCH --job-name=pe-test-baseline
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/test/%j_baseline.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/test/%j_baseline.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=30:00

# 快速测试RoPE基线（无YaRN）

set -e

CODE_DIR="/home/qijunrong/03-proj/PE"
DATA_DIR="/data/qijunrong/03-proj/PE"

mkdir -p ${DATA_DIR}/checkpoints/test ${DATA_DIR}/logs/test

export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED=1

PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
TRAIN_SCRIPT="${CODE_DIR}/train_hipe_c4_pretrain.py"
EVAL_SCRIPT="${CODE_DIR}/eval_fewshot_extrap.py"

SEED=6198

echo "=================================================="
echo "Quick Test: RoPE Baseline (no YaRN)"
echo "=================================================="

# 1. 训练RoPE基线
RUN_ID="TEST_20M_rope_s${SEED}"
OUTPUT_DIR="${DATA_DIR}/checkpoints/test/${RUN_ID}"

echo "Step 1: Training RoPE baseline..."
${PYTHON_BIN} ${TRAIN_SCRIPT} \
    --output_dir ${OUTPUT_DIR} \
    --run_id ${RUN_ID} \
    --dataset_path ${DATA_DIR} \
    --local_tokenizer_path "${DATA_DIR}/wikitext/tokenizer" \
    --model_size "20M" \
    --seq_len 512 \
    --train_size 1000 \
    --val_size 100 \
    --max_tokens 10000000 \
    --global_batch_size 32 \
    --micro_batch_size 8 \
    --lr 3e-4 \
    --pe_type "rope" \
    --eval_interval 50 \
    --save_interval 100 \
    --seed ${SEED} \
    --wandb_mode offline

echo ""
echo "Step 2: Evaluating RoPE baseline (direct extrapolation, no YaRN)..."

# 2. 评估RoPE基线（不使用YaRN，直接外推）
${PYTHON_BIN} ${EVAL_SCRIPT} \
    --model_path "${OUTPUT_DIR}/model_final.pt" \
    --model_size "20M" \
    --arxiv_data_path "${DATA_DIR}/arxiv_data/arxiv_validation" \
    --local_tokenizer_path "${DATA_DIR}/wikitext/tokenizer" \
    --few_shot_k 50 \
    --few_shot_steps 10 \
    --few_shot_lr 1e-5 \
    --base_len 512 \
    --test_lengths 1024 2048 \
    --pe_type "rope" \
    --seed ${SEED} \
    --output_file "${DATA_DIR}/results/test/rope_baseline_result.json" \
    --eval_batches 5

echo ""
echo "=================================================="
echo "RoPE Baseline Test Complete!"
echo "=================================================="
