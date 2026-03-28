#!/bin/bash
# 运行所有4个模型的修复版评估

set -e

CODE_DIR="/home/qijunrong/03-proj/PE"
SCRIPT_DIR="${CODE_DIR}/scripts"

MODEL_SIZE="300M"
TRAIN_LEN=512
SIGMA=700.0
SEED=42

CKPT_ROOT="/data/qijunrong/03-proj/PE/checkpoints/pretrain_c4"

echo "=================================================="
echo "Running All Fixed Evaluations"
echo "Changes: C4 domain + K=2000 + 300 steps"
echo "=================================================="

# 1. RoPE baseline
echo "Submitting: RoPE baseline (fixed)..."
sbatch ${SCRIPT_DIR}/run_fixed_evaluation.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_rope_L${TRAIN_LEN}_sig0.0_s${SEED}/model_final.pt" \
    rope 0.0 ${SEED}

# 2. RoPE + YaRN
echo "Submitting: RoPE + YaRN (fixed)..."
sbatch ${SCRIPT_DIR}/run_fixed_evaluation.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_rope_yarn_L${TRAIN_LEN}_sig0.0_s${SEED}/model_final.pt" \
    rope_yarn 0.0 ${SEED}

# 3. HIPE only
echo "Submitting: HIPE only (fixed)..."
sbatch ${SCRIPT_DIR}/run_fixed_evaluation.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe ${SIGMA} ${SEED}

# 4. HIPE + YaRN (主要关注)
echo "Submitting: HIPE + YaRN (fixed)..."
sbatch ${SCRIPT_DIR}/run_fixed_evaluation.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_yarn_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe_yarn ${SIGMA} ${SEED}

echo ""
echo "=================================================="
echo "All 4 fixed evaluations submitted!"
echo "=================================================="
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: /data/qijunrong/03-proj/PE/results/fixed/"
