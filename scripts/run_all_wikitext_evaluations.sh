#!/bin/bash
# 使用WikiText-103对所有4个模型进行评估（跨领域，避免数据污染）

set -e

CODE_DIR="/home/qijunrong/03-proj/PE"
SCRIPT_DIR="${CODE_DIR}/scripts"

MODEL_SIZE="300M"
TRAIN_LEN=512
SIGMA=700.0
SEED=42

CKPT_ROOT="/data/qijunrong/03-proj/PE/checkpoints/pretrain_c4"

echo "=================================================="
echo "Running All WikiText-103 Evaluations"
echo "Data: WikiText-103 (cross-domain from C4)"
echo "Few-shot K: 5000"
echo "=================================================="

# 1. RoPE baseline
echo "Submitting: RoPE baseline (WikiText)..."
sbatch ${SCRIPT_DIR}/run_wikitext_evaluation.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_rope_L${TRAIN_LEN}_sig0.0_s${SEED}/model_final.pt" \
    rope 0.0 ${SEED}

# 2. RoPE + YaRN
echo "Submitting: RoPE + YaRN (WikiText)..."
sbatch ${SCRIPT_DIR}/run_wikitext_evaluation.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_rope_yarn_L${TRAIN_LEN}_sig0.0_s${SEED}/model_final.pt" \
    rope_yarn 0.0 ${SEED}

# 3. HIPE only
echo "Submitting: HIPE only (WikiText)..."
sbatch ${SCRIPT_DIR}/run_wikitext_evaluation.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe ${SIGMA} ${SEED}

# 4. HIPE + YaRN (主要关注)
echo "Submitting: HIPE + YaRN (WikiText)..."
sbatch ${SCRIPT_DIR}/run_wikitext_evaluation.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_yarn_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe_yarn ${SIGMA} ${SEED}

echo ""
echo "=================================================="
echo "All WikiText evaluations submitted!"
echo "=================================================="
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: /data/qijunrong/03-proj/PE/results/wikitext/"
echo ""
echo "Expected time: 4-8 hours per job"
echo ""
