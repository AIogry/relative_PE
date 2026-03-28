#!/bin/bash
# 使用小样本（K=128, steps=20）对所有4个模型进行评估

set -e

CODE_DIR="/home/qijunrong/03-proj/PE"
SCRIPT_DIR="${CODE_DIR}/scripts"

MODEL_SIZE="300M"
TRAIN_LEN=512
SIGMA=700.0
SEED=42

CKPT_ROOT="/data/qijunrong/03-proj/PE/checkpoints/pretrain_c4"

echo "=================================================="
echo "Small-Shot Evaluation (K=128, steps=20)"
echo "Purpose: Avoid overfitting, test true extrapolation"
echo "=================================================="

# 1. RoPE baseline
echo "Submitting: RoPE baseline..."
sbatch ${SCRIPT_DIR}/run_wikitext_smallshot.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_rope_L${TRAIN_LEN}_sig0.0_s${SEED}/model_final.pt" \
    rope 0.0 ${SEED}

# 2. RoPE + YaRN
echo "Submitting: RoPE + YaRN..."
sbatch ${SCRIPT_DIR}/run_wikitext_smallshot.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_rope_yarn_L${TRAIN_LEN}_sig0.0_s${SEED}/model_final.pt" \
    rope_yarn 0.0 ${SEED}

# 3. HIPE only
echo "Submitting: HIPE only..."
sbatch ${SCRIPT_DIR}/run_wikitext_smallshot.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe ${SIGMA} ${SEED}

# 4. HIPE + YaRN
echo "Submitting: HIPE + YaRN..."
sbatch ${SCRIPT_DIR}/run_wikitext_smallshot.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_yarn_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe_yarn ${SIGMA} ${SEED}

echo ""
echo "=================================================="
echo "All small-shot evaluations submitted!"
echo "=================================================="
echo ""
echo "Config: K=128, steps=20, lr=5e-6"
echo "Monitor: squeue -u \$USER"
echo "Results: /data/qijunrong/03-proj/PE/results/wikitext/*K128_small.json"
