#!/bin/bash
# 立即运行所有评估实验

set -e

CODE_DIR="/home/qijunrong/03-proj/PE"
SCRIPT_DIR="${CODE_DIR}/scripts"

MODEL_SIZE="300M"
TRAIN_LEN=512
SIGMA=700.0
SEED=42

CKPT_ROOT="/data/qijunrong/03-proj/PE/checkpoints/pretrain_c4"

echo "=================================================="
echo "Running All Evaluations"
echo "=================================================="

# 1. RoPE baseline
echo "Submitting: RoPE baseline evaluation..."
sbatch ${SCRIPT_DIR}/run_fewshot_extrap.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_rope_L${TRAIN_LEN}_sig0.0_s${SEED}/model_final.pt" \
    rope 0.0 ${SEED}

# 2. RoPE + YaRN
echo "Submitting: RoPE + YaRN evaluation..."
sbatch ${SCRIPT_DIR}/run_fewshot_extrap.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_rope_yarn_L${TRAIN_LEN}_sig0.0_s${SEED}/model_final.pt" \
    rope_yarn 0.0 ${SEED}

# 3. HIPE only
echo "Submitting: HIPE only evaluation..."
sbatch ${SCRIPT_DIR}/run_fewshot_extrap.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe ${SIGMA} ${SEED}

# 4. HIPE + YaRN (主要评估)
echo "Submitting: HIPE + YaRN evaluation..."
sbatch ${SCRIPT_DIR}/run_fewshot_extrap.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_yarn_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe_yarn ${SIGMA} ${SEED}

echo ""
echo "=================================================="
echo "All 4 baseline evaluations submitted!"
echo "=================================================="
echo ""
echo "Next: Submit multi-shot experiments for HIPE+YaRN..."
sbatch ${SCRIPT_DIR}/run_multi_shot_experiments.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_yarn_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe_yarn ${SIGMA} ${SEED}

echo ""
echo "=================================================="
echo "All evaluation jobs submitted!"
echo "=================================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results will be saved to: /data/qijunrong/03-proj/PE/results/"
