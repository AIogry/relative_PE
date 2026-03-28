#!/bin/bash
# 评估所有预训练模型（需要在预训练完成后手动运行）

CODE_DIR="/home/qijunrong/03-proj/PE"
SCRIPT_DIR="${CODE_DIR}/scripts"
MODEL_SIZE="300M"
TRAIN_LEN=512
SIGMA=700.0
SEED=6198

CKPT_ROOT="/data/qijunrong/03-proj/PE/checkpoints/pretrain_c4"

# RoPE baseline
echo "Evaluating RoPE baseline..."
sbatch ${SCRIPT_DIR}/run_fewshot_extrap.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_rope_L${TRAIN_LEN}_sig0.0_s${SEED}/model_final.pt" \
    rope 0.0 ${SEED}

# RoPE + YaRN
echo "Evaluating RoPE + YaRN..."
sbatch ${SCRIPT_DIR}/run_fewshot_extrap.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_rope_yarn_L${TRAIN_LEN}_sig0.0_s${SEED}/model_final.pt" \
    rope_yarn 0.0 ${SEED}

# HIPE only
echo "Evaluating HIPE only..."
sbatch ${SCRIPT_DIR}/run_fewshot_extrap.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe ${SIGMA} ${SEED}

# HIPE + YaRN
echo "Evaluating HIPE + YaRN..."
sbatch ${SCRIPT_DIR}/run_fewshot_extrap.sh \
    "${CKPT_ROOT}/${MODEL_SIZE}_hipe_yarn_L${TRAIN_LEN}_sig${SIGMA}_s${SEED}/model_final.pt" \
    hipe_yarn ${SIGMA} ${SEED}

echo "All evaluation jobs submitted!"
