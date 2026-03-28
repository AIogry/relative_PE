#!/bin/bash
# 运行所有C4评估（修复Padding Leakage）

set -e

CODE_DIR="/home/qijunrong/03-proj/PE"
SCRIPT_DIR="${CODE_DIR}/scripts"

MODEL_SIZE="300M"
TRAIN_LEN=512
SIGMA=700.0
SEED=42

CKPT_ROOT="/data/qijunrong/03-proj/PE/checkpoints/pretrain_c4"

echo "=================================================="
echo "C4 Evaluations V2 (Padding Leakage Fixed)"
echo "=================================================="
echo "Features:"
echo "  1. group_texts - no padding"
echo "  2. per-length adaptation"
echo "  3. ignore_index=50256 in loss"
echo "=================================================="

# 提交所有4个模型
for pe_type in rope rope_yarn hipe hipe_yarn; do
    sigma_val=0.0
    if [ "$pe_type" == "hipe" ] || [ "$pe_type" == "hipe_yarn" ]; then
        sigma_val=${SIGMA}
    fi
    
    echo "Submitting: ${pe_type}..."
    sbatch ${SCRIPT_DIR}/run_c4_fixed_v2.sh \
        "${CKPT_ROOT}/${MODEL_SIZE}_${pe_type}_L${TRAIN_LEN}_sig${sigma_val}_s${SEED}/model_final.pt" \
        ${pe_type} ${sigma_val} ${SEED}
done

echo ""
echo "=================================================="
echo "All C4 fixed v2 evaluations submitted!"
echo "=================================================="
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: /data/qijunrong/03-proj/PE/results/c4_fixed_v2/"
