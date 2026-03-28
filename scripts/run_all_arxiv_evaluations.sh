#!/bin/bash
# 运行所有ArXiv评估（两种adapt模式）

set -e

CODE_DIR="/home/qijunrong/03-proj/PE"
SCRIPT_DIR="${CODE_DIR}/scripts"

MODEL_SIZE="300M"
TRAIN_LEN=512
SIGMA=700.0
SEED=42

CKPT_ROOT="/data/qijunrong/03-proj/PE/checkpoints/pretrain_c4"

echo "=================================================="
echo "ArXiv Evaluations - All Models & Modes"
echo "=================================================="

# 模式1：在base长度（512）下adapt，然后外推（标准做法）
echo ""
echo "=== Mode 1: Adapt on BASE length (512) ==="

for pe_type in rope rope_yarn hipe hipe_yarn; do
    sigma_val=0.0
    if [ "$pe_type" == "hipe" ] || [ "$pe_type" == "hipe_yarn" ]; then
        sigma_val=${SIGMA}
    fi
    
    echo "Submitting: ${pe_type} (base adapt)..."
    sbatch ${SCRIPT_DIR}/run_arxiv_evaluation.sh \
        "${CKPT_ROOT}/${MODEL_SIZE}_${pe_type}_L${TRAIN_LEN}_sig${sigma_val}_s${SEED}/model_final.pt" \
        ${pe_type} ${sigma_val} base ${SEED}
done

# 模式2：在每个外推长度下分别adapt（更强的领域适应）
echo ""
echo "=== Mode 2: Adapt on EACH extrap length ==="

for pe_type in rope rope_yarn hipe hipe_yarn; do
    sigma_val=0.0
    if [ "$pe_type" == "hipe" ] || [ "$pe_type" == "hipe_yarn" ]; then
        sigma_val=${SIGMA}
    fi
    
    echo "Submitting: ${pe_type} (extrap adapt)..."
    sbatch ${SCRIPT_DIR}/run_arxiv_evaluation.sh \
        "${CKPT_ROOT}/${MODEL_SIZE}_${pe_type}_L${TRAIN_LEN}_sig${sigma_val}_s${SEED}/model_final.pt" \
        ${pe_type} ${sigma_val} extrap ${SEED}
done

echo ""
echo "=================================================="
echo "All ArXiv evaluations submitted!"
echo "=================================================="
echo ""
echo "Total: 8 jobs (4 models × 2 modes)"
echo "Results: /data/qijunrong/03-proj/PE/results/arxiv/"
echo ""
echo "Mode 1 (base adapt): Tests zero-shot extrapolation after brief domain adaptation"
echo "Mode 2 (extrap adapt): Tests length-specific adaptation capability"
