#!/bin/bash
#SBATCH --job-name=pe-multi-shot
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/extrap/%j_multi_shot.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/extrap/%j_multi_shot.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00

# 多shot大小对比实验
# 在单个模型上测试不同few-shot大小的效果

set -e

# 参数
MODEL_PATH=${1:-"/data/qijunrong/03-proj/PE/checkpoints/pretrain_c4/300M_hipe_yarn_L512_sig700.0_s42/model_final.pt"}
PE_TYPE=${2:-"hipe_yarn"}
SIGMA=${3:-700.0}
SEED=${4:-42}

CODE_DIR="/home/qijunrong/03-proj/PE"
DATA_DIR="/data/qijunrong/03-proj/PE"

ARXIV_TRAIN="${DATA_DIR}/arxiv_data/arxiv_train"
ARXIV_VAL="${DATA_DIR}/arxiv_data/arxiv_validation"
TOKENIZER_PATH="${DATA_DIR}/wikitext/tokenizer"
RESULTS_ROOT="${DATA_DIR}/results/multi_shot"

mkdir -p ${RESULTS_ROOT}

export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="${CODE_DIR}/eval_fewshot_extrap_v2.py"

echo "=================================================="
echo "Multi-Shot Size Experiments"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "PE Type: ${PE_TYPE}"
echo "=================================================="

# 不同的K值设置
# K值从小到大，调整steps和lr以保持总计算量合理
SHOT_CONFIGS=(
    "16:10:1e-5"    # K=16, steps=10, lr=1e-5
    "64:20:1e-5"    # K=64, steps=20, lr=1e-5
    "128:50:5e-6"   # K=128, steps=50, lr=5e-6
    "256:100:5e-6"  # K=256, steps=100, lr=5e-6
    "512:150:3e-6"  # K=512, steps=150, lr=3e-6
)

MODEL_SIZE="300M"
BASE_LEN=512
TEST_LENGTHS=(1024 2048 4096)

for config in "${SHOT_CONFIGS[@]}"; do
    IFS=':' read -r K STEPS LR <<< "$config"
    
    echo ""
    echo "--------------------------------------------------"
    echo "Running experiment: K=${K}, steps=${STEPS}, lr=${LR}"
    echo "--------------------------------------------------"
    
    RESULT_FILE="${RESULTS_ROOT}/${PE_TYPE}_K${K}_s${SEED}.json"
    
    ${PYTHON_BIN} ${SCRIPT} \
        --model_path ${MODEL_PATH} \
        --model_size ${MODEL_SIZE} \
        --arxiv_train_path ${ARXIV_TRAIN} \
        --arxiv_val_path ${ARXIV_VAL} \
        --local_tokenizer_path ${TOKENIZER_PATH} \
        --few_shot_k ${K} \
        --few_shot_steps ${STEPS} \
        --few_shot_lr ${LR} \
        --base_len ${BASE_LEN} \
        --test_lengths ${TEST_LENGTHS[@]} \
        --pe_type ${PE_TYPE} \
        --sigma ${SIGMA} \
        --rope_scaling_threshold 7 \
        --decay_func gaussian \
        --seed ${SEED} \
        --output_file ${RESULT_FILE} \
        --use_train_for_fewshot
    
    echo "Result saved: ${RESULT_FILE}"
done

echo ""
echo "=================================================="
echo "All multi-shot experiments complete!"
echo "=================================================="
echo "Results directory: ${RESULTS_ROOT}"

# 汇总结果
echo ""
echo "Summary of all experiments:"
for config in "${SHOT_CONFIGS[@]}"; do
    IFS=':' read -r K STEPS LR <<< "$config"
    RESULT_FILE="${RESULTS_ROOT}/${PE_TYPE}_K${K}_s${SEED}.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "  K=${K}: ${RESULT_FILE}"
    fi
done
