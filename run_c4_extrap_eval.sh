#!/bin/bash
# C4 Length Extrapolation Evaluation Script
# 评估RoPE、RoPE+YaRN、HIPE、HIPE+YaRN在C4数据集上的长度外推性能
# 
# 关键修复：
# 1. 使用group_texts连续拼接，避免padding leakage
# 2. 在每个目标长度下重新加载模型并微调
# 3. Loss计算使用ignore_index=50256忽略padding token
# 4. 动态调整batch size（4096长度使用batch_size=2）

set -e

# 配置路径
PROJECT_DIR="/home/qijunrong/03-proj/PE"
DATA_DIR="/data/qijunrong/03-proj/PE"
CHECKPOINT_DIR="/data/qijunrong/03-proj/PE/checkpoints/pretrain_c4"
RESULT_DIR="/data/qijunrong/03-proj/PE/results/c4_extrap"
LOG_DIR="/data/qijunrong/03-proj/PE/logs/c4_extrap"
TOKENIZER_PATH="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

# SLURM配置
PARTITION="debug"
TIME_LIMIT="5:00:00"
NUM_GPUS=1
NUM_CPUS=8
MEMORY="64G"

# 评估参数
MODEL_SIZE="300M"
BASE_LEN=512
TEST_LENGTHS=(1024 2048 4096)
FEW_SHOT_K=1000
FEW_SHOT_STEPS=100
FEW_SHOT_LR=5e-6
SIGMA=700.0

# 创建输出目录
mkdir -p "${RESULT_DIR}"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "C4 Length Extrapolation Evaluation"
echo "=========================================="
echo "Results: ${RESULT_DIR}"
echo "Logs: ${LOG_DIR}"
echo ""

# ========================================
# 1. RoPE Baseline
# ========================================
echo "[1/4] Submitting RoPE baseline..."
sbatch \
    --job-name=c4_rope \
    --partition=${PARTITION} \
    --gres=gpu:${NUM_GPUS} \
    --cpus-per-task=${NUM_CPUS} \
    --mem=${MEMORY} \
    --time=${TIME_LIMIT} \
    --output=${LOG_DIR}/rope_%j.out \
    --error=${LOG_DIR}/rope_%j.err \
    --wrap="python ${PROJECT_DIR}/eval_fewshot_c4_fixed_v2.py \
        --model_path ${CHECKPOINT_DIR}/300M_rope_L512_sig0.0_s42/model_final.pt \
        --model_size ${MODEL_SIZE} \
        --dataset_path ${DATA_DIR} \
        --local_tokenizer_path ${TOKENIZER_PATH} \
        --pe_type rope \
        --few_shot_k ${FEW_SHOT_K} \
        --few_shot_steps ${FEW_SHOT_STEPS} \
        --few_shot_lr ${FEW_SHOT_LR} \
        --base_len ${BASE_LEN} \
        --test_lengths ${TEST_LENGTHS[@]} \
        --output_file ${RESULT_DIR}/rope_c4_extrap.json"

# ========================================
# 2. RoPE + YaRN
# ========================================
echo "[2/4] Submitting RoPE + YaRN..."
sbatch \
    --job-name=c4_rope_yarn \
    --partition=${PARTITION} \
    --gres=gpu:${NUM_GPUS} \
    --cpus-per-task=${NUM_CPUS} \
    --mem=${MEMORY} \
    --time=${TIME_LIMIT} \
    --output=${LOG_DIR}/rope_yarn_%j.out \
    --error=${LOG_DIR}/rope_yarn_%j.err \
    --wrap="python ${PROJECT_DIR}/eval_fewshot_c4_fixed_v2.py \
        --model_path ${CHECKPOINT_DIR}/300M_rope_L512_sig0.0_s42/model_final.pt \
        --model_size ${MODEL_SIZE} \
        --dataset_path ${DATA_DIR} \
        --local_tokenizer_path ${TOKENIZER_PATH} \
        --pe_type rope_yarn \
        --few_shot_k ${FEW_SHOT_K} \
        --few_shot_steps ${FEW_SHOT_STEPS} \
        --few_shot_lr ${FEW_SHOT_LR} \
        --base_len ${BASE_LEN} \
        --test_lengths ${TEST_LENGTHS[@]} \
        --output_file ${RESULT_DIR}/rope_yarn_c4_extrap.json"

# ========================================
# 3. HIPE (sigma=700)
# ========================================
echo "[3/4] Submitting HIPE (sigma=700)..."
sbatch \
    --job-name=c4_hipe \
    --partition=${PARTITION} \
    --gres=gpu:${NUM_GPUS} \
    --cpus-per-task=${NUM_CPUS} \
    --mem=${MEMORY} \
    --time=${TIME_LIMIT} \
    --output=${LOG_DIR}/hipe_%j.out \
    --error=${LOG_DIR}/hipe_%j.err \
    --wrap="python ${PROJECT_DIR}/eval_fewshot_c4_fixed_v2.py \
        --model_path ${CHECKPOINT_DIR}/300M_hipe_L512_sig700.0_s42/model_final.pt \
        --model_size ${MODEL_SIZE} \
        --dataset_path ${DATA_DIR} \
        --local_tokenizer_path ${TOKENIZER_PATH} \
        --pe_type hipe \
        --sigma ${SIGMA} \
        --few_shot_k ${FEW_SHOT_K} \
        --few_shot_steps ${FEW_SHOT_STEPS} \
        --few_shot_lr ${FEW_SHOT_LR} \
        --base_len ${BASE_LEN} \
        --test_lengths ${TEST_LENGTHS[@]} \
        --output_file ${RESULT_DIR}/hipe_c4_extrap.json"

# ========================================
# 4. HIPE + YaRN (sigma=700)
# ========================================
echo "[4/4] Submitting HIPE + YaRN (sigma=700)..."
sbatch \
    --job-name=c4_hipe_yarn \
    --partition=${PARTITION} \
    --gres=gpu:${NUM_GPUS} \
    --cpus-per-task=${NUM_CPUS} \
    --mem=${MEMORY} \
    --time=${TIME_LIMIT} \
    --output=${LOG_DIR}/hipe_yarn_%j.out \
    --error=${LOG_DIR}/hipe_yarn_%j.err \
    --wrap="python ${PROJECT_DIR}/eval_fewshot_c4_fixed_v2.py \
        --model_path ${CHECKPOINT_DIR}/300M_hipe_yarn_L512_sig700.0_s42/model_final.pt \
        --model_size ${MODEL_SIZE} \
        --dataset_path ${DATA_DIR} \
        --local_tokenizer_path ${TOKENIZER_PATH} \
        --pe_type hipe_yarn \
        --sigma ${SIGMA} \
        --few_shot_k ${FEW_SHOT_K} \
        --few_shot_steps ${FEW_SHOT_STEPS} \
        --few_shot_lr ${FEW_SHOT_LR} \
        --base_len ${BASE_LEN} \
        --test_lengths ${TEST_LENGTHS[@]} \
        --output_file ${RESULT_DIR}/hipe_yarn_c4_extrap.json"

echo ""
echo "=========================================="
echo "All 4 jobs submitted successfully!"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs with: tail -f ${LOG_DIR}/*.err"
echo "Results will be saved in: ${RESULT_DIR}"
