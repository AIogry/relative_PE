#!/bin/bash
#SBATCH --job-name=olmo-60m-len2048-zeroshot
#SBATCH --ntasks=1
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/%x_%j.out
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mem=128G

# ================= 环境配置 =================

# 关键路径配置 (与你的训练脚本保持一致)
export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # 推理时通常不需要这个，除非报错

# 指定 Python 解释器
PYTHON_EXE="/home/qijunrong/anaconda3/bin/python"

# ================= 路径配置 =================
# ⚠️ 请务必确认这里的 checkpoint 路径与你实际生成的文件夹名一致
BASELINE_CKPT="/data/qijunrong/03-proj/PE/checkpoints/olmo-60m-Baseline-RoPE-flash-len2048-1.5B/final_model.pt"
SCALED_CKPT="/data/qijunrong/03-proj/PE/checkpoints/olmo-60m-ScaledRoPE-flash-len2048-1.5B/final_model.pt"
DATA_PATH="/data/qijunrong/03-proj/PE/c4_30M_validation"

# 配置文件
CONFIG_PATH="./configs/olmo_60m.yaml"

# ================= 实验 1: 评估 Baseline =================
echo "===================================================================="
echo "Starting Zero-shot Eval: Baseline"
echo "Checkpoint: $BASELINE_CKPT"
echo "Target Lengths: 2048, 4096, 8192"
echo "===================================================================="

$PYTHON_EXE eval_extrapolation.py \
    --config $CONFIG_PATH \
    --checkpoint "$BASELINE_CKPT" \
    --data_path "$DATA_PATH" \
    --lengths 2048 4096 8192

echo "Finished Eval: Baseline"
echo "--------------------------------------------------------------------"
echo ""

# ================= 实验 2: 评估 Scaled RoPE =================
SIGMA=85.0

echo "===================================================================="
echo "Starting Zero-shot Eval: Scaled RoPE (Inductive Bias Check)"
echo "Checkpoint: $SCALED_CKPT"
echo "Forcing Scaled RoPE with Sigma: $SIGMA"
echo "Target Lengths: 2048, 4096, 8192"
echo "===================================================================="

$PYTHON_EXE eval_extrapolation.py \
    --config $CONFIG_PATH \
    --checkpoint "$SCALED_CKPT" \
    --data_path "$DATA_PATH" \
    --lengths 2048 4096 8192 \
    --force_scaled_rope \
    --sigma $SIGMA

echo "Finished Eval: Scaled RoPE"
echo "===================================================================="