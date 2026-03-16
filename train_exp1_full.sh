#!/bin/bash

#SBATCH --job-name=exp1-full
#SBATCH --output=./logs/exp1_full_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"

# [关键] 确保这里指向你新命名的 Python 文件
SCRIPT="train_exp1_full.py"  

# === 路径 ===
CHECKPOINT_ROOT="/data/qijunrong/03-proj/PE/checkpoints_synthetic_full"
mkdir -p $CHECKPOINT_ROOT
mkdir -p ./logs

# === 基础参数 ===
# 保持与之前的设定一致
COMMON_ARGS="--vocab_size 50 --seq_len 64 --num_pairs 4 --steps 100000 --batch_size 64"

# ============================================================
# 定义参数循环空间
# ============================================================

# 1. 模型大小
MODELS=("60M") #  "20M")

# 2. 任务类型
TASKS=("standard") # "block" standard)

# 3. Sigma 列表 (在这里添加你想测试的 Sigma 值)
# 建议: 100.0 (中等模糊), 200.0 (强模糊)
SIGMAS=(700.0 100.0 10.0 500.0 250.0 1.0)  # 50.0 200.0 300.0 80.0) # 700.0 100.0 10.0 500.0 250.0 1.0)

# 4. Threshold 列表
# -1: Uniform (全层模糊) -> 验证“浅层模糊会损害短距离性能”
#  3: Bio-Gradient (最佳策略) -> 验证“最佳平衡”
THRESHOLDS=(2 3)

# ============================================================
# 实验主循环
# ============================================================

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        
        echo "======================================================="
        echo ">>> Processing Model: $model | Task: $task"
        echo "======================================================="

        # --- A. 跑 Baseline (Standard RoPE) ---
        # Baseline 不依赖 Sigma/Threshold，每个模型/任务只跑一次
        RUN_ID="exp1_${task}_${model}_baseline"
        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"
        
        echo ">>> Running Baseline..."
        $PYTHON_BIN $SCRIPT \
            --output_dir $OUTPUT_DIR \
            --run_id $RUN_ID \
            --model_size $model \
            --task_mode $task \
            $COMMON_ARGS \
            # 不加 --use_scaled_rope 即为 Baseline

        # --- B. 跑 Scaled RoPE (循环 Sigma 和 Threshold) ---
        for sigma in "${SIGMAS[@]}"; do
            for thr in "${THRESHOLDS[@]}"; do
                
                # 命名逻辑: uniform 或 grad
                if [ "$thr" -eq -1 ]; then
                    TYPE="uniform"
                else
                    TYPE="grad_thr${thr}"
                fi
                
                RUN_ID="exp1_${task}_${model}_${TYPE}_sig${sigma}"
                OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"
                
                echo ">>> Running $TYPE (Sigma=$sigma, Thr=$thr)..."
                $PYTHON_BIN $SCRIPT \
                    --output_dir $OUTPUT_DIR \
                    --run_id $RUN_ID \
                    --model_size $model \
                    --task_mode $task \
                    --use_scaled_rope \
                    --sigma $sigma \
                    --rope_scaling_threshold $thr \
                    $COMMON_ARGS
            done
        done
        
    done
done

echo ">>> All exp1 full experiments finished."