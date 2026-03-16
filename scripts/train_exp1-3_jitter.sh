#!/bin/bash -x

#SBATCH --job-name=olmo-exp1-jitter
#SBATCH --output=./logs/exp1-jitter_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
# === 环境设置 ===
export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp1_induction.py"
OUTPUT_DIR="./results_exp1/jitter" 

mkdir -p $OUTPUT_DIR
mkdir -p ./logs

echo "Starting Jittered Induction Experiment (Robustness Test)..."

# === 核心参数配置 ===
# 任务: jitter
# 难度: max_jitter 5 (位置随机偏移 0-5)
# 规模: num_pairs 8 (中等密度), vocab 100 (防止随机猜中)
# 长度: seq_len 128 (留足空间给 jitter)
COMMON_ARGS="--task jitter --max_jitter 5 --vocab_size 100 --num_pairs 8 --steps 30000 --batch_size 256 --seq_len 128"

# ==========================================
# 2. Scaled RoPE (Exp Decay)
# ==========================================
# 预期：Sigma 2.0 - 5.0 表现最佳，能忽略 jitter 带来的高频位置噪声
declare -a EXP_SIGMAS=(2.0 5.0 10.0)

for sigma in "${EXP_SIGMAS[@]}"; do
    run_id="scaled_jitter_sigma_${sigma}"
    
    echo "Running Scaled RoPE (Exp Decay) with Sigma: $sigma"
    
    $PYTHON_BIN $SCRIPT \
        --output_dir $OUTPUT_DIR \
        --run_id "$run_id" \
        --use_scaled_rope \
        --sigma $sigma \
        --decay_func "exp" \
        $COMMON_ARGS
done

echo "Jitter Experiment Finished. Data saved to $OUTPUT_DIR"