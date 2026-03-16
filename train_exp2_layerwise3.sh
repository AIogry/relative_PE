#!/bin/bash

#SBATCH --job-name=exp2-grad-fix
#SBATCH --output=./logs/exp2_grad_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_layerwise.py"

# === 路径配置 ===
CHECKPOINT_ROOT="/data/qijunrong/03-proj/PE/checkpoints_gradient_full"
LOCAL_DATA="/data/qijunrong/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

mkdir -p $CHECKPOINT_ROOT
mkdir -p ./logs

# === 全局参数 ===
GLOBAL_BS=64
SEED=6198
MAX_TOKENS=100000000

# ==========================================
# DEBUG 开关
# ==========================================
# DEBUG_STEPS=50      # <--- 测试时解开
DEBUG_STEPS=""       # <--- 正式跑解开

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE] Running for $DEBUG_STEPS steps only."
    TRAIN_ARGS="--max_train_steps $DEBUG_STEPS"
else
    echo ">>> [FULL MODE] Training for $MAX_TOKENS tokens."
    TRAIN_ARGS="--max_tokens $MAX_TOKENS"
fi

# ==========================================
# 1. 实验循环
# ==========================================

MODELS=("20M")
LENGTHS=(2048)

# Bio-Gradient 配置 (Corrected: Small -> Large)
# 消融实验
# EXP_CONFIGS=(
#    "grad_exp:None None 5.0 10.0 25.0 50.0 100.0 250.0"
#    "grad_linear:None None 10.0 30.0 50.0 70.0 90.0 120.0"
#    "grad_late:None None None None None None 50.0 200.0"
#    "grad_early:None None 20.0 40.0 80.0 150.0 300.0 500.0"
# )

EXP_CONFIGS=(
    "grad_1:None None None 50.0 200.0 500.0 700.0 1000.0"
    "grad_2:None None 10.0 50.0 200.0 500.0 700.0 1000.0"
    "grad_3:None None None None 50.0 200.0 500.0 700.0"
    "grad_4:None 1.0 10.0 50.0 200.0 500.0 700.0 1000.0"
    "grad_5:None None None None None None 200.0 700.0"
)

echo ">>> Starting Experiments..."

for MODEL in "${MODELS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do
        
        # --- [关键修正] 动态调整 Micro Batch Size 防止 OOM ---
        # 针对 5090 (32GB) + WikiText (Vocab 50k) 的优化策略
        
        if [ "$SEQ_LEN" -eq 512 ]; then
            MICRO_BS=64  # 512长度，显存够用，全速跑
        elif [ "$SEQ_LEN" -eq 1024 ]; then
            MICRO_BS=16  # 1024长度，BS=64会爆，降到16 (Accum=4)
        elif [ "$SEQ_LEN" -ge 2048 ]; then
            MICRO_BS=8   # 2048长度，进一步降低 (Accum=8)
        else
            MICRO_BS=8   # 兜底
        fi
        
        # 如果是 60M 模型且长度很长，再保守一点
        if [ "$MODEL" == "60M" ] && [ "$SEQ_LEN" -ge 2048 ]; then
            MICRO_BS=4
        fi
        
        for config in "${EXP_CONFIGS[@]}"; do
            
            EXP_SUFFIX="${config%%:*}" 
            SIGMA_LIST="${config#*:}" 
            
            RUN_ID="${MODEL}_L${SEQ_LEN}_${EXP_SUFFIX}"
            
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            if [ -n "$DEBUG_STEPS" ]; then
                OUTPUT_DIR="$CHECKPOINT_ROOT/debug_${RUN_ID}_${TIMESTAMP}"
            else
                OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}_${TIMESTAMP}"
            fi

            echo "================================================================"
            echo ">>> Run: $RUN_ID"
            echo ">>> Len: $SEQ_LEN | Global BS: $GLOBAL_BS | Micro BS: $MICRO_BS"
            echo ">>> Accum Steps: $((GLOBAL_BS / MICRO_BS))"
            echo "================================================================"

            $PYTHON_BIN $SCRIPT \
                --output_dir $OUTPUT_DIR \
                --run_id $RUN_ID \
                --model_size $MODEL \
                --local_data_path $LOCAL_DATA \
                --local_tokenizer_path $LOCAL_TOKENIZER \
                --seq_len $SEQ_LEN \
                --global_batch_size $GLOBAL_BS \
                --micro_batch_size $MICRO_BS \
                --seed $SEED \
                --use_scaled_rope \
                --sigma_list $SIGMA_LIST \
                $TRAIN_ARGS
            
            if [ $? -ne 0 ]; then
                echo ">>> [ERROR] Run ${RUN_ID} Failed!"
                if [ -n "$DEBUG_STEPS" ]; then exit 1; fi
            fi
            
        done
    done
done

echo ">>> All Experiments Finished."