#!/bin/bash
#SBATCH --job-name=exp2-300m-baselines
#SBATCH --output=./logs/exp2_300m_baselines_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_wikifull.py"

# === 路径配置 ===
CHECKPOINT_ROOT="/data/qijunrong/03-proj/PE/checkpoints_variable_len"
LOCAL_DATA="/data/qijunrong/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

mkdir -p $CHECKPOINT_ROOT
mkdir -p ./logs

# === 全局配置 ===
GLOBAL_BS=64
SEEDS=(6198 1024 7 568 3427)
MAX_TOKENS=100000000 # 1亿 Token

# === DEBUG 配置 ===
DEBUG_STEPS=""
#DEBUG_STEPS=100  # <--- 取消注释开启调试

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    # 300M 模型学习率统一调降至 3e-4
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS --lr 3e-4"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS --lr 3e-4"
fi

# ============================================================
# 实验配置
# ============================================================
MODELS=("300M")
LENGTHS=(512 1024 2048)

get_mbs() {
    local seq_len=$1
    # 300M 显存压力极大，基础 MBS 设为 8
    local mbs=8
    if [ "$seq_len" -ge 2048 ]; then mbs=4; fi
    echo $mbs
}

# ============================================================
# 1. Baseline: Standard RoPE
# ============================================================
echo ">>> [BATCH START] Running Standard RoPE..."
for SEED in "${SEEDS[@]}"; do
    for M_SIZE in "${MODELS[@]}"; do
        for SEQ_LEN in "${LENGTHS[@]}"; do
            CUR_MICRO_BS=$(get_mbs $SEQ_LEN)
            RUN_ID="baseline_rope_${M_SIZE}_L${SEQ_LEN}"
            if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
            
            OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

            echo ">>> [RoPE] Model: $M_SIZE | Len: $SEQ_LEN | MBS: $CUR_MICRO_BS | SEED: $SEED"
            $PYTHON_BIN $SCRIPT \
                --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
                --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
                --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
                $LIMIT_ARGS --seed $SEED
        done
    done
done

# ============================================================
# 2. Baseline: NoPE (No Positional Encoding)
# ============================================================
echo ">>> [BATCH START] Running NoPE..."
for SEED in "${SEEDS[@]}"; do
    for M_SIZE in "${MODELS[@]}"; do
        for SEQ_LEN in "${LENGTHS[@]}"; do
            CUR_MICRO_BS=$(get_mbs $SEQ_LEN)
            RUN_ID="baseline_nope_${M_SIZE}_L${SEQ_LEN}"
            if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
            
            OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

            echo ">>> [NoPE] Model: $M_SIZE | Len: $SEQ_LEN | MBS: $CUR_MICRO_BS | SEED: $SEED"
            $PYTHON_BIN $SCRIPT \
                --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
                --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
                --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
                --nope \
                $LIMIT_ARGS --seed $SEED
        done
    done
done

# ============================================================
# 3. Baseline: XPos
# ============================================================
echo ">>> [BATCH START] Running XPos..."
for SEED in "${SEEDS[@]}"; do
    for M_SIZE in "${MODELS[@]}"; do
        for SEQ_LEN in "${LENGTHS[@]}"; do
            CUR_MICRO_BS=$(get_mbs $SEQ_LEN)
            RUN_ID="baseline_xpos_${M_SIZE}_L${SEQ_LEN}"
            if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
            
            OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

            echo ">>> [XPos] Model: $M_SIZE | Len: $SEQ_LEN | MBS: $CUR_MICRO_BS | SEED: $SEED"
            $PYTHON_BIN $SCRIPT \
                --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
                --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
                --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
                --xpos \
                $LIMIT_ARGS --seed $SEED
        done
    done
done

# ============================================================
# 4. Baseline: ALiBi
# ============================================================
echo ">>> [BATCH START] Running ALiBi..."
for SEED in "${SEEDS[@]}"; do
    for M_SIZE in "${MODELS[@]}"; do
        for SEQ_LEN in "${LENGTHS[@]}"; do
            
            # ALiBi 关了 FlashAttention，极其吃显存，MBS 强制设为 1 或更低
            CUR_MICRO_BS=1

            RUN_ID="baseline_alibi_${M_SIZE}_L${SEQ_LEN}"
            if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
            
            OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

            echo ">>> [ALiBi] Model: $M_SIZE | Len: $SEQ_LEN | MBS: $CUR_MICRO_BS | SEED: $SEED"
            $PYTHON_BIN $SCRIPT \
                --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
                --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
                --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
                --alibi \
                $LIMIT_ARGS --seed $SEED
        done
    done
done

echo ">>> All 300M Baselines Completed."