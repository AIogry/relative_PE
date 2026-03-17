#!/bin/bash

#SBATCH --job-name=exp-c4-baselines
#SBATCH --output=./logs/exp2_c4_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"

# [修改] 更新为支持 NoPE/XPos 的新脚本文件名
SCRIPT="train_exp2_C4_layerwise2.py" 

# === 路径配置 (全部本地) ===
CHECKPOINT_ROOT="/data/qijunrong/03-proj/PE/checkpoints_c4"
C4_DATA_ROOT="/data/qijunrong/03-proj/PE"
LOCAL_TOKENIZER="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

mkdir -p $CHECKPOINT_ROOT
mkdir -p ./logs

# === 全局配置 ===
GLOBAL_BS=64
SEED=6198
MAX_TOKENS=1000000000 # 1B

# [关键修改] 大幅减少加载的数据量
# 30万条 C4 数据足够提供 >1.5亿 Token，满足 MAX_TOKENS 需求
TRAIN_SAMPLES=5000000      # 5M
VAL_SAMPLES=10000

# === DEBUG 开关 ===
DEBUG_STEPS="" 
# DEBUG_STEPS=50 

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS"
fi

# ============================================================
# 实验配置
# ============================================================
MODELS=("20M" "60M")
LENGTHS=(512 1024 2048)

get_mbs() {
    local m_size=$1
    local seq_len=$2
    local mbs=32
    if [ "$m_size" == "60M" ]; then mbs=16; fi
    if [ "$seq_len" -ge 2048 ]; then mbs=$((mbs / 2)); fi
    echo $mbs
}


# ============================================================
# 4. [新增] NoPE (No Positional Encoding) on C4
# ============================================================
echo ">>> [BATCH START] Running NoPE on C4..."

for M_SIZE in "${MODELS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do
        
        CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
        RUN_ID="c4_nope_${M_SIZE}_L${SEQ_LEN}"
        if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

        echo ">>> [NoPE] Model: $M_SIZE | Len: $SEQ_LEN | Micro BS: $CUR_MICRO_BS"
        
        $PYTHON_BIN $SCRIPT \
            --output_dir $OUTPUT_DIR \
            --run_id $RUN_ID \
            --model_size $M_SIZE \
            --dataset_path $C4_DATA_ROOT \
            --local_tokenizer_path $LOCAL_TOKENIZER \
            --seq_len $SEQ_LEN \
            --global_batch_size $GLOBAL_BS \
            --micro_batch_size $CUR_MICRO_BS \
            --train_size $TRAIN_SAMPLES \
            --val_size $VAL_SAMPLES \
            --nope \
            $LIMIT_ARGS \
            --seed $SEED
    done
done

# ============================================================
# 5. [新增] XPos on C4
# ============================================================
echo ">>> [BATCH START] Running XPos on C4..."

for M_SIZE in "${MODELS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do
        
        CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
        RUN_ID="c4_xpos_${M_SIZE}_L${SEQ_LEN}"
        if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

        echo ">>> [XPos] Model: $M_SIZE | Len: $SEQ_LEN | Micro BS: $CUR_MICRO_BS"
        
        $PYTHON_BIN $SCRIPT \
            --output_dir $OUTPUT_DIR \
            --run_id $RUN_ID \
            --model_size $M_SIZE \
            --dataset_path $C4_DATA_ROOT \
            --local_tokenizer_path $LOCAL_TOKENIZER \
            --seq_len $SEQ_LEN \
            --global_batch_size $GLOBAL_BS \
            --micro_batch_size $CUR_MICRO_BS \
            --train_size $TRAIN_SAMPLES \
            --val_size $VAL_SAMPLES \
            --xpos \
            $LIMIT_ARGS \
            --seed $SEED
    done
done


# ============================================================
# 1. Baseline (Standard RoPE) on C4
# ============================================================
echo ">>> [BATCH START] Running Baseline (RoPE) on C4..."

for M_SIZE in "${MODELS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do
        
        CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
        RUN_ID="c4_rope_${M_SIZE}_L${SEQ_LEN}"
        if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

        echo ">>> [RoPE] Model: $M_SIZE | Len: $SEQ_LEN | Micro BS: $CUR_MICRO_BS"
        
        $PYTHON_BIN $SCRIPT \
            --output_dir $OUTPUT_DIR \
            --run_id $RUN_ID \
            --model_size $M_SIZE \
            --dataset_path $C4_DATA_ROOT \
            --local_tokenizer_path $LOCAL_TOKENIZER \
            --seq_len $SEQ_LEN \
            --global_batch_size $GLOBAL_BS \
            --micro_batch_size $CUR_MICRO_BS \
            --train_size $TRAIN_SAMPLES \
            --val_size $VAL_SAMPLES \
            $LIMIT_ARGS \
            --seed $SEED
    done
done

# ============================================================
# 2. ALiBi on C4
# ============================================================
echo ">>> [BATCH START] Running ALiBi on C4..."

for M_SIZE in "${MODELS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do
        
        CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
        RUN_ID="c4_alibi_${M_SIZE}_L${SEQ_LEN}"
        if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

        echo ">>> [ALiBi] Model: $M_SIZE | Len: $SEQ_LEN | Micro BS: $CUR_MICRO_BS"
        
        $PYTHON_BIN $SCRIPT \
            --output_dir $OUTPUT_DIR \
            --run_id $RUN_ID \
            --model_size $M_SIZE \
            --dataset_path $C4_DATA_ROOT \
            --local_tokenizer_path $LOCAL_TOKENIZER \
            --seq_len $SEQ_LEN \
            --global_batch_size $GLOBAL_BS \
            --micro_batch_size $CUR_MICRO_BS \
            --train_size $TRAIN_SAMPLES \
            --val_size $VAL_SAMPLES \
            --alibi \
            $LIMIT_ARGS \
            --seed $SEED
    done
done

# ============================================================
# 3. FoPE (Linear Scaling) on C4
# ============================================================
# echo ">>> [BATCH START] Running FoPE on C4..."

# for M_SIZE in "${MODELS[@]}"; do
#    for SEQ_LEN in "${LENGTHS[@]}"; do
        
#        CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
#        SCALE=$(echo "scale=1; $SEQ_LEN / 512.0" | bc)
#        if (( $(echo "$SCALE < 1.0" | bc -l) )); then SCALE=1.0; fi

#        RUN_ID="c4_fope_${M_SIZE}_L${SEQ_LEN}"
#        if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
#        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

#        echo ">>> [FoPE] Model: $M_SIZE | Len: $SEQ_LEN | Scale: $SCALE | Micro BS: $CUR_MICRO_BS"
        
#        $PYTHON_BIN $SCRIPT \
#            --output_dir $OUTPUT_DIR \
#            --run_id $RUN_ID \
#            --model_size $M_SIZE \
#            --dataset_path $C4_DATA_ROOT \
#            --local_tokenizer_path $LOCAL_TOKENIZER \
#            --seq_len $SEQ_LEN \
#            --global_batch_size $GLOBAL_BS \
#            --micro_batch_size $CUR_MICRO_BS \
#            --train_size $TRAIN_SAMPLES \
#            --val_size $VAL_SAMPLES \
#            --fope \
#            --rope_scale $SCALE \
#            $LIMIT_ARGS \
#            --seed $SEED
#    done
#done



echo ">>> All C4 Experiments Completed."