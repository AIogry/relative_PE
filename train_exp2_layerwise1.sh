#!/bin/bash

#SBATCH --job-name=exp2-baselines
#SBATCH --output=./logs/exp2-baselines_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_layerwise.py" 

# === 路径配置 ===
CHECKPOINT_ROOT="/data/qijunrong/03-proj/PE/checkpoints_variable_len"
LOCAL_DATA="/data/qijunrong/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

mkdir -p $CHECKPOINT_ROOT
mkdir -p ./logs

# === 全局配置 ===
GLOBAL_BS=64
SEED=6198
MAX_TOKENS=100000000 # 1亿 Token

# === DEBUG 配置 ===
DEBUG_STEPS="" 
# DEBUG_STEPS=100  # <--- 取消注释开启调试

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
    
    # 稳妥的显存设置：20M -> 16, 60M -> 8
    local mbs=32
    if [ "$m_size" == "60M" ]; then mbs=16; fi
    if [ "$seq_len" -ge 2048 ]; then mbs=$((mbs / 2)); fi
    echo $mbs
}


# ============================================================
# 3. Baseline: XPos
# ============================================================
echo ">>> [BATCH START] Running XPos..."
for M_SIZE in "${MODELS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do
        CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
        RUN_ID="baseline_xpos_${M_SIZE}_L${SEQ_LEN}"
        
        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

        echo ">>> [XPos] Model: $M_SIZE | Len: $SEQ_LEN"
        $PYTHON_BIN $SCRIPT \
            --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
            --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
            --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
            --xpos \
            $LIMIT_ARGS --seed $SEED
    done
done


# ============================================================
# 5. Baseline: ALiBi
# ============================================================
# 注意：你的 Python 脚本中 ALiBi 关闭了 FlashAttention，
# 显存开销会大幅增加 (O(N^2))，这里将 Micro Batch Size 减半以防 OOM。

echo ">>> [BATCH START] Running ALiBi..."
for M_SIZE in "${MODELS[@]}"; do
   for SEQ_LEN in "${LENGTHS[@]}"; do
       
       # 获取基础 MBS
       BASE_MBS=$(get_mbs $M_SIZE $SEQ_LEN)
       
       # 针对 ALiBi 减半 MBS (因为没有 FlashAttn)不减半
       # CUR_MICRO_BS=$((BASE_MBS / 2))
       if [ "$CUR_MICRO_BS" -lt 1 ]; then CUR_MICRO_BS=1; fi

       RUN_ID="baseline_alibi_${M_SIZE}_L${SEQ_LEN}"
       if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
       
       OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

       echo ">>> [ALiBi] Model: $M_SIZE | Len: $SEQ_LEN | MBS: $CUR_MICRO_BS (Reduced for No-FlashAttn)"
       $PYTHON_BIN $SCRIPT \
           --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
           --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
           --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
           --alibi \
           $LIMIT_ARGS --seed $SEED
   done
done


# ============================================================
# 1. Baseline: Standard RoPE
# ============================================================
echo ">>> [BATCH START] Running Standard RoPE..."
for M_SIZE in "${MODELS[@]}"; do
   for SEQ_LEN in "${LENGTHS[@]}"; do
       CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
       RUN_ID="baseline_rope_${M_SIZE}_L${SEQ_LEN}"
        if [ -n "$DEBUG_STEPS" ]; then RUN_ID="${RUN_ID}_debug"; fi
        
        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}" # 去掉时间戳方便覆盖或查看

        echo ">>> [RoPE] Model: $M_SIZE | Len: $SEQ_LEN"
        $PYTHON_BIN $SCRIPT \
            --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
            --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
            --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
            $LIMIT_ARGS --seed $SEED
    done
done

# ============================================================
# 2. Baseline: NoPE (No Positional Encoding)
# ============================================================
echo ">>> [BATCH START] Running NoPE..."
for M_SIZE in "${MODELS[@]}"; do
   for SEQ_LEN in "${LENGTHS[@]}"; do
        CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
        RUN_ID="baseline_nope_${M_SIZE}_L${SEQ_LEN}"
        
        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

        echo ">>> [NoPE] Model: $M_SIZE | Len: $SEQ_LEN"
        $PYTHON_BIN $SCRIPT \
            --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
            --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
            --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
            --nope \
            $LIMIT_ARGS --seed $SEED
    done
done


# ============================================================
# 4. Baseline: FoPE
# ============================================================
# echo ">>> [BATCH START] Running FoPE..."
# for M_SIZE in "${MODELS[@]}"; do
#    for SEQ_LEN in "${LENGTHS[@]}"; do
#        CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
#        SCALE=$(echo "scale=1; $SEQ_LEN / 512.0" | bc)
#        if (( $(echo "$SCALE < 1.0" | bc -l) )); then SCALE=1.0; fi
        
#        RUN_ID="baseline_fope_${M_SIZE}_L${SEQ_LEN}"
        
#        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"

#        echo ">>> [FoPE] Model: $M_SIZE | Len: $SEQ_LEN | Scale: $SCALE"
#        $PYTHON_BIN $SCRIPT \
#            --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
#            --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
#            --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
#            --fope --rope_scale $SCALE \
#            $LIMIT_ARGS --seed $SEED
#    done
# done

echo ">>> All Baselines Completed."