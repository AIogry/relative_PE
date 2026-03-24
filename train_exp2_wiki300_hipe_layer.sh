#!/bin/bash

#SBATCH --job-name=exp2-wiki300-hipe
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/exp2_wiki300/tmp_hipe_%j.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/exp2_wiki300/tmp_hipe_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
export WANDB_MODE="offline"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_wikifull.py" 

# === 路径配置 ===
ROOT_DIR="/data/qijunrong/03-proj/PE"
LOG_DIR="${ROOT_DIR}/logs/exp2_wiki300/$(date +%Y%m%d)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp2/wiki300/hipe"
WANDB_DIR="${ROOT_DIR}/wandb/offline/exp2_wiki300"

LOCAL_DATA="/data/qijunrong/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR

JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/hipe_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/hipe_${JOB_ID}.err"
exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f /data/qijunrong/03-proj/PE/logs/exp2_wiki300/tmp_hipe_${JOB_ID}.out
    rm -f /data/qijunrong/03-proj/PE/logs/exp2_wiki300/tmp_hipe_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"

# === 全局配置 ===
GLOBAL_BS=64
SEEDS=(6198)    # 先只测试一个seed，看一下结果能怎么样 (1024 7 568 3427)
MAX_TOKENS=100000000 

# === DEBUG 配置 ===
DEBUG_STEPS="" 
# DEBUG_STEPS=20 

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS --lr 3e-4"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS --lr 3e-4"
fi

# ============================================================
# [核心修改 1] 16 层模型的逐层 Sigma 分布方案 (Layer-wise Schemes)
# ============================================================
# 这里定义的是 16 层模型每一层对应的 Sigma 值。None 代表使用原版 RoPE。
# 方案 A: 底部锚定 (前4层RoPE，后12层缩放，等价于以前的 threshold=3)
SIGMA_BOTTLE1="None None None None 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 None None None None"

# 方案 B: 对半开 (前8层RoPE，后8层缩放，等价于以前的 threshold=7)
SIGMA_BOTTLE2="None None None None 500.0 500.0 500.0 500.0 500.0 500.0 500.0 500.0 None None None None"

# 方案 C: 多阶段平滑递增 (0-3: 纯正, 4-7: 轻微, 8-11: 中度, 12-15: 极限)
SIGMA_BOTTLE3="None None None None 700.0 700.0 700.0 700.0 700.0 700.0 700.0 700.0 None None None None"

# 使用字典将方案名和对应的字符串映射起来
declare -A SCHEMES
SCHEMES=( ["bottle1"]="$SIGMA_BOTTLE1" ["bottle2"]="$SIGMA_BOTTLE2" ["bottle3"]="$SIGMA_BOTTLE3" )

MODELS=("300M")
LENGTHS=(512 1024 2048)

# ============================================================
# 核心函数：计算Micro Batch Size (300M 专属)
# ============================================================
get_mbs() {
    local m_size=$1
    local seq_len=$2
    
    local mbs=16
    if [ "$seq_len" -ge 2048 ]; then mbs=8; fi

    echo $mbs
}

# ============================================================
# 核心函数：执行单次HIPE实验
# ============================================================
run_experiment() {
    local M_SIZE=$1
    local SEQ_LEN=$2
    local SCHEME_NAME=$3   # [核心修改 2] 接收方案名
    local SIGMA_STR=$4     # [核心修改 2] 接收16层的分布字符串
    local SEED=$5
    
    CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
    
    # [核心修改 3] 更新 RUN_ID 命名，使用 scheme 名称代替单一的 sigma 和 thr
    RUN_ID="hipe_${M_SIZE}_L${SEQ_LEN}_scheme-${SCHEME_NAME}"
    OUTPUT_DIR="${CHECKPOINT_ROOT}/${M_SIZE}/${RUN_ID}/seed_${SEED}"

    echo -e "\n>>> [HIPE Layer-wise] Model: $M_SIZE | Len: $SEQ_LEN | Scheme: $SCHEME_NAME | MBS: $CUR_MICRO_BS | SEED: $SEED"
    echo ">>> Output Dir: $OUTPUT_DIR"
    
    $PYTHON_BIN $SCRIPT \
        --output_dir $OUTPUT_DIR \
        --run_id $RUN_ID \
        --model_size $M_SIZE \
        --local_data_path $LOCAL_DATA \
        --local_tokenizer_path $LOCAL_TOKENIZER \
        --wandb_dir $WANDB_DIR \
        --wandb_mode $WANDB_MODE \
        --seq_len $SEQ_LEN \
        --global_batch_size $GLOBAL_BS \
        --micro_batch_size $CUR_MICRO_BS \
        --seed $SEED \
        --use_scaled_rope \
        --sigma_list $SIGMA_STR \
        --rope_scaling_threshold -1 \
        $LIMIT_ARGS

    if [ $? -ne 0 ]; then
        echo ">>> [ERROR] HIPE实验失败！Model: $M_SIZE, Len: $SEQ_LEN, Scheme: $SCHEME_NAME, SEED: $SEED"
    fi
}

echo -e "\n>>> [BATCH START] Running 300M HIPE experiments (Layer-wise)..."

# [核心修改 4] 循环结构调整：遍历 SCHEMES 字典，而不是嵌套遍历单一变量
for SEED in "${SEEDS[@]}"; do
    for M_SIZE in "${MODELS[@]}"; do
        for SEQ_LEN in "${LENGTHS[@]}"; do
            for SCHEME_NAME in "${!SCHEMES[@]}"; do
                run_experiment $M_SIZE $SEQ_LEN $SCHEME_NAME "${SCHEMES[$SCHEME_NAME]}" $SEED
            done
        done
    done
done

echo -e "\n>>> All 300M HIPE Experiments Completed."