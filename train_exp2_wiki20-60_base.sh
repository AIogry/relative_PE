#!/bin/bash

#SBATCH --job-name=exp2-wiki20-60-base
#SBATCH --output=/data/qijunrong/03-proj/PE/logs/exp2_wiki20-60/tmp_base_%j.out
#SBATCH --error=/data/qijunrong/03-proj/PE/logs/exp2_wiki20-60/tmp_base_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# === 环境配置 ===
export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
# 统一Wandb离线配置，集中管理离线文件
export WANDB_MODE="offline"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_wiki20-60.py" 

# === 路径配置（核心修复：移除无效模板，定义实际根路径） ===
ROOT_DIR="/data/qijunrong/03-proj/PE"
# 日志目录：按日期分层，确保存在
LOG_DIR="${ROOT_DIR}/logs/exp2_wiki20-60/$(date +%Y%m%d)"
# Checkpoint根目录：按实验类型（base）固定，后续动态拼接子路径
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp2/wiki20-60/base"
# Wandb离线目录（已通过环境变量设置，此处仅创建目录）
WANDB_DIR="${ROOT_DIR}/wandb/offline/exp2_wiki20-60"

# 数据/Tokenizer路径
LOCAL_DATA="/data/qijunrong/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

# 创建所有必要目录（容错处理）
mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR


JOB_ID=${SLURM_JOB_ID}

# 3. 定义最终的日志文件路径（带日期）
FINAL_OUT="${LOG_DIR}/base_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/base_${JOB_ID}.err"


exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

# === 可选：删除SLURM的临时日志（避免冗余） ===
function cleanup {
    rm -f /data/qijunrong/03-proj/PE/logs/exp2_wiki20-60/tmp_base_${JOB_ID}.out
    rm -f /data/qijunrong/03-proj/PE/logs/exp2_wiki20-60/tmp_base_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"

# === 全局配置 ===
GLOBAL_BS=64
SEEDS=(6198 1024 7 568 3427)     # 5个随机种子
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

# ============================================================
# 核心函数1：计算Micro Batch Size（兼容ALiBi特殊逻辑）
# 参数：
# $1: 模型大小 (20M/60M)
# $2: 序列长度 (512/1024/2048)
# $3: baseline类型 (rope/xpos/nope/alibi/fope)
# ============================================================
get_mbs() {
    local m_size=$1
    local seq_len=$2
    local baseline_type=$3
    
    # 基础MBS配置
    local mbs=32
    if [ "$m_size" == "60M" ]; then mbs=16; fi
    if [ "$seq_len" -ge 2048 ]; then mbs=$((mbs / 2)); fi

    # ALiBi特殊处理（关闭FlashAttention，显存开销大）
    if [ "$baseline_type" == "alibi" ]; then
        if [ "$seq_len" -eq 2048 ]; then
            if [ "$m_size" == "20M" ]; then mbs=8; else mbs=4; fi
        fi
    fi

    echo $mbs
}

# ============================================================
# 核心函数2：统一运行Baseline实验
# 参数：
# $1: baseline类型 (rope/xpos/nope/alibi/fope)
# $2: 额外参数（如FoPE的--rope_scale，无则传空字符串）
# ============================================================
run_baseline_experiment() {
    local baseline_type=$1
    local extra_args=$2  # 存放FoPE等专属参数
    
    # 打印实验开始提示
    echo -e "\n>>> [BATCH START] Running $baseline_type..."
    
    # 遍历所有实验参数组合
    for SEED in "${SEEDS[@]}"; do
        for M_SIZE in "${MODELS[@]}"; do
            for SEQ_LEN in "${LENGTHS[@]}"; do
                # 1. 计算当前Micro Batch Size
                CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN $baseline_type)
                
                # 2. 生成RUN_ID
                RUN_ID="baseline_${baseline_type}_${M_SIZE}_L${SEQ_LEN}"
                
                # 3. 构建输出目录（按SEED分层，无TIMESTAMP）
                OUTPUT_DIR="${CHECKPOINT_ROOT}/${M_SIZE}/${RUN_ID}/seed_${SEED}"
                
                # 4. 打印当前实验信息
                echo ">>> [$baseline_type] Model: $M_SIZE | Len: $SEQ_LEN | MBS: $CUR_MICRO_BS | SEED: $SEED"
                
                # 5. 执行训练命令
                $PYTHON_BIN $SCRIPT \
                    --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
                    --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
                    --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
                    $extra_args \
                    $LIMIT_ARGS --seed $SEED
                
                # 6. 检查命令执行状态（可选，增强鲁棒性）
                if [ $? -ne 0 ]; then
                    echo ">>> [ERROR] $baseline_type实验失败！Model: $M_SIZE, Len: $SEQ_LEN, SEED: $SEED"
                    # 可选：失败后退出整个脚本（取消注释启用）
                    # exit 1
                fi
            done
        done
    done
}

# ============================================================
# 批量调用统一函数运行所有Baseline
# ============================================================
# 1. Standard RoPE（无额外参数）
run_baseline_experiment "rope" ""

# 2. XPos（额外参数：--xpos）
run_baseline_experiment "xpos" "--xpos"

# 3. NoPE（额外参数：--nope）
run_baseline_experiment "nope" "--nope"

# 4. ALiBi（额外参数：--alibi）
run_baseline_experiment "alibi" "--alibi"

# 5. FoPE（可选启用，需计算scale参数，单独处理）
# echo -e "\n>>> [BATCH START] Running FoPE..."
# for SEED in "${SEEDS[@]}"; do
#     for M_SIZE in "${MODELS[@]}"; do
#         for SEQ_LEN in "${LENGTHS[@]}"; do
#             CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN "fope")
#             # 计算FoPE缩放因子，确保不小于1.0
#             SCALE=$(echo "scale=1; $SEQ_LEN / 512.0" | bc)
#             if (( $(echo "$SCALE < 1.0" | bc -l) )); then SCALE=1.0; fi
            
#             RUN_ID="baseline_fope_${M_SIZE}_L${SEQ_LEN}"
#             OUTPUT_DIR="${CHECKPOINT_ROOT}/${M_SIZE}/${RUN_ID}/seed_${SEED}"

#             echo ">>> [FoPE] Model: $M_SIZE | Len: $SEQ_LEN | Scale: $SCALE | MBS: $CUR_MICRO_BS | SEED: $SEED"
#             $PYTHON_BIN $SCRIPT \
#                 --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
#                 --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
#                 --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
#                 --fope --rope_scale $SCALE \
#                 $LIMIT_ARGS --seed $SEED
#         done
#     done
# done

echo -e "\n>>> All Baselines Completed."