#!/bin/bash
#SBATCH --job-name=exp2-300m-hipe
#SBATCH --output=./logs/exp2_300m_hipe_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
SCRIPT="train_exp2_wikifull.py"

CHECKPOINT_ROOT="/data/qijunrong/03-proj/PE/checkpoints_variable_len"
LOCAL_DATA="/data/qijunrong/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/qijunrong/03-proj/PE/wikitext/tokenizer"

GLOBAL_BS=64
SEEDS=(6198 1024 7 568 3427)
MAX_TOKENS=100000000
TRAIN_ARGS="--max_tokens $MAX_TOKENS --lr 3e-4"

# 参数空间
SIGMAS=(50.0 100.0 200.0 500.0 700.0 1000.0)
THRESHOLDS=(3)
MODEL="300M"
LENGTHS=(512 1024 2048)

echo ">>> Starting PHASE 3: 300M HIPE Experiments..."

for SEED in "${SEEDS[@]}"; do
    for len in "${LENGTHS[@]}"; do
        for sigma in "${SIGMAS[@]}"; do
            for thr in "${THRESHOLDS[@]}"; do
                
                # 显存控制
                CUR_MICRO_BS=8
                if [ "$len" -ge 2048 ]; then CUR_MICRO_BS=4; fi
                
                TIMESTAMP=$(date +%Y%m%d_%H%M%S)
                RUN_ID="${MODEL}_L${len}_sig${sigma}_thr${thr}"
                OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}_${TIMESTAMP}"

                echo ">>> [HIPE] Len: $len | Sig: $sigma | Thr: $thr | Seed: $SEED | MBS: $CUR_MICRO_BS"
                $PYTHON_BIN $SCRIPT \
                    --output_dir $OUTPUT_DIR \
                    --run_id $RUN_ID \
                    --model_size $MODEL \
                    --local_data_path $LOCAL_DATA \
                    --local_tokenizer_path $LOCAL_TOKENIZER \
                    --seq_len $len \
                    --global_batch_size $GLOBAL_BS \
                    --micro_batch_size $CUR_MICRO_BS \
                    --seed $SEED \
                    --use_scaled_rope \
                    --sigma $sigma \
                    --rope_scaling_threshold $thr \
                    $TRAIN_ARGS
            done
        done
    done
done
echo ">>> All 300M HIPE Experiments Finished."