#!/bin/bash -x

#SBATCH --job-name=olmo-60m-ScaledRotary-sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/%x_%j.out
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00

export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

TRAIN_BASE_PATH="/data/qijunrong/03-proj/PE/preprocessed_data"
VAL_CACHE_PATH="/data/qijunrong/03-proj/PE/preprocessed_data/val_size_50000_processed.pkl"

# Define sigma sets and names
SIGMA_SETS=(
    "1.0 5.0 20.0 20.0 25.0 25.0 30.0 30.0"
    "0.5 1.0 2.0 5.0 10.0 15.0 20.0 40.0"
)
SIGMA_NAMES=("groupA" "groupB")

declare -a DECAY_FUNCS=("gaussian" "exp" "power")
declare -a TRAIN_SIZES=(500000 1000000)
declare -a LENGTHS=(512)  # 1024)

for train_size in "${TRAIN_SIZES[@]}"; do
    train_cache_path="${TRAIN_BASE_PATH}/train_size_${train_size}_processed.pkl"
    for decay_func in "${DECAY_FUNCS[@]}"; do
        for length in "${LENGTHS[@]}"; do
            for idx in "${!SIGMA_SETS[@]}"; do
                sigma_list="${SIGMA_SETS[$idx]}"
                sigma_name="${SIGMA_NAMES[$idx]}"
                run_name="olmo60m-train${train_size}-${decay_func}-len${length}-${sigma_name}"

                echo "===================================================================="
                echo "Starting run: $run_name"
                echo "decay_func=$decay_func, train_size=$train_size, length=$length"
                echo "sigma set: $sigma_name"
                echo "===================================================================="

                /home/qijunrong/anaconda3/bin/python 30M_train_single.py \
                    --config ./configs/olmo_60m.yaml \
                    --position_embedding rope \
                    --use_scaledrotaryembedding \
                    --scaled_rope_sigma 20.0\
                    --decay_func "$decay_func" \
                    --run_name "$run_name" \
                    --max_sequence_length "$length" \
                    --batch_size 16 \
                    --micro_batch_size 2 \
                    --max_steps 100000 \
                    --save_interval 500 \
                    --log_interval 100 \
                    --seed 6198 \
                    --train_cache_path "$train_cache_path" \
                    --val_cache_path "$VAL_CACHE_PATH"

                echo "Finished run: $run_name"
                echo ""
            done
        done
    done
done

echo "All parameter sweeps completed!"

                    #--sigmas $sigma_list \