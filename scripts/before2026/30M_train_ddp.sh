#!/bin/bash -x

#SBATCH --job-name=olmo-60m-ScaledRotaryEmbedding-2gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/%x_%j.out
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00

export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Define cache paths for different training sizes
declare -a TRAIN_SIZES=(500000 1000000 2000000 5000000 10000000)
TRAIN_BASE_PATH="/data/qijunrong/03-proj/PE/preprocessed_data"
VAL_CACHE_PATH="/data/qijunrong/03-proj/PE/preprocessed_data/val_size_50000_processed.pkl"

# for shujun's ScaledRotaryEmbedding
declare -a DECAY_FUNCS=("gaussian" "exp" "power")
declare -a SIGMAS=(0.1 1.0 5.0 10.0 20.0 30.0)

for train_size in "${TRAIN_SIZES[@]}"; do
  for decay_func in "${DECAY_FUNCS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
      run_name="olmo-60m-ScaledRotaryEmbedding-${decay_func}-sigma${sigma}-train${train_size}-2gpu"
      echo "===================================================================="
      echo "Starting run: $run_name"
      echo "decay_func=$decay_func, sigma=$sigma, train_size=$train_size"
      echo "Using 2 GPUs"
      echo "===================================================================="

      TRAIN_CACHE_PATH="${TRAIN_BASE_PATH}/train_size_${train_size}_processed.pkl"
      
      /home/qijunrong/anaconda3/bin/python 30M_train_parallel.py \
          --config ./configs/olmo_60m.yaml \
          --position_embedding rope \
          --use_scaledrotaryembedding \
          --scaled_rope_sigma "$sigma" \
          --decay_func "$decay_func" \
          --run_name "$run_name" \
          --max_sequence_length 512 \
          --batch_size 64 \  # Double the batch size for 2 GPUs
          --micro_batch_size 32 \
          --max_steps 30000 \
          --save_interval 500 \
          --log_interval 100 \
          --seed 6198 \
          --train_cache_path "$TRAIN_CACHE_PATH" \
          --val_cache_path "$VAL_CACHE_PATH"

      echo "Finished run: $run_name"
      echo ""
    done
  done
done

echo "All parameter sweeps completed!"