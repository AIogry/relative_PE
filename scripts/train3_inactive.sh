#!/bin/bash -x


#SBATCH --job-name=olmo-60m-scaledRoPE-single-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/%x_%j.out
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00


export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8


declare -a SIN_LAMBDAS=(10.0 20.0 50.0 100.0)
declare -a COS_SIGMAS=(1.0 5.0 10.0 20.0 30.0)

# 循环所有组合

for sin_lambda in "${SIN_LAMBDAS[@]}"; do
  for cos_sigma in "${COS_SIGMAS[@]}"; do
    run_name="olmo-60m-scaledRoPE-lambda${sin_lambda}-sigma${cos_sigma}"
    echo "===================================================================="
    echo "Starting run: $run_name"
    echo "sin_lambda=$sin_lambda, cos_sigma=$cos_sigma"
    echo "===================================================================="

    /home/qijunrong/anaconda3/bin/python train3.py \
        --config ./configs/olmo_60m.yaml \
        --use_scaled_rope \
        --sin_lambda "$sin_lambda" \
        --cos_sigma "$cos_sigma" \
        --run_name "$run_name" \
        --max_sequence_length 512 \
        --batch_size 32 \
        --micro_batch_size 32 \
        --max_steps 30000 \
        --save_interval 500 \
        --log_interval 100 \
        --seed 1024

    echo "Finished run: $run_name"
    echo ""
  done
done

echo "All parameter sweeps completed!"