#!/bin/bash -x


#SBATCH --job-name=olmo-60m-scaledRoPE-single-gpu-debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/%x_%j.out
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00


export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8



cd /home/qijunrong/03-proj/PE/

/home/qijunrong/anaconda3/bin/python debug.py \
    --config ./configs/olmo_60m.yaml \
    --use_scaled_rope \
    --sin_lambda 20.0 \
    --cos_sigma 5.0 \
    --run_name "olmo-60m-scaledRoPE-single-gpu-debug" \
    --max_sequence_length 512 \
    --batch_size 32 \
    --micro_batch_size 8 \
    --max_steps 100 \
    --save_interval 50 \
    --log_interval 10 \
    --seed 1024