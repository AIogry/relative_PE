#!/bin/bash -x

#SBATCH --job-name=preprocess-data-debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/%x_%j.out
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:0
#SBATCH --time=96:00:00


export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8



cd /home/qijunrong/03-proj/PE/


/home/qijunrong/anaconda3/bin/python 30M_preprocess_validation_data.py
# 30M_preprocess_test_data.py