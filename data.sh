#!/bin/bash
#SBATCH --job-name=30M_preprocess
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/30M_preprocess_%j.log
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err

export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /home/qijunrong/03-proj/PE
# 下载：/home/qijunrong/anaconda3/bin/python download_c4.py
# /home/qijunrong/anaconda3/bin/python concat_c4.py
/home/qijunrong/anaconda3/bin/python 30M_preprocess_train_data.py