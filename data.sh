#!/bin/bash

#SBATCH --job-name=download_pg19
#SBATCH --output=./logs/download_pg19_%j.out
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# === 环境变量设置 ===
# 1. 设置 Python 路径 (使用您之前的环境)
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"

# 2. [重要] 设置 HuggingFace 缓存路径
# 默认情况下 HF 会下载到 ~/.cache，这可能会撑爆 Home 目录配额。
# 我们将其临时指向数据盘的一个缓存目录。
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
mkdir -p $HF_HOME

# 3. 创建日志目录
mkdir -p ./logs

echo ">>> Job started on $(hostname) at $(date)"
echo ">>> Working directory: $(pwd)"
echo ">>> Using Python: $PYTHON_BIN"
echo ">>> HF Cache: $HF_HOME"

# === 运行下载脚本 ===
# 假设你的 python 脚本名为 prepare_data.py
$PYTHON_BIN wiki_data.py

echo ">>> Job finished at $(date)"