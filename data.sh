#!/bin/bash

#SBATCH --job-name=download_arxiv
#SBATCH --output=./logs/download_arxiv_%j.out
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# === 1. 环境变量设置 ===
PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"

# [极其重要] 设置外网代理 (请替换为你的真实IP和端口)
export HTTP_PROXY="http://你的电脑IP:端口"
export HTTPS_PROXY="http://你的电脑IP:端口"

# 设置 HuggingFace 缓存路径
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
mkdir -p $HF_HOME

# 创建日志目录
mkdir -p ./logs

echo "================================================="
echo ">>> Job started on $(hostname) at $(date)"
echo ">>> Working directory: $(pwd)"
echo ">>> Using Python: $PYTHON_BIN"
echo ">>> HF Cache: $HF_HOME"
echo "================================================="

# === 2. 准备运行环境 ===
# ⚠️ 这里必须加上 hf_olmo 和 ai2-olmo，否则最后下 Tokenizer 时会报错！
echo ">>> Installing dependencies..."
$PYTHON_BIN -m pip install "datasets==2.19.2" ai2-olmo -q

# === 3. 运行下载脚本 ===
echo ">>> Starting download..."
$PYTHON_BIN download_arxiv_data.py

# === 4. 恢复环境并结束 ===
echo ">>> Restoring datasets version..."
$PYTHON_BIN -m pip install --upgrade datasets -q
echo ">>> Job finished at $(date)"