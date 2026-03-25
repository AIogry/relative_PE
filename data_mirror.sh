#!/bin/bash
#SBATCH --job-name=mirror_arxiv
#SBATCH --output=./logs/download_arxiv_mirror%j.out
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"

# 1. 设置国内镜像环境变量 (核心步骤)
export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_OFFLINE=0

# 2. 建议降级 datasets 库以确保兼容性
$PYTHON_BIN -m pip install "datasets==2.19.2" -q

# 3. 运行下载
echo ">>> 开始通过国内镜像下载..."
$PYTHON_BIN download_arxiv_mirror.py