#!/bin/bash
#SBATCH --job-name=build_c4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/build_c4_%j.log

cd /home/qijunrong/03-proj/PE/
/home/qijunrong/anaconda3/bin/python build_c4_dataset.py