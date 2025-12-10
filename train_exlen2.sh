#!/bin/bash -x

#SBATCH --job-name=olmo-60m-one-sigma
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


# echo "Baseline"
# 2025-12-5 无长度泛化下测试更长序列、训练数据量为100万、训练步长60000
# 这个可以不用测试了
# /home/qijunrong/anaconda3/bin/python train_exlen.py \
#    --config ./configs/olmo_60m.yaml \
#    --run_name "olmo-60m-RoPEbaseline-singlegpu-100w-exlen" \
#    --position_embedding rope \
#    --train_max_sequence_length 512 \
#    --val_max_sequence_length 2048 \
#    --train_size 1000000 \
#    --val_size 10000 \
#    --batch_size 8 \
#    --micro_batch_size 4 \
#    --max_steps 60000 \
#    --eval_interval 500\
#    --save_interval 10000 \
#    --log_interval 200 \
#    --seed 6198


# declare -a SIGMAS=(75.0 85.0 70.0 80.0 60.0 90.0)
# for sigma in "${SIGMAS[@]}"; do
#    run_name="olmo-60m-ScaledRotaryEmbedding-exp-${sigma}-exlen-yarn-100w"
    # run_name="olmo-60m-ScaledRotaryEmbedding-${decay_func}-sigmas_seq"
#    echo "===================================================================="
#    echo "Starting run: $run_name"
#    echo "decay_func=exp, sigma=$sigma"
#    echo "===================================================================="

#    /home/qijunrong/anaconda3/bin/python train_exlen.py \
#        --config ./configs/olmo_60m.yaml \
#        --position_embedding rope \
#        --use_scaled_rope1 \
#        --scaled_rope_sigma  $sigma\
#        --decay_func "exp" \
#        --yarn_enabled \
#        --yarn_dynamic_scaling \
#        --yarn_max_position_embeddings 512 \
#        --yarn_beta_fast 32.0 \
#        --yarn_beta_slow 1.0 \
#        --run_name "$run_name" \
#        --train_max_sequence_length 512 \
#        --val_max_sequence_length 2048 \
#        --train_size 1000000 \
#        --val_size 10000 \
#        --batch_size 8 \
#        --micro_batch_size 4 \
#        --max_steps 60000 \
#        --eval_interval 500 \
#        --save_interval 10000 \
#        --log_interval 200 \
#        --seed 6198

#    echo "Finished run: $run_name"
#    echo ""
# done


declare -a SIGMAS=(20.0 40.0 60.0 80.0 100.0)
for sigma in "${SIGMAS[@]}"; do
    run_name="olmo-60m-Grid-exp-${sigma}-50w"
    # run_name="olmo-60m-ScaledRotaryEmbedding-${decay_func}-sigmas_seq"
    echo "===================================================================="
    echo "Starting run: $run_name"
    echo "decay_func=exp, sigma=$sigma"
    echo "===================================================================="

    /home/qijunrong/anaconda3/bin/python train_exlen.py \
        --config ./configs/olmo_60m.yaml \
        --position_embedding grid \
        --grid_sigma  $sigma\
        --decay_func "exp" \
        --run_name "$run_name" \
        --train_max_sequence_length 512 \
        --val_max_sequence_length 512 \
        --train_size 500000 \
        --val_size 10000 \
        --batch_size 16 \
        --micro_batch_size 8 \
        --max_steps 60000 \
        --eval_interval 1000 \
        --save_interval 10000 \
        --log_interval 200 \
        --seed 6198

    echo "Finished run: $run_name"
    echo ""
done