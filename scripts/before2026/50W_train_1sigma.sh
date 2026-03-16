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

# declare -a SIGMAS=(55.0 60.0 65.0 70.0 80.0 90.0 100.0)

# (25.0 27.0 32.0 35.0 37.0 40.0 45.0 50.0)


# for sigma in "${SIGMAS[@]}"; do
#    run_name="olmo-60m-ScaledRotaryEmbedding-exp-${sigma}-50w"
#    # run_name="olmo-60m-ScaledRotaryEmbedding-${decay_func}-sigmas_seq"
#    echo "===================================================================="
#    echo "Starting run: $run_name"
#    echo "decay_func=exp, sigma=$sigma"
#    echo "===================================================================="

#    /home/qijunrong/anaconda3/bin/python 50W_train.py \
#        --config ./configs/olmo_60m.yaml \
#        --position_embedding rope \
#        --use_scaled_rope1 \
#        --scaled_rope_sigma  $sigma\
#        --decay_func "exp" \
#        --run_name "$run_name" \
#        --max_sequence_length 512 \
#        --batch_size 32 \
#        --micro_batch_size 16 \
#        --max_steps 30000 \
#        --save_interval 500 \
#        --log_interval 100 \
#        --seed 1024

#    echo "Finished run: $run_name"
#    echo ""
#done


# (25.0 27.0 32.0 35.0 37.0 40.0 45.0 50.0)

# echo "Baseline"
# 调整了save/log的步数: 500->100, 100->50
# 2025-12-3: 还原回log: 100, save: 500，max_step: 50000， 暂时还是用50万数据，测试大于sigma大于60的情况， 
# 2025-12-5: 测试100w数据
# /home/qijunrong/anaconda3/bin/python 50W_train.py \
#    --config ./configs/olmo_60m.yaml \
#    --run_name "olmo-60m-RoPEbaseline-singlegpu-100w" \
#    --position_embedding rope \
#    --max_sequence_length 512 \
#    --batch_size 32 \
#    --micro_batch_size 16 \
#    --max_steps 60000 \
#    --save_interval 1000 \
#    --log_interval 100 \
#    --seed 1024



# less data test
# 调整了save/log的步数: 500->100, 100->50       declare -a SIGMAS=(25.0 27.0 32.0 35.0 37.0 40.0 45.0 50.0 55.0 60.0 65.0 70.0 80.0 90.0 100.0)
# 2025-12-3: 还原回log: 100, save: 500，max_step: 50000， 暂时还是用50万数据，测试大于sigma大于60的情况， declare -a SIGMAS=(65.0 70.0 75.0 80.0 90.0 100.0)

# 2025-12-4: declare -a SIGMAS=(80.0 85.0 90.0 95.0 100.0 110.0 150.0 200.0)

# 2025-12-5: 测试100w数据 declare -a SIGMAS=(75.0 80.0 85.0)
declare -a SIGMAS=(30.0 40.0 50.0)
for sigma in "${SIGMAS[@]}"; do
    run_name="olmo-60m-ScaledRotaryEmbedding-exp-${sigma}-100w"
    # run_name="olmo-60m-ScaledRotaryEmbedding-${decay_func}-sigmas_seq"
    echo "===================================================================="
    echo "Starting run: $run_name"
    echo "decay_func=exp, sigma=$sigma"
    echo "===================================================================="

    /home/qijunrong/anaconda3/bin/python 50W_train.py \
        --config ./configs/olmo_60m.yaml \
        --position_embedding rope \
        --use_scaled_rope1 \
        --scaled_rope_sigma  $sigma\
        --decay_func "exp" \
        --run_name "$run_name" \
        --max_sequence_length 512 \
        --batch_size 32 \
        --micro_batch_size 16 \
        --max_steps 60000 \
        --save_interval 1000 \
        --log_interval 100 \
        --seed 6198

    echo "Finished run: $run_name"
    echo ""
done


# /home/qijunrong/anaconda3/bin/python 50W_train.py \
#    --config ./configs/olmo_60m.yaml \
#    --run_name "olmo-60m-RoPEbaseline-singlegpu" \
#    --position_embedding rope \
#    --max_sequence_length 512 \
#    --batch_size 32 \
#    --micro_batch_size 16 \
#    --max_steps 30000 \
#    --save_interval 500 \
#    --log_interval 100 \
#    --seed 1024