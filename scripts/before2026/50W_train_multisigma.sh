#!/bin/bash -x

#SBATCH --job-name=olmo-60m-ScaledRotary-sweep
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



# for shujun's ScaledRotaryEmbedding
# 调整了save/log的步数: 500->100, 100->50
declare -a DECAY_FUNCS=("exp" "power")      # "gaussian" 
# declare -a SIGMA_SEQUENCES=(           2025-12-1测试了gaussian对这所有的效果
#    "10.0 10.0 37.0 37.0 37.0 37.0 55.0 55.0"
#    "5.0 5.0 18.5 18.5 18.5 18.5 27.5 27.5"
#    "8.1 8.1 30.0 30.0 30.0 30.0 44.6 44.6"
#    "15.0 15.0 55.5	55.5 55.5 55.5 82.5 82.5"
#    "20.0 20.0 74.0 74.0 74.0 74.0 110.0 110.0"
#    "11.0 11.0 30.7 30.7 30.7 30.7 41.0 41.0"
#    "16.5 16.5 46.05 46.05 46.05 46.05 61.5 61.5"
#    "10.0 37.0 37.0 37.0 37.0 55.0 55.0 55.0"
#    "11.0 30.7 30.7 30.7 30.7 41.0 41.0 41.0"
#)

# 2025-12-3 save/log换回500, 100
# declare -a SIGMA_SEQUENCES=(
#    "11.0 30.7 30.7 30.7 30.7 41.0 41.0 41.0"
#    "11.0 11.0 30.7 30.7 30.7 30.7 41.0 41.0"
#)

declare -a SIGMA_SEQUENCES=(
    "22.9 85.0 85.0 85.0 85.0 126.4 126.4 126.4"
    "29.7 82.9 82.9 82.9 82.9 110.7 110.7 110.7"
    "20.25 75.0 75.0 75.0 75.0 111.5 111.5 111.5"
)

# 计数器，用于生成唯一的运行名称
counter=1

for decay_func in "${DECAY_FUNCS[@]}"; do
    for sigma_seq in "${SIGMA_SEQUENCES[@]}"; do
        # 将sigma序列转换为适合文件名的格式（用下划线替换空格和点）
        # sigma_seq_filename=$(echo "$sigma_seq")
        run_name="olmo-60m-ScaledRotaryEmbedding-${decay_func}-50w-seq${counter}-${sigma_seq}"
        # run_name="olmo-60m-ScaledRotaryEmbedding-${decay_func}-sigmas_seq"
        echo "===================================================================="
        echo "Starting run: $run_name"
        echo "decay_func=$decay_func, sigma=$sigma_seq"
        echo "===================================================================="

        /home/qijunrong/anaconda3/bin/python 50W_train.py \
            --config ./configs/olmo_60m.yaml \
            --position_embedding rope \
            --use_scaled_rope1 \
            --sigmas  $sigma_seq\
            --decay_func "$decay_func" \
            --run_name "$run_name" \
            --max_sequence_length 512 \
            --batch_size 32 \
            --micro_batch_size 16 \
            --max_steps 30000 \
            --save_interval 500 \
            --log_interval 100 \
            --seed 6198

        echo "Finished run: $run_name"
        echo ""

        ((counter++))
    done
done

# 2025-11-24测试了1.0 5.0 20.0 20.0 25.0 25.0 30.0 30.0
echo "All parameter sweeps completed!"


# echo "Baseline"
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
#    --seed 6198




# declare -a SIN_LAMBDAS=(10.0 20.0 50.0 100.0)
# declare -a COS_SIGMAS=(1.0 5.0 10.0 20.0 30.0)

# for sin_lambda in "${SIN_LAMBDAS[@]}"; do
#   for cos_sigma in "${COS_SIGMAS[@]}"; do
#     run_name="olmo-60m-scaledRoPE-lambda${sin_lambda}-sigma${cos_sigma}"
#     echo "===================================================================="
#     echo "Starting run: $run_name"
#     echo "sin_lambda=$sin_lambda, cos_sigma=$cos_sigma"
#     echo "===================================================================="

#     /home/qijunrong/anaconda3/bin/python train3.py \
#         --config ./configs/olmo_60m.yaml \
#         --use_scaled_rope \
#         --sin_lambda "$sin_lambda" \
#         --cos_sigma "$cos_sigma" \
#         --run_name "$run_name" \
#         --max_sequence_length 512 \
#         --batch_size 32 \
#         --micro_batch_size 16 \
#         --max_steps 30000 \
#         --save_interval 500 \
#         --log_interval 100 \
#         --seed 6198

#     echo "Finished run: $run_name"
#     echo ""
#   done
# done

# echo "All parameter sweeps completed!"


# baseline RoPE
# /home/qijunrong/anaconda3/bin/python train3.py \
#     --config ./configs/olmo_60m.yaml \
#     --run_name "olmo-60m-RoPEbaseline-singlegpu" \
#     --max_sequence_length 512 \
#     --batch_size 32 \
#     --micro_batch_size 16 \
#     --max_steps 30000 \
#     --save_interval 500 \
#     --log_interval 100 \
#     --seed 6198




