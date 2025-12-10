#!/bin/bash -x

#SBATCH --job-name=olmo-60m-ScaledRotary-sweep-ddp
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/qijunrong/03-proj/PE/logs/%x_%j.out
#SBATCH --error=/home/qijunrong/03-proj/PE/logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00

export PYTHONPATH="/home/qijunrong/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/qijunrong/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8


torchrun \
    --nproc_per_node=2 \
    --master_port=29502 \
    50W_train_ddp.py \
    --config ./configs/olmo_60m.yaml \
    --position_embedding rope \
    --run_name "olmo-60m-RoPEbaseline-ddp" \
    --max_sequence_length 512 \
    --batch_size 60 \
    --micro_batch_size 30 \
    --max_steps 30000 \
    --save_interval 500 \
    --log_interval 100 \
    --seed 6198


# for shujun's ScaledRotaryEmbedding
# declare -a DECAY_FUNCS=("gaussian" "exp" "power")

#for decay_func in "${DECAY_FUNCS[@]}"; do
#    run_name="olmo-60m-ScaledRotaryEmbedding-${decay_func}-sigma_seq_ddp2"
#    echo "===================================================================="
#    echo "Starting run: $run_name"
#    echo "decay_func=$decay_func"
#    echo "===================================================================="

    # Use torchrun for DDP
#    torchrun \
#        --nproc_per_node=2 \
#        --master_port=29501 \
#        50W_train_ddp.py \
#        --config ./configs/olmo_60m.yaml \
#        --position_embedding rope \
#        --use_scaledrotaryembedding \
#        --sigmas 1.0 5.0 20.0 20.0 25.0 25.0 30.0 30.0 \
#        --decay_func "$decay_func" \
#        --run_name "$run_name" \
#        --max_sequence_length 512 \
#        --batch_size 60 \
#        --micro_batch_size 30 \
#        --max_steps 30000 \
#        --save_interval 500 \
#        --log_interval 100 \
#        --seed 6198

#    echo "Finished run: $run_name"
#    echo ""
#done

#echo "All parameter sweeps completed!"