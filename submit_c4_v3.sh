#!/bin/bash
# C4 V3 评估 - 修复OOM问题

cd /home/qijunrong/03-proj/PE

# RoPE baseline
echo "Submitting RoPE baseline..."
sbatch --job-name=c4v3_rope --partition=debug --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=/data/qijunrong/03-proj/PE/logs/c4_fixed_v3/%j_c4.out \
  --error=/data/qijunrong/03-proj/PE/logs/c4_fixed_v3/%j_c4.err \
  --wrap="python eval_fewshot_c4_fixed_v2.py --model_path /data/qijunrong/03-proj/PE/checkpoints/pretrain_c4/300M_rope_L512_sig0.0_s42/model_final.pt --model_size 300M --dataset_path /data/qijunrong/03-proj/PE/data/c4 --local_tokenizer_path /data/qijunrong/03-proj/PE/tokenizer --pe_type rope --few_shot_k 1000 --few_shot_steps 100 --few_shot_lr 5e-6 --base_len 512 --test_lengths 1024 2048 4096 --output_file /data/qijunrong/03-proj/PE/results/c4_fixed_v3/rope_c4_extrap.json"

# RoPE + YaRN
echo "Submitting RoPE+YaRN..."
sbatch --job-name=c4v3_rope_yarn --partition=debug --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=/data/qijunrong/03-proj/PE/logs/c4_fixed_v3/%j_c4.out \
  --error=/data/qijunrong/03-proj/PE/logs/c4_fixed_v3/%j_c4.err \
  --wrap="python eval_fewshot_c4_fixed_v2.py --model_path /data/qijunrong/03-proj/PE/checkpoints/pretrain_c4/300M_rope_L512_sig0.0_s42/model_final.pt --model_size 300M --dataset_path /data/qijunrong/03-proj/PE/data/c4 --local_tokenizer_path /data/qijunrong/03-proj/PE/tokenizer --pe_type rope_yarn --few_shot_k 1000 --few_shot_steps 100 --few_shot_lr 5e-6 --base_len 512 --test_lengths 1024 2048 4096 --output_file /data/qijunrong/03-proj/PE/results/c4_fixed_v3/rope_yarn_c4_extrap.json"

# HIPE
echo "Submitting HIPE..."
sbatch --job-name=c4v3_hipe --partition=debug --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=/data/qijunrong/03-proj/PE/logs/c4_fixed_v3/%j_c4.out \
  --error=/data/qijunrong/03-proj/PE/logs/c4_fixed_v3/%j_c4.err \
  --wrap="python eval_fewshot_c4_fixed_v2.py --model_path /data/qijunrong/03-proj/PE/checkpoints/pretrain_c4/300M_rope_L512_sig700.0_s42/model_final.pt --model_size 300M --dataset_path /data/qijunrong/03-proj/PE/data/c4 --local_tokenizer_path /data/qijunrong/03-proj/PE/tokenizer --pe_type hipe --sigma 700 --few_shot_k 1000 --few_shot_steps 100 --few_shot_lr 5e-6 --base_len 512 --test_lengths 1024 2048 4096 --output_file /data/qijunrong/03-proj/PE/results/c4_fixed_v3/hipe_c4_extrap.json"

# HIPE + YaRN
echo "Submitting HIPE+YaRN..."
sbatch --job-name=c4v3_hipe_yarn --partition=debug --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=/data/qijunrong/03-proj/PE/logs/c4_fixed_v3/%j_c4.out \
  --error=/data/qijunrong/03-proj/PE/logs/c4_fixed_v3/%j_c4.err \
  --wrap="python eval_fewshot_c4_fixed_v2.py --model_path /data/qijunrong/03-proj/PE/checkpoints/pretrain_c4/300M_rope_L512_sig700.0_s42/model_final.pt --model_size 300M --dataset_path /data/qijunrong/03-proj/PE/data/c4 --local_tokenizer_path /data/qijunrong/03-proj/PE/tokenizer --pe_type hipe_yarn --sigma 700 --few_shot_k 1000 --few_shot_steps 100 --few_shot_lr 5e-6 --base_len 512 --test_lengths 1024 2048 4096 --output_file /data/qijunrong/03-proj/PE/results/c4_fixed_v3/hipe_yarn_c4_extrap.json"

echo "All jobs submitted!"
