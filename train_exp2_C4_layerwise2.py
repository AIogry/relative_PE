import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import islice
import argparse
import os
import math
import json
import random
import numpy as np
from torch.amp import autocast
from transformers import AutoTokenizer

# Imports from OLMo and Datasets
from OLMo.olmo.config import ModelConfig
from OLMo.olmo.model import OLMo
from datasets import load_from_disk, interleave_datasets

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    
    # === 本地路径配置 ===
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path containing c4_30M_train/val")
    parser.add_argument("--local_tokenizer_path", type=str, required=True, help="Path to local tokenizer directory")
    
    parser.add_argument("--train_size", type=int, default=1000000, help="Max training samples to load")
    parser.add_argument("--val_size", type=int, default=5000, help="Max validation samples to load")

    # === 模型大小 ===
    parser.add_argument("--model_size", type=str, default="20M", choices=["20M", "60M"])

    # === Strong Baselines 参数 ===
    parser.add_argument("--alibi", action="store_true", help="Enable ALiBi")
    parser.add_argument("--fope", action="store_true", help="Enable FoPE (Linear Scaling)")
    parser.add_argument("--yarn", action="store_true", help="Enable YaRN")
    # [新增] NoPE 和 XPos 参数
    parser.add_argument("--nope", action="store_true", help="Enable NoPE (No Positional Encoding)")
    parser.add_argument("--xpos", action="store_true", help="Enable XPos (Extrapolatable Position Embedding)")
    
    parser.add_argument("--rope_scale", type=float, default=None, help="Linear scaling factor")
    
    # === Bio-Gradient 参数 ===
    parser.add_argument("--use_scaled_rope", action="store_true")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--rope_scaling_threshold", type=int, default=-1)
    parser.add_argument("--sigma_list", nargs='+', default=None)

    # === 训练超参 ===
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=8) 
    parser.add_argument("--max_tokens", type=int, default=100_000_000)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=20)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sigma 处理
    final_sigmas = None
    if args.sigma_list is not None:
        final_sigmas = []
        for s in args.sigma_list:
            if s == "None": final_sigmas.append(None)
            else: final_sigmas.append(float(s))

    # === 1. Tokenizer 加载 ===
    print(f"Loading Tokenizer from LOCAL path: {args.local_tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer_path, local_files_only=True)
    except Exception as e:
        print(f"Error loading AutoTokenizer: {e}")
        from OLMo.olmo.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_pretrained(args.local_tokenizer_path, eos_token_id=50256, pad_token_id=50256)
    
    raw_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    vocab_size = ((raw_vocab_size + 63) // 64) * 64
    print(f">>> Resizing Vocab: {raw_vocab_size} -> {vocab_size}")

    # === 2. C4 数据加载 ===
    print(f"Loading C4 Data from: {args.dataset_path}")
    train_path = os.path.join(args.dataset_path, "c4_30M_train")
    val_path = os.path.join(args.dataset_path, "c4_30M_validation")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_full = load_from_disk(train_path)
    val_full = load_from_disk(val_path)

    real_train_size = min(args.train_size, len(train_full))
    real_val_size = min(args.val_size, len(val_full))
    print(f"Selecting subset: Train={real_train_size}, Val={real_val_size}")
    
    hf_train_datasets = [train_full.select(range(real_train_size))]
    eval_dataset = val_full.select(range(real_val_size))

    # 制作 Chunk 函数
    chunk_size = args.seq_len + 1
    
    def make_chunk_fn(chunk_len):
        def tokenize_and_chunk(examples):
            all_token_ids = []
            for text in examples.get("text", []):
                if text:
                    if hasattr(tokenizer, 'encode'):
                        ids = tokenizer.encode(text, add_special_tokens=False)
                    else: 
                        ids = tokenizer.encode(text, add_special_tokens=False)
                    all_token_ids.extend(ids)
            
            total_tokens = (len(all_token_ids) // chunk_len) * chunk_len
            chunks = [all_token_ids[i:i+chunk_len] for i in range(0, total_tokens, chunk_len)]
            return {"input_ids": chunks}
        return tokenize_and_chunk

    print("Tokenizing and Chunking Training Data (Using 8 processes)...")
    tokenize_train = make_chunk_fn(chunk_size)
    
    # [优化] 开启 num_proc=8 多进程处理，大幅加速 C4 数据处理
    processed_train_datasets = [
        ds.map(tokenize_train, batched=True, batch_size=1000, remove_columns=ds.column_names, num_proc=8)
        for ds in hf_train_datasets
    ]
    interleaved_train_dataset = interleave_datasets(processed_train_datasets)
    
    print("Tokenizing and Chunking Validation Data...")
    processed_eval_dataset = eval_dataset.map(tokenize_train, batched=True, batch_size=1000, remove_columns=eval_dataset.column_names, num_proc=4)

    def collate_fn(batch):
        data = [item['input_ids'] for item in batch]
        data = torch.tensor(data, dtype=torch.long)
        x = data[:, :-1].contiguous()
        y = data[:, 1:].contiguous()
        return x, y

    train_loader = DataLoader(interleaved_train_dataset, batch_size=args.micro_batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(processed_eval_dataset, batch_size=args.micro_batch_size, collate_fn=collate_fn)

    grad_accum_steps = args.global_batch_size // args.micro_batch_size

    if args.max_train_steps is not None and args.max_train_steps > 0:
        total_steps = args.max_train_steps
        print(f">>> [DEBUG MODE] Training for fixed steps: {total_steps}")
    else:
        total_steps = args.max_tokens // (args.global_batch_size * args.seq_len)
        print(f">>> [FULL MODE] Training for max tokens: {args.max_tokens} (~{total_steps} steps)")

    # === 模型配置 ===
    if args.model_size == "20M":
        current_d_model = 256
        print(">>> Using OLMo-20M Configuration")
    elif args.model_size == "60M":
        current_d_model = 512
        print(">>> Using OLMo-60M Configuration")
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    # =========================================================================
    # [Baseline Configuration Logic] - 支持 NoPE/XPos
    # =========================================================================
    use_alibi = False
    use_fope = False
    use_yarn = False
    use_nope = False # [新增]
    use_xpos = False # [新增]
    use_rope = True 
    rope_scaling_config = None 
    use_flash_attention = True

    if args.alibi:
        use_alibi = True
        use_rope = False 
        use_flash_attention = False
        print(">>> [Config] ALiBi ENABLED | RoPE DISABLED | FlashAttention DISABLED")
    elif args.fope:
        use_fope = True
        scale = args.rope_scale if args.rope_scale else max(1.0, args.seq_len / 512.0)
        rope_scaling_config = {"type": "linear", "factor": scale}
        print(f">>> [Config] FoPE (Linear Scaling) ENABLED | Scale: {scale}")
    elif args.yarn:
        use_yarn = True
        print(f">>> [Config] YaRN ENABLED | Target Len: {args.seq_len}")
    elif args.nope: # [新增分支]
        use_nope = True
        use_rope = False
        use_fope = False
        print(">>> [Config] NoPE ENABLED | RoPE DISABLED")
    elif args.xpos: # [新增分支]
        use_xpos = True
        use_rope = False 
        print(">>> [Config] XPos ENABLED | RoPE DISABLED")

    cfg = ModelConfig(
        d_model=current_d_model, 
        n_heads=8, 
        n_layers=8, 
        mlp_ratio=8,
        max_sequence_length=args.seq_len,
        vocab_size=vocab_size,
        embedding_size=vocab_size, 
        
        rope=use_rope,
        alibi=use_alibi,
        fope=use_fope, 
        yarn_enabled=use_yarn,
        yarn_target_max_position_embeddings=args.seq_len if use_yarn else None,
        yarn_max_position_embeddings=512, 
        
        use_scaled_rope1=args.use_scaled_rope,
        scaled_rope_sigma=args.sigma,
        scaled_rope_sigmas=final_sigmas,
        rope_scaling_threshold=args.rope_scaling_threshold,
        flash_attention=use_flash_attention
    )

    if rope_scaling_config is not None:
        cfg.rope_scaling = rope_scaling_config
        if args.fope:
             cfg.scaled_rope_sigma = rope_scaling_config["factor"]

    # 动态注入配置
    if use_nope: cfg.nope = True
    if use_xpos: cfg.xpos = True

    model = OLMo(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"Experiment: {args.run_id} -> Saving to {args.output_dir}")
    
    model.train()
    step = 0
    total_loss = 0.0 
    
    optimizer.zero_grad()
    train_iter = iter(train_loader)
    
    log_path = os.path.join(args.output_dir, "log.txt")
    log_file = open(log_path, "w")
    LOG_INTERVAL = 10 

    while step < total_steps:
        current_step_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device), y.to(device)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=x)
                loss = nn.functional.cross_entropy(outputs.logits.view(-1, vocab_size), y.view(-1))
                loss = loss / grad_accum_steps
            
            loss.backward()
            current_step_loss += loss.item() 

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += current_step_loss
        step += 1
        
        if step % LOG_INTERVAL == 0:
            avg_loss = total_loss / LOG_INTERVAL
            ppl = math.exp(avg_loss) if avg_loss < 20 else 1e9
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")
            log_file.write(f"{step},{avg_loss},{ppl}\n")
            log_file.flush()
            total_loss = 0.0

        if step % args.eval_interval == 0:
            print(">>> Running Validation...")
            model.eval()
            val_loss_accum = 0.0
            val_count = 0
            val_iter = iter(val_loader)
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    try:
                        vx, vy = next(val_iter)
                    except StopIteration:
                        break
                    vx, vy = vx.to(device), vy.to(device)
                    
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(input_ids=vx)
                        loss = nn.functional.cross_entropy(outputs.logits.view(-1, vocab_size), vy.view(-1))
                    
                    val_loss_accum += loss.item()
                    val_count += 1
            
            if val_count > 0:
                avg_val_loss = val_loss_accum / val_count
                val_ppl = math.exp(avg_val_loss)
                print(f">>> VAL PPL: {val_ppl:.2f}")
                log_file.write(f"VAL,{step},{avg_val_loss},{val_ppl}\n")
            model.train()

    print("Saving model checkpoint...")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print("Training Finished.")
    log_file.close()

if __name__ == "__main__":
    main()