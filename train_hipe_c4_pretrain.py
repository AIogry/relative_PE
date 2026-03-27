"""
预训练脚本：C4数据集 + HIPE（可选YaRN）
用于预训练阶段，序列长度固定为512
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import math
import random
import numpy as np
import wandb
import subprocess
import sys
from torch.amp import autocast
from transformers import AutoTokenizer

from OLMo.olmo.config import ModelConfig
from OLMo.olmo.model import OLMo
from datasets import load_from_disk

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_git_info():
    """获取当前代码仓库的Git信息"""
    git_info = {}
    try:
        git_info["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        git_info["short_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        git_info["is_dirty"] = len(git_status) > 0
    except Exception as e:
        git_info["error"] = str(e)
    return git_info

def main():
    parser = argparse.ArgumentParser(description="C4 Pretraining with HIPE")
    
    # 路径配置
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path containing c4_30M_train/val")
    parser.add_argument("--local_tokenizer_path", type=str, required=True)
    
    # 模型配置
    parser.add_argument("--model_size", type=str, default="300M", 
                        choices=["20M", "60M", "300M"])
    parser.add_argument("--seq_len", type=int, default=512,
                        help="训练序列长度（固定512）")
    
    # 训练配置
    parser.add_argument("--train_size", type=int, default=5000000)
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--max_tokens", type=int, default=1_000_000_000)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # 位置编码配置
    parser.add_argument("--pe_type", type=str, required=True,
                        choices=["rope", "hipe", "rope_yarn", "hipe_yarn"],
                        help="位置编码类型")
    parser.add_argument("--sigma", type=float, default=700.0,
                        help="HIPE sigma参数")
    parser.add_argument("--rope_scaling_threshold", type=int, default=7,
                        help="HIPE层级阈值（前N层用标准RoPE）")
    parser.add_argument("--decay_func", type=str, default="gaussian",
                        choices=["gaussian", "exp", "power", "segmented"])
    
    # 其他
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--wandb_dir", type=str, default=None)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    git_info = get_git_info()
    
    # 配置wandb
    run_tags = [args.model_size, f"pe_{args.pe_type}", f"len_{args.seq_len}", 
                f"seed_{args.seed}", f"commit_{git_info.get('short_commit', 'unknown')}"]
    if args.pe_type in ["hipe", "hipe_yarn"]:
        run_tags.append(f"sigma_{args.sigma}")
    
    wandb.init(
        project="PE-Pretrain-C4",
        group=f"pretrain_{args.model_size}",
        tags=run_tags,
        name=args.run_id,
        config=vars(args),
        dir=args.wandb_dir or args.output_dir,
        mode=args.wandb_mode
    )
    wandb.config.update(git_info)
    
    # 加载Tokenizer
    print(f"Loading Tokenizer from {args.local_tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer_path, local_files_only=True)
    except:
        from OLMo.olmo.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_pretrained(args.local_tokenizer_path, eos_token_id=50256, pad_token_id=50256)
    
    raw_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    vocab_size = ((raw_vocab_size + 63) // 64) * 64
    print(f">>> Vocab size: {raw_vocab_size} -> {vocab_size}")
    
    # 加载C4数据
    print(f"Loading C4 Data from: {args.dataset_path}")
    train_path = os.path.join(args.dataset_path, "c4_30M_train")
    val_path = os.path.join(args.dataset_path, "c4_30M_validation")
    
    train_full = load_from_disk(train_path)
    val_full = load_from_disk(val_path)
    
    train_ds = train_full.select(range(min(args.train_size, len(train_full))))
    val_ds = val_full.select(range(min(args.val_size, len(val_full))))
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)
    
    tokenized_train = train_ds.map(tokenize_function, batched=True, 
                                   remove_columns=train_ds.column_names, 
                                   num_proc=8, desc="Tokenizing Train")
    tokenized_val = val_ds.map(tokenize_function, batched=True,
                               remove_columns=val_ds.column_names,
                               num_proc=8, desc="Tokenizing Val")
    
    # Group texts
    block_size = args.seq_len + 1
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        return {
            k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
    
    lm_train = tokenized_train.map(group_texts, batched=True, num_proc=8)
    lm_val = tokenized_val.map(group_texts, batched=True, num_proc=8)
    
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        data = torch.tensor(input_ids, dtype=torch.long)
        return data[:, :-1].contiguous(), data[:, 1:].contiguous()
    
    train_loader = DataLoader(lm_train, batch_size=args.micro_batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(lm_val, batch_size=args.micro_batch_size,
                            collate_fn=collate_fn)
    
    # 模型配置
    if args.model_size == "300M":
        cur_d, cur_h, cur_l, cur_mlp = 1024, 16, 16, 8
    elif args.model_size == "60M":
        cur_d, cur_h, cur_l, cur_mlp = 512, 8, 8, 8
    else:
        cur_d, cur_h, cur_l, cur_mlp = 256, 8, 8, 8
    
    # 根据pe_type配置
    use_rope = args.pe_type in ["rope", "rope_yarn", "hipe", "hipe_yarn"]
    use_hipe = args.pe_type in ["hipe", "hipe_yarn"]
    use_yarn = args.pe_type in ["rope_yarn", "hipe_yarn"]
    
    cfg = ModelConfig(
        d_model=cur_d, n_heads=cur_h, n_layers=cur_l, mlp_ratio=cur_mlp,
        max_sequence_length=args.seq_len,
        vocab_size=vocab_size, embedding_size=vocab_size,
        init_std=0.02,
        rope=use_rope,
        yarn_enabled=use_yarn,
        yarn_max_position_embeddings=512,
        yarn_target_max_position_embeddings=args.seq_len if use_yarn else None,
        use_scaled_rope1=use_hipe,
        scaled_rope_sigma=args.sigma if use_hipe else 1.0,
        rope_scaling_threshold=args.rope_scaling_threshold if use_hipe else -1,
        decay_func=args.decay_func if use_hipe else 'gaussian',
        flash_attention=True,
    )
    
    model = OLMo(cfg).to(device)
    
    grad_accum_steps = args.global_batch_size // args.micro_batch_size
    total_steps = args.max_tokens // (args.global_batch_size * args.seq_len)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    print(f"Model: {args.model_size}, Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"PE Type: {args.pe_type}, YaRN: {use_yarn}, HIPE: {use_hipe}")
    print(f"Training steps: {total_steps}")
    
    # 训练循环
    model.train()
    step = 0
    total_loss = 0.0
    optimizer.zero_grad()
    train_iter = iter(train_loader)
    
    log_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    log_file.write(f"Git Commit: {git_info.get('commit_hash', 'unknown')}\n")
    log_file.write("Step,Loss,PPL,LR\n")
    
    while step < total_steps:
        for _ in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device), y.to(device)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=x)
                loss = F.cross_entropy(outputs.logits.view(-1, vocab_size), y.view(-1))
                loss = loss / grad_accum_steps
            
            loss.backward()
            total_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        step += 1
        
        # 日志
        if step % 10 == 0:
            avg_loss = total_loss / 10
            ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")
            wandb.log({"train/loss": avg_loss, "train/ppl": ppl, "train/lr": lr, "step": step})
            log_file.write(f"{step},{avg_loss:.4f},{ppl:.4f},{lr:.6e}\n")
            log_file.flush()
            total_loss = 0.0
        
        # 验证
        if step % args.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_tokens = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(input_ids=vx)
                        loss = F.cross_entropy(outputs.logits.view(-1, vocab_size), vy.view(-1), reduction='sum')
                    val_loss += loss.item()
                    val_tokens += vy.numel()
            
            avg_val_loss = val_loss / val_tokens
            val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
            print(f">>> Validation | Loss: {avg_val_loss:.4f} | PPL: {val_ppl:.2f}")
            wandb.log({"val/loss": avg_val_loss, "val/ppl": val_ppl, "step": step})
            log_file.write(f"VAL,{step},{avg_val_loss:.4f},{val_ppl:.4f}\n")
            log_file.flush()
            model.train()
        
        # 保存模型
        if step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"model_step_{step}.pt")
            torch.save(model.state_dict(), save_path)
            print(f">>> Saved checkpoint: {save_path}")
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f">>> Training complete. Final model: {final_path}")
    
    log_file.close()
    wandb.finish()

if __name__ == "__main__":
    main()
