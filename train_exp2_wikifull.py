import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import math
import json
import random
import numpy as np
import wandb
import subprocess
import sys
from torch.amp import autocast
from transformers import AutoTokenizer

# Imports from OLMo
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
    """获取当前代码仓库的Git信息，返回字典"""
    git_info = {}
    try:
        # 1. 获取完整commit hash（唯一标识代码版本）
        git_info["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        
        # 2. 获取短commit hash（更易读）
        git_info["short_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        
        # 3. 获取当前分支名
        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        
        # 4. 检查是否有未提交的修改（dirty/clean）
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        git_info["is_dirty"] = len(git_status) > 0
        git_info["dirty_files"] = git_status if git_info["is_dirty"] else "None"
        
        # 5. 获取远程仓库地址（用于跳转到GitHub）
        git_info["remote_url"] = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        
        # 6. 转换为GitHub网页链接（适配HTTPS/SSH格式）
        if git_info["remote_url"].startswith("git@"):
            git_info["github_commit_url"] = git_info["remote_url"].replace(
                "git@github.com:", "https://github.com/"
            ).replace(".git", "") + f"/commit/{git_info['commit_hash']}"
        elif git_info["remote_url"].startswith("https"):
            git_info["github_commit_url"] = git_info["remote_url"].replace(
                ".git", ""
            ) + f"/commit/{git_info['commit_hash']}"
        else:
            git_info["github_commit_url"] = "Unknown"
            
    except subprocess.CalledProcessError as e:
        # Git命令执行失败（如非Git仓库、无权限）
        git_info["error"] = f"Git command failed: {e.output.decode('utf-8')}"
        git_info["commit_hash"] = "unknown"
        git_info["short_commit"] = "unknown"
    except Exception as e:
        # 其他异常（如编码错误）
        git_info["error"] = f"Get git info failed: {str(e)}"
        git_info["commit_hash"] = "unknown"
        git_info["short_commit"] = "unknown"
    
    return git_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--local_data_path", type=str, required=True)
    parser.add_argument("--local_tokenizer_path", type=str, required=True)
    
    # === 模型大小选择 ===
    parser.add_argument("--model_size", type=str, default="20M", choices=["20M", "60M", "300M"])

    # === Strong Baselines 参数 ===
    parser.add_argument("--alibi", action="store_true")
    parser.add_argument("--fope", action="store_true")
    parser.add_argument("--yarn", action="store_true")
    parser.add_argument("--nope", action="store_true") # NoPE
    parser.add_argument("--xpos", action="store_true") # XPos
    parser.add_argument("--rope_scale", type=float, default=None)
    
    # === HIPE 参数 ===
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
    
    # === wandb参数设置 ===
    parser.add_argument("--wandb_mode", type=str, default="offline", help="Wandb mode")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Wandb offline tracking directory")

    args = parser.parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    git_info = get_git_info()
    # 将Git短哈希加入tags，方便在wandb中快速筛选

    run_tags = [args.model_size, f"len_{args.seq_len}", f"seed_{args.seed}"]  # 先初始化
    run_tags.append(f"commit_{git_info['short_commit']}")  # 后追加
    run_tags.append(f"dirty_{git_info['is_dirty']}" if "is_dirty" in git_info else "dirty_unknown")

    if args.alibi or args.xpos or args.fope or args.nope or not args.use_scaled_rope:
        run_group = "Exp2-wiki-Baselines"
        run_tags.append("baseline")
    else:
        run_group = "Exp2-wiki-HIPE"
        run_tags.append("hipe")
        run_tags.append(f"sigma_{args.sigma}")

    if args.wandb_dir is not None:
        os.makedirs(args.wandb_dir, exist_ok=True)

    wandb.init(
        project="Position Embedding",   # 替换为你的项目名称
        group=run_group,
        tags=run_tags,
        name=args.run_id,             # 使用 bash 脚本中生成的精确 run_id 作为实验名
        config=vars(args),            # 将所有 argparse 的参数自动保存到 wandb config 中
        dir=args.wandb_dir if args.wandb_dir else args.output_dir,
        mode=args.wandb_mode
    )

    wandb.config.update(git_info)

    # 处理 Sigma List
    final_sigmas = None
    if args.sigma_list is not None:
        final_sigmas = []
        for s in args.sigma_list:
            if s == "None": final_sigmas.append(None)
            else: final_sigmas.append(float(s))

    print(f"Loading Tokenizer from {args.local_tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer_path)
    raw_vocab_size = tokenizer.vocab_size
    vocab_size = ((raw_vocab_size + 63) // 64) * 64
    print(f">>> Resizing Vocab: {raw_vocab_size} -> {vocab_size} (Aligned to 64)")

    wandb.config.update({"actual_vocab_size": vocab_size})
    # =========================================================================
    # [优化] 数据预处理：一次性分词，不再在训练循环中分词
    # =========================================================================
    print(f"Loading Dataset from {args.local_data_path}...")
    dataset_dict = load_from_disk(args.local_data_path)
    
    # 1. 预分词
    print(">>> Pre-tokenizing dataset (this improves speed significantly)...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    tokenized_datasets = dataset_dict.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"], 
        num_proc=4, # 使用多核加速
        desc="Tokenizing"
    )

    # 2. 拼接与切块 (Chunking)
    # 我们需要 input (L) 和 target (L)，所以切块长度为 seq_len + 1
    block_size = args.seq_len + 1

    def group_texts(examples):
        # 拼接所有文本
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # 丢弃最后不够一个 block 的部分
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # 切分
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        desc="Grouping"
    )

    # 3. 构建 DataLoader 的 Collate 函数
    def collate_fn(batch):
        # batch 是一个 list of dict, 每个 dict 包含 'input_ids' 等
        input_ids = [item['input_ids'] for item in batch]
        data = torch.tensor(input_ids, dtype=torch.long)
        x = data[:, :-1].contiguous() # 输入
        y = data[:, 1:].contiguous()  # 标签
        return x, y

    train_ds = lm_datasets['train']
    val_ds = lm_datasets['validation']
    
    print(f">>> Processed Dataset: Train={len(train_ds)} chunks, Val={len(val_ds)} chunks")

    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size, collate_fn=collate_fn)
    # =========================================================================

    grad_accum_steps = args.global_batch_size // args.micro_batch_size

    if args.max_train_steps is not None and args.max_train_steps > 0:
        total_steps = args.max_train_steps
        print(f">>> [DEBUG MODE] Training for fixed steps: {total_steps}")
    else:
        total_steps = args.max_tokens // (args.global_batch_size * args.seq_len)
        print(f">>> [FULL MODE] Training for max tokens: {args.max_tokens} (~{total_steps} steps)")

    # === 模型大小选择 ===
    init_std_val = 0.02
    if args.model_size == "20M":
        cur_d, cur_h, cur_l, cur_mlp = 256, 8, 8, 8
    elif args.model_size == "60M":
        cur_d, cur_h, cur_l, cur_mlp = 512, 8, 8, 8
    elif args.model_size == "300M":
        # 你的新配置：1024维，16头，16层，MLP ratio为8
        cur_d, cur_h, cur_l, cur_mlp = 1024, 16, 16, 8
        # init_std 保持默认的 0.02，无需特殊覆盖
        print(">>> Using OLMo-300M Configuration (16 Layers, MLP Ratio 8)")
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    # =========================================================================
    # [Baseline Configuration Logic]
    # =========================================================================
    use_alibi = False
    use_fope = False
    use_yarn = False
    use_nope = False 
    use_xpos = False 
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
    elif args.nope:
        use_nope = True
        use_rope = False
        use_fope = False
        print(">>> [Config] NoPE ENABLED | RoPE DISABLED")
    elif args.xpos:
        use_xpos = True
        use_rope = False 
        print(">>> [Config] XPos ENABLED | RoPE DISABLED")

    cfg = ModelConfig(
        d_model=cur_d, 
        n_heads=cur_h, 
        n_layers=cur_l, 
        mlp_ratio=cur_mlp,
        max_sequence_length=args.seq_len,
        vocab_size=vocab_size,
        embedding_size=vocab_size, 
        init_std=init_std_val,
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

    if use_nope: cfg.nope = True
    if use_xpos: cfg.xpos = True
    
    model = OLMo(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"Experiment: {args.run_id} -> Saving to {args.output_dir}")
    print(f">>> Git Commit: {git_info['commit_hash']} (short: {git_info['short_commit']})")
    print(f">>> Git Branch: {git_info.get('branch', 'unknown')}")
    print(f">>> Code Dirty: {git_info.get('is_dirty', 'unknown')}")



    model.train()
    step = 0
    total_loss = 0.0 
    
    optimizer.zero_grad()
    train_iter = iter(train_loader)
    
    log_path = os.path.join(args.output_dir, "log.txt")
    log_file = open(log_path, "w")
    log_file.write(f"Git Commit: {git_info['commit_hash']}\n")
    log_file.write(f"Git Short Commit: {git_info['short_commit']}\n")
    log_file.write(f"Git Branch: {git_info.get('branch', 'unknown')}\n")
    log_file.write(f"Code Dirty: {git_info.get('is_dirty', 'unknown')}\n")
    log_file.write(f"GitHub Commit URL: {git_info.get('github_commit_url', 'unknown')}\n")
    log_file.write("Step,Loss,PPL\n")
    log_file.flush()

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
            print(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.4f} | LR: {lr:.2e}")

            wandb.log({
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "train/lr": lr,
                "step": step
            })

            log_file.write(f"Step {step},Loss: {avg_loss:.4f},PPL: {ppl:.4f}\n")
            log_file.flush()
            total_loss = 0.0

        if step % args.eval_interval == 0:
            print(">>> Running Validation...")
            model.eval()
            total_val_loss = 0.0
            total_val_tokens = 0
            
            with torch.no_grad():
                # 直接遍历整个验证集，不再受 eval_steps 和 mbs 限制
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(input_ids=vx)
                        # 注意：这里改为 reduction='sum'，计算当前批次所有 token 的 loss 总和
                        loss = nn.functional.cross_entropy(
                            outputs.logits.view(-1, vocab_size), 
                            vy.view(-1), 
                            reduction='sum'
                        )
                    
                    total_val_loss += loss.item()
                    total_val_tokens += vy.numel() # 累加当前批次的有效 token 数量
            
            if total_val_tokens > 0:
                # 严谨的全局平均 Loss
                avg_val_loss = total_val_loss / total_val_tokens
                # 增加上限保护，防止训练早期极度发散时 math.exp 报错 (OverflowError)
                val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else 1e9 
                
                print(f">>> VAL PPL: {val_ppl:.4f}")

                wandb.log({
                    "val/loss": avg_val_loss,
                    "val/ppl": val_ppl,
                    "step": step
                })

                log_file.write(f"VAL,{step},{avg_val_loss},{val_ppl}\n")
                log_file.flush() # 加上 flush，确保验证日志能立刻写入文件，哪怕程序中断也不丢失
                
            model.train()

    print("Saving model checkpoint...")
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    # [新增] 告诉 wandb 保存这个模型文件，它会自动上传到云端
    wandb.save(model_path, base_path=args.output_dir)
    print("Training Finished.")
    log_file.close()
    
    # [新增] 结束 wandb 记录
    wandb.finish()

if __name__ == "__main__":
    main()