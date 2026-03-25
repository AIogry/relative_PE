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
        git_info["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.STDOUT
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
        git_info["dirty_files"] = git_status if git_info["is_dirty"] else "None"
        
        git_info["remote_url"] = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        
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
        git_info["error"] = f"Git command failed: {e.output.decode('utf-8')}"
        git_info["commit_hash"] = "unknown"
        git_info["short_commit"] = "unknown"
    except Exception as e:
        git_info["error"] = f"Get git info failed: {str(e)}"
        git_info["commit_hash"] = "unknown"
        git_info["short_commit"] = "unknown"
    
    return git_info

# === 提取 DataLoader 生成逻辑，方便为不同验证长度创建对应的数据集 ===
def create_dataloader(tokenized_ds, seq_len, batch_size, shuffle=False, desc="Grouping"):
    block_size = seq_len + 1
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        return {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    
    grouped_ds = tokenized_ds.map(group_texts, batched=True, num_proc=8, desc=desc)
    
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        data = torch.tensor(input_ids, dtype=torch.long)
        x = data[:, :-1].contiguous() 
        y = data[:, 1:].contiguous()  
        return x, y
        
    return DataLoader(grouped_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# === 独立的验证运行函数 ===
def run_evaluation(model, dataloader, vocab_size, device):
    model.eval()
    total_val_loss = 0.0
    total_val_tokens = 0
    
    with torch.no_grad():
        for vx, vy in dataloader:
            vx, vy = vx.to(device), vy.to(device)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=vx)
                loss = nn.functional.cross_entropy(
                    outputs.logits.view(-1, vocab_size), 
                    vy.view(-1), 
                    reduction='sum'
                )
            total_val_loss += loss.item()
            total_val_tokens += vy.numel() 
            
    avg_val_loss = total_val_loss / total_val_tokens if total_val_tokens > 0 else 0.0
    val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else 1e9 
    return avg_val_loss, val_ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True, help="C4 datasets")
    parser.add_argument("--arxiv_val_path", type=str, default="/data/qijunrong/03-proj/PE/arxiv_data/arxiv_validation", help="Arxiv validation datasets")

    parser.add_argument("--train_size", type=int, default=5000000)
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--local_tokenizer_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="20M", choices=["20M", "60M", "300M"])

    # Baselines & HIPE 参数
    parser.add_argument("--alibi", action="store_true")
    parser.add_argument("--fope", action="store_true")
    parser.add_argument("--yarn", action="store_true")
    parser.add_argument("--nope", action="store_true")
    parser.add_argument("--xpos", action="store_true")
    parser.add_argument("--rope_scale", type=float, default=None)
    parser.add_argument("--use_scaled_rope", action="store_true")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--rope_scaling_threshold", type=int, default=-1)
    parser.add_argument("--sigma_list", nargs='+', default=None)
    
    # 训练超参
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=8) 
    parser.add_argument("--max_tokens", type=int, default=100_000_000)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # 验证间隔控制
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000, help="每多少步保存一次模型权重")
    parser.add_argument("--extrap_eval_interval", type=int, default=2000, help="多少步进行一次长文本外推测试")
    
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--wandb_dir", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    git_info = get_git_info()
    run_tags = [args.model_size, f"len_{args.seq_len}", f"seed_{args.seed}"] 
    
    if args.alibi or args.xpos or args.fope or args.nope or not args.use_scaled_rope:
        run_group = "Exp2-C4-Baselines"
        run_tags.append("baseline")
    else:
        run_group = "Exp2-C4-HIPE"
        run_tags.append("hipe")
        run_tags.append(f"sigma_{args.sigma}")

    if args.wandb_dir is not None: os.makedirs(args.wandb_dir, exist_ok=True)

    wandb.init(project="Position Embedding", group=run_group, tags=run_tags, name=args.run_id,
               config=vars(args), dir=args.wandb_dir if args.wandb_dir else args.output_dir, mode=args.wandb_mode)

    wandb.config.update(git_info)

    print(f"Loading Tokenizer from {args.local_tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer_path, local_files_only=True)
    except Exception:
        from OLMo.olmo.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_pretrained(args.local_tokenizer_path, eos_token_id=50256, pad_token_id=50256)
    
    raw_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    vocab_size = ((raw_vocab_size + 63) // 64) * 64
    wandb.config.update({"actual_vocab_size": vocab_size})


    def tokenize_function(examples): return tokenizer(examples["text"], truncation=False)

    # =========================================================================
    # 1. 极速数据导入：按需加载 (Lazy Loading)
    # =========================================================================
    print(f"Loading C4 Data from: {args.dataset_path}")
    train_path = os.path.join(args.dataset_path, "c4_30M_train")
    c4_val_path = os.path.join(args.dataset_path, "c4_30M_validation")
    
    # 加载磁盘索引（这步是瞬间完成的，不会占用内存）
    train_full = load_from_disk(train_path)
    c4_val_full = load_from_disk(c4_val_path)
    arxiv_val_full = load_from_disk(args.arxiv_val_path)

    # 【关键修改】在进行任何 Map 操作前，先 Select 子集
    real_train_size = min(args.train_size, len(train_full))
    real_val_size = min(args.val_size, len(c4_val_full))
    
    train_ds = train_full.select(range(real_train_size))
    c4_val_ds = c4_val_full.select(range(real_val_size))
    # Arxiv 验证集通常较小（我们之前存了 2000 条），建议全量使用
    arxiv_val_ds = arxiv_val_full

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    # 定义一个内部函数来处理 Tokenize + Grouping，减少代码冗余
    def prepare_loader(dataset, target_seq_len, mbs, shuffle=False, desc=""):
        print(f">>> Processing {desc} (Target Len: {target_seq_len})...")
        # 1. Tokenize
        tokenized = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=dataset.column_names, 
            num_proc=8, 
            desc=f"Tokenizing {desc}"
        )
        
        # 2. 如果是 Arxiv 外推验证，需要过滤掉太短的文本
        if "Arxiv" in desc:
            old_len = len(tokenized)
            tokenized = tokenized.filter(lambda x: len(x['input_ids']) >= target_seq_len)
            print(f"    Filter: {old_len} -> {len(tokenized)} examples (len >= {target_seq_len})")

        # 3. Chunking / Grouping
        block_size = target_seq_len + 1
        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            return {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
        
        lm_ds = tokenized.map(group_texts, batched=True, num_proc=8, desc=f"Grouping {desc}")
        
        # 4. DataLoader
        def collate_fn(batch):
            input_ids = [item['input_ids'] for item in batch]
            data = torch.tensor(input_ids, dtype=torch.long)
            return data[:, :-1].contiguous(), data[:, 1:].contiguous()
        
        return DataLoader(lm_ds, batch_size=mbs, shuffle=shuffle, collate_fn=collate_fn)

    # =========================================================================
    # 2. 创建多源、多长度的 DataLoader
    # =========================================================================
    print(f">>> Creating Dataloaders for {args.run_id}...")
    
    # 训练集: 512
    train_loader = prepare_loader(train_ds, args.seq_len, args.micro_batch_size, shuffle=True, desc="Train C4")
    
    # 基准验证: 512
    val_loader_512 = prepare_loader(c4_val_ds, args.seq_len, args.micro_batch_size, desc="Val C4 512")
    
    # 外推验证: 使用 Arxiv 数据集测试 1024 和 2048
    extrap_lengths = [1024, 2048]
    extrap_loaders = {}
    for elen in extrap_lengths:
        # 动态调整 MBS：长度翻倍，MBS 减半，防止验证时 OOM
        eval_mbs = max(1, args.micro_batch_size // (elen // args.seq_len))
        extrap_loaders[elen] = prepare_loader(arxiv_val_ds, elen, eval_mbs, desc=f"Val Arxiv {elen}")

    # =========================================================================
    # 3. 300M 模型架构锁定 (24层版)
    # =========================================================================
    if args.model_size == "300M":
        cur_d, cur_h, cur_l, cur_mlp = 1024, 16, 24, 8
        print(">>> Using CUSTOM 300M Configuration (24 Layers, MLP-Ratio 8)")
    else:
        cur_d, cur_h, cur_l, cur_mlp = 256, 8, 8, 8

    # ================= 初始化模型 =================
    grad_accum_steps = args.global_batch_size // args.micro_batch_size
    total_steps = args.max_train_steps if args.max_train_steps else args.max_tokens // (args.global_batch_size * args.seq_len)

    cfg = ModelConfig(
        d_model=cur_d, n_heads=cur_h, n_layers=cur_l, mlp_ratio=cur_mlp,
        max_sequence_length=args.seq_len, vocab_size=vocab_size, embedding_size=vocab_size, 
        init_std=0.02, rope=True, flash_attention=True,
        yarn_enabled=args.yarn, # 训练时是否开启 yarn (当前外推实验中设为 False)
        yarn_target_max_position_embeddings=args.seq_len if args.yarn else None,
        yarn_max_position_embeddings=512, 
        use_scaled_rope1=args.use_scaled_rope, # 控制 HIPE 的开关
        scaled_rope_sigma=args.sigma
    )
    
    model = OLMo(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ================= 核心训练与验证循环 =================
    model.train()
    step = 0
    total_loss = 0.0 
    train_iter = iter(train_loader)
    
    log_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    log_file.write("Step,Loss,PPL\n")

    while step < total_steps:
        current_step_loss = 0.0
        for _ in range(grad_accum_steps):
            try: x, y = next(train_iter)
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

        if step % args.save_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_path = os.path.join(ckpt_dir, f"model_step_{step}.pt")
            
            # 推荐只保存 state_dict 以节省空间
            torch.save(model.state_dict(), save_path)
            print(f">>> Checkpoint saved at step {step}: {save_path}")
        
        if step % 10 == 0:
            avg_loss = total_loss / 10
            ppl = math.exp(avg_loss) if avg_loss < 20 else 1e9
            wandb.log({"train/loss": avg_loss, "train/ppl": ppl, "train/lr": scheduler.get_last_lr()[0], "step": step})
            total_loss = 0.0

        # === 1. 基础验证 (512 长度) ===
        if step % args.eval_interval == 0:
            print(f">>> Running Validation 512 at step {step}...")
            avg_val_loss, val_ppl = run_evaluation(model, val_loader_512, vocab_size, device)
            print(f">>> VAL 512 PPL: {val_ppl:.4f}")
            wandb.log({"val/loss_512": avg_val_loss, "val/ppl_512": val_ppl, "step": step})
            log_file.write(f"VAL512,{step},{avg_val_loss},{val_ppl}\n")
            log_file.flush()
            model.train()

        # === 2. 动态外推验证 (1024 / 2048 长度) ===
        if step % args.extrap_eval_interval == 0:
            print(f">>> Running Arxiv Zero-shot Extrapolation at step {step}...")
            
            orig_yarn_enabled = model.config.yarn_enabled
            orig_max_len = model.config.max_sequence_length
            orig_target_len = getattr(model.config, "yarn_target_max_position_embeddings", 512)
            orig_base_len = getattr(model.config, "yarn_max_position_embeddings", 512)
            
            for elen in extrap_lengths:
                eloader = extrap_loaders[elen]
                
                # [新增修改 5]: 严谨的配置更新与缓存清除
                model.config.yarn_enabled = True
                model.config.max_sequence_length = elen
                model.config.yarn_max_position_embeddings = 512
                model.config.yarn_target_max_position_embeddings = elen
                
                # 强制重新计算 YaRN 频率
                model.transformer.rope.inv_freq = model.transformer.rope.get_inv_freq(device)
                model.transformer.rope._cache.clear() # 极其重要：清空 sin/cos 缓存
                
                # 强制重新计算 HIPE 幅值衰减 (调用刚才在 model.py 增加的函数)
                if hasattr(model.transformer.rope, "_update_scale_factor"):
                    model.transformer.rope._update_scale_factor(device)
                
                # 运行验证
                e_loss, e_ppl = run_evaluation(model, eloader, vocab_size, device)
                print(f">>> EXTRAP {elen} Arxiv PPL: {e_ppl:.4f}")
                wandb.log({f"extrap_arxiv/loss_{elen}": e_loss, f"extrap_arxiv/ppl_{elen}": e_ppl, "step": step})
                log_file.write(f"EXTRAP_ARXIV_{elen},{step},{e_loss},{e_ppl}\n")
            
            # 恢复原配置继续训练
            model.config.yarn_enabled = orig_yarn_enabled
            model.config.max_sequence_length = orig_max_len
            model.config.yarn_max_position_embeddings = orig_base_len
            model.config.yarn_target_max_position_embeddings = orig_target_len
            
            model.transformer.rope.inv_freq = model.transformer.rope.get_inv_freq(device)
            model.transformer.rope._cache.clear()
            if hasattr(model.transformer.rope, "_update_scale_factor"):
                model.transformer.rope._update_scale_factor(device)
            
            log_file.flush()
            model.train()

    print("Saving model checkpoint...")
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print("Training Finished.")
    log_file.close()
    wandb.finish()

if __name__ == "__main__":
    main()