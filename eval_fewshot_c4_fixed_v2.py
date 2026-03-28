"""
Few-Shot外推评估脚本 V2 - 修复Padding Leakage
关键修复：
1. 使用group_texts连续拼接，避免Padding
2. 在每个目标长度下重新加载模型并微调
3. Loss计算使用ignore_index=50256
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
import json
from torch.amp import autocast
from transformers import AutoTokenizer
from tqdm import tqdm

from OLMo.olmo.config import ModelConfig
from OLMo.olmo.model import OLMo
from datasets import load_from_disk

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def evaluate_perplexity(model, dataloader, vocab_size, device, max_batches=None, pad_token_id=50256):
    """评估困惑度 - 使用ignore_index忽略padding"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            x, y = x.to(device), y.to(device)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=x)
                # 【关键修复】忽略padding token的loss
                loss = F.cross_entropy(
                    outputs.logits.view(-1, vocab_size),
                    y.view(-1),
                    reduction='sum',
                    ignore_index=pad_token_id  # 忽略padding
                )
            # 计算非padding token数量
            valid_tokens = (y != pad_token_id).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens
            num_batches += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, ppl, total_tokens

def few_shot_adaptation(model, train_loader, optimizer, steps, vocab_size, device, pad_token_id=50256):
    """在few-shot数据上进行快速适应 - 忽略padding"""
    model.train()
    train_iter = iter(train_loader)
    
    for step in range(steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=x)
            # 【关键修复】忽略padding token的loss
            loss = F.cross_entropy(
                outputs.logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=pad_token_id  # 忽略padding
            )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Few-shot step {step+1}/{steps}, Loss: {loss.item():.4f}")

def configure_yarn_for_length(model, base_len, target_len, device):
    """配置YaRN用于指定长度的外推"""
    print(f"Configuring YaRN: base={base_len}, target={target_len}")
    
    model.config.yarn_enabled = True
    model.config.yarn_max_position_embeddings = base_len
    model.config.yarn_target_max_position_embeddings = target_len
    model.config.max_sequence_length = target_len
    
    if not hasattr(model.config, 'yarn_beta_slow'):
        model.config.yarn_beta_slow = 1.0
    if not hasattr(model.config, 'yarn_beta_fast'):
        model.config.yarn_beta_fast = 32.0
    
    scale = target_len / base_len
    print(f"  YaRN scale factor: {scale:.2f}x")
    
    if hasattr(model.transformer, 'blocks'):
        blocks = model.transformer.blocks
    elif hasattr(model.transformer, 'block_groups'):
        blocks = [block for group in model.transformer.block_groups for block in group]
    else:
        blocks = []
    
    mscale_values = []
    for i, block in enumerate(blocks):
        if hasattr(block, 'rotary_emb') and block.rotary_emb is not None:
            block.rotary_emb.inv_freq = block.rotary_emb.get_inv_freq(device)
            if hasattr(block.rotary_emb, '_mscale'):
                mscale_values.append(block.rotary_emb._mscale)
            if hasattr(block.rotary_emb, '_cache'):
                block.rotary_emb._cache.clear()
    
    if mscale_values:
        print(f"  YaRN mscale range: [{min(mscale_values):.4f}, {max(mscale_values):.4f}]")
    
    return model

def group_texts(examples, block_size):
    """【关键】连续拼接文本，避免padding - 与预训练一致"""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # 确保长度是block_size的整数倍
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

def load_and_prepare_data(dataset_path, tokenizer, seq_len, few_shot_k, seed=42):
    """加载并准备数据 - 使用group_texts避免padding，只加载部分数据"""
    train_path = os.path.join(dataset_path, "c4_30M_train")
    val_path = os.path.join(dataset_path, "c4_30M_validation")
    
    print("Loading C4 dataset (subset)...")
    # 【优化】只加载前10万条，避免内存和速度问题
    train_full = load_from_disk(train_path)
    val_full = load_from_disk(val_path)
    
    # 【关键】只使用部分数据
    TRAIN_SUBSET = 100000  # 10万条足够
    VAL_SUBSET = 10000     # 1万条测试
    
    train_full = train_full.select(range(min(TRAIN_SUBSET, len(train_full))))
    val_full = val_full.select(range(min(VAL_SUBSET, len(val_full))))
    
    print(f"  Using subset: Train={len(train_full)}, Val={len(val_full)}")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)
    
    print("Tokenizing train data...")
    tokenized_train = train_full.map(tokenize_function, batched=True, 
                                     remove_columns=train_full.column_names,
                                     num_proc=8, desc="Tokenizing train")
    
    print("Tokenizing val data...")
    tokenized_val = val_full.map(tokenize_function, batched=True,
                                 remove_columns=val_full.column_names,
                                 num_proc=8, desc="Tokenizing val")
    
    # 【关键】使用group_texts连续拼接，生成无padding的固定长度序列
    block_size = seq_len + 1
    
    print(f"Grouping texts into {block_size}-token blocks (no padding)...")
    grouped_train = tokenized_train.map(
        lambda x: group_texts(x, block_size),
        batched=True,
        num_proc=8,
        desc="Grouping train"
    )
    grouped_val = tokenized_val.map(
        lambda x: group_texts(x, block_size),
        batched=True,
        num_proc=8,
        desc="Grouping val"
    )
    
    # 【关键】使用validation数据进行few-shot和测试（避免与预训练数据混合）
    # 预训练使用了C4 train的前5M条，我们用val数据做few-shot和测试
    random.seed(seed)
    
    # 从validation采样few-shot（训练完全未见过）
    few_shot_indices = random.sample(range(len(grouped_val)), min(few_shot_k, len(grouped_val)))
    few_shot_ds = grouped_val.select(few_shot_indices)
    
    # 剩余validation作为测试
    test_indices = [i for i in range(len(grouped_val)) if i not in few_shot_indices]
    test_size = min(2000, len(test_indices))
    test_ds = grouped_val.select(test_indices[:test_size])
    
    print(f"  Few-shot from val: {len(few_shot_ds)} chunks")
    print(f"  Test from val: {len(test_ds)} chunks")
    
    return few_shot_ds, test_ds

def collate_fn_no_pad(batch):
    """【关键】无需padding的collate函数 - 数据已经是固定长度"""
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    x = input_ids[:, :-1].contiguous()
    y = input_ids[:, 1:].contiguous()
    return x, y

def get_batch_size_for_length(length, base_batch_size=8):
    """根据序列长度动态调整batch size以避免OOM"""
    # 4096需要更小的batch size
    if length >= 4096:
        return 2
    elif length >= 2048:
        return 4
    else:
        return base_batch_size

def main():
    parser = argparse.ArgumentParser(description="Few-Shot Extrapolation Evaluation (C4, Fixed)")
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="300M",
                        choices=["20M", "60M", "300M"])
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--local_tokenizer_path", type=str, required=True)
    
    # Few-shot配置
    parser.add_argument("--few_shot_k", type=int, default=2000)
    parser.add_argument("--few_shot_steps", type=int, default=100)
    parser.add_argument("--few_shot_lr", type=float, default=5e-6)
    
    # 外推测试长度
    parser.add_argument("--base_len", type=int, default=512)
    parser.add_argument("--test_lengths", type=int, nargs='+', 
                        default=[1024, 2048, 4096])
    
    # 模型配置
    parser.add_argument("--pe_type", type=str, required=True,
                        choices=["rope", "hipe", "rope_yarn", "hipe_yarn"])
    parser.add_argument("--sigma", type=float, default=700.0)
    parser.add_argument("--rope_scaling_threshold", type=int, default=7)
    parser.add_argument("--decay_func", type=str, default="gaussian")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="c4_extrap_results_fixed.json")
    parser.add_argument("--eval_batches", type=int, default=None)
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pad_token_id = 50256  # GPT2 tokenizer的pad token
    
    print("="*60)
    print(f"Few-Shot Extrapolation Evaluation (C4, Fixed v2)")
    print(f"【修复】使用group_texts避免padding")
    print(f"【修复】在每个目标长度下重新加载并微调")
    print(f"【修复】Loss计算忽略padding token")
    print("="*60)
    
    # 加载Tokenizer
    print(f"Loading Tokenizer from {args.local_tokenizer_path}...")
    from transformers import PreTrainedTokenizerFast
    tokenizer_path = os.path.join(args.local_tokenizer_path, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<|padding|>"
    tokenizer.eos_token = "<|endoftext|>"
    print(f"  Loaded PreTrainedTokenizerFast, vocab_size={len(tokenizer)}")
    
    raw_vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else 50257
    vocab_size = ((raw_vocab_size + 63) // 64) * 64
    
    # 构建模型配置
    if args.model_size == "300M":
        cur_d, cur_h, cur_l, cur_mlp = 1024, 16, 16, 8
    elif args.model_size == "60M":
        cur_d, cur_h, cur_l, cur_mlp = 512, 8, 8, 8
    else:
        cur_d, cur_h, cur_l, cur_mlp = 256, 8, 8, 8
    
    use_rope = args.pe_type in ["rope", "rope_yarn", "hipe", "hipe_yarn"]
    use_hipe = args.pe_type in ["hipe", "hipe_yarn"]
    
    # 准备结果存储
    results = {
        "config": vars(args),
        "extrap_eval": {},
    }
    
    # ==================== 对每个长度进行独立评估 ====================
    all_lengths = [args.base_len] + args.test_lengths
    
    for eval_len in all_lengths:
        print(f"\n{'='*60}")
        print(f"Evaluating at length: {eval_len}")
        print(f"{'='*60}")
        
        # 【关键】为每个长度重新加载数据（不同block_size）
        print(f"Loading data for length {eval_len}...")
        few_shot_ds, test_ds = load_and_prepare_data(
            args.dataset_path, tokenizer, eval_len, 
            args.few_shot_k, args.seed
        )
        print(f"  Few-shot: {len(few_shot_ds)} chunks")
        print(f"  Test: {len(test_ds)} chunks")
        
        # 【关键】为每个长度重新加载模型
        print(f"Loading fresh model...")
        cfg = ModelConfig(
            d_model=cur_d, n_heads=cur_h, n_layers=cur_l, mlp_ratio=cur_mlp,
            max_sequence_length=eval_len,
            vocab_size=vocab_size, embedding_size=vocab_size,
            init_std=0.02,
            rope=use_rope,
            yarn_enabled=False,
            use_scaled_rope1=use_hipe,
            scaled_rope_sigma=args.sigma if use_hipe else 1.0,
            rope_scaling_threshold=args.rope_scaling_threshold if use_hipe else -1,
            decay_func=args.decay_func if use_hipe else 'gaussian',
            flash_attention=True,
        )
        cfg.yarn_beta_slow = 1.0
        cfg.yarn_beta_fast = 32.0
        cfg.yarn_max_position_embeddings = args.base_len
        
        model = OLMo(cfg).to(device)
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model loaded. Parameters: {count_parameters(model)/1e6:.1f}M")
        
        # 根据长度动态调整batch size
        batch_size = get_batch_size_for_length(eval_len)
        print(f"  Using batch_size={batch_size} for length={eval_len}")
        
        # 如果是base长度，评估但不训练（参考点）
        if eval_len == args.base_len:
            print(f"\nBase length {eval_len}: Evaluating without adaptation...")
            test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn_no_pad)
            loss, ppl, tokens = evaluate_perplexity(
                model, test_loader, vocab_size, device, 
                args.eval_batches, pad_token_id
            )
            results[f"base_{eval_len}"] = {"loss": loss, "ppl": ppl, "tokens": tokens}
            print(f"Base {eval_len} | Loss: {loss:.4f} | PPL: {ppl:.2f}")
            continue
        
        # 外推长度：配置YaRN（如果需要）
        if args.pe_type in ["rope_yarn", "hipe_yarn"]:
            model = configure_yarn_for_length(model, args.base_len, eval_len, device)
        else:
            model.config.max_sequence_length = eval_len
            print(f"Direct extrapolation (no YaRN): length={eval_len}")
        
        # 【关键】在目标长度下进行few-shot微调
        if args.few_shot_steps > 0:
            print(f"\nAdapting on length {eval_len} ({args.few_shot_steps} steps)...")
            few_shot_loader = DataLoader(few_shot_ds, batch_size=batch_size, 
                                         shuffle=True, collate_fn=collate_fn_no_pad)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.few_shot_lr)
            few_shot_adaptation(model, few_shot_loader, optimizer,
                               args.few_shot_steps, vocab_size, device, pad_token_id)
        
        # 评估
        print(f"\nEvaluating on length {eval_len}...")
        test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn_no_pad)
        loss, ppl, tokens = evaluate_perplexity(
            model, test_loader, vocab_size, device,
            args.eval_batches, pad_token_id
        )
        
        results["extrap_eval"][eval_len] = {
            "length": eval_len,
            "loss": loss,
            "ppl": ppl,
            "tokens_evaluated": tokens
        }
        
        # 计算ratio
        base_ppl = results.get(f"base_{args.base_len}", {}).get("ppl", ppl)
        ratio = ppl / base_ppl if base_ppl else float('inf')
        print(f"Length {eval_len} | Loss: {loss:.4f} | PPL: {ppl:.2f} | Ratio: {ratio:.2f}x")
        
        # 释放显存
        del model
        torch.cuda.empty_cache()
    
    # ==================== 保存结果 ====================
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.output_file}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Summary ({args.pe_type}):")
    print(f"{'='*60}")
    base_ppl = results.get(f"base_{args.base_len}", {}).get("ppl", 0)
    print(f"Base ({args.base_len}): PPL = {base_ppl:.2f}")
    for length in sorted(results["extrap_eval"].keys()):
        res = results["extrap_eval"][length]
        ratio = res['ppl'] / base_ppl if base_ppl else float('inf')
        print(f"Extrap ({length}): PPL = {res['ppl']:.2f} (ratio: {ratio:.2f}x)")
    print("="*60)

if __name__ == "__main__":
    main()
