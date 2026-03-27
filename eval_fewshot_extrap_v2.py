"""
Few-Shot外推评估脚本 V2
支持从arxiv_train采样不同数量的few-shot样本
支持多shot大小对比实验
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
from datasets import load_from_disk, concatenate_datasets

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def evaluate_perplexity(model, dataloader, vocab_size, device, max_batches=None):
    """评估困惑度"""
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
                loss = F.cross_entropy(
                    outputs.logits.view(-1, vocab_size),
                    y.view(-1),
                    reduction='sum'
                )
            total_loss += loss.item()
            total_tokens += y.numel()
            num_batches += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, ppl, total_tokens

def few_shot_adaptation(model, train_loader, optimizer, steps, vocab_size, device):
    """在few-shot数据上进行快速适应"""
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
            loss = F.cross_entropy(outputs.logits.view(-1, vocab_size), y.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Few-shot step {step+1}/{steps}, Loss: {loss.item():.4f}")

def configure_yarn_for_length(model, base_len, target_len, device):
    """配置YaRN用于指定长度的外推"""
    model.config.yarn_enabled = True
    model.config.yarn_max_position_embeddings = base_len
    model.config.yarn_target_max_position_embeddings = target_len
    model.config.max_sequence_length = target_len
    
    # 更新每个block的rotary embedding
    if hasattr(model.transformer, 'blocks'):
        blocks = model.transformer.blocks
    elif hasattr(model.transformer, 'block_groups'):
        blocks = [block for group in model.transformer.block_groups for block in group]
    else:
        blocks = []
    
    for block in blocks:
        if hasattr(block, 'rotary_emb') and block.rotary_emb is not None:
            block.rotary_emb.inv_freq = block.rotary_emb.get_inv_freq(device)
            if hasattr(block.rotary_emb, '_cache'):
                block.rotary_emb._cache.clear()
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Few-Shot Extrapolation Evaluation V2")
    
    # 模型路径
    parser.add_argument("--model_path", type=str, required=True,
                        help="预训练模型路径")
    parser.add_argument("--model_size", type=str, default="300M",
                        choices=["20M", "60M", "300M"])
    
    # 数据路径 - 支持train和validation
    parser.add_argument("--arxiv_train_path", type=str, required=True,
                        help="ArXiv训练集路径（用于few-shot采样）")
    parser.add_argument("--arxiv_val_path", type=str, required=True,
                        help="ArXiv验证集路径（用于测试）")
    parser.add_argument("--local_tokenizer_path", type=str, required=True)
    
    # Few-shot配置 - 支持多种K值
    parser.add_argument("--few_shot_k", type=int, default=128,
                        help="Few-shot样本数量")
    parser.add_argument("--few_shot_steps", type=int, default=50,
                        help="Few-shot微调步数")
    parser.add_argument("--few_shot_lr", type=float, default=5e-6,
                        help="Few-shot学习率")
    
    # 外推测试长度
    parser.add_argument("--base_len", type=int, default=512,
                        help="预训练时的基础长度")
    parser.add_argument("--test_lengths", type=int, nargs='+', 
                        default=[1024, 2048, 4096],
                        help="测试的外推长度列表")
    
    # 模型配置（必须与预训练一致）
    parser.add_argument("--pe_type", type=str, required=True,
                        choices=["rope", "hipe", "rope_yarn", "hipe_yarn"])
    parser.add_argument("--sigma", type=float, default=700.0)
    parser.add_argument("--rope_scaling_threshold", type=int, default=7)
    parser.add_argument("--decay_func", type=str, default="gaussian")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="extrap_results.json")
    parser.add_argument("--eval_batches", type=int, default=None)
    parser.add_argument("--use_train_for_fewshot", action="store_true",
                        help="从arxiv_train采样few-shot样本")
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print(f"Few-Shot Extrapolation Evaluation V2")
    print(f"Few-shot K: {args.few_shot_k}, Steps: {args.few_shot_steps}")
    print(f"Data: {'arxiv_train' if args.use_train_for_fewshot else 'arxiv_val'} (few-shot) + arxiv_val (test)")
    print("="*60)
    
    # 加载Tokenizer
    print(f"Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer_path, local_files_only=True)
    except:
        from OLMo.olmo.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_pretrained(args.local_tokenizer_path, eos_token_id=50256, pad_token_id=50256)
    
    raw_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    vocab_size = ((raw_vocab_size + 63) // 64) * 64
    
    # 加载ArXiv数据
    print(f"Loading ArXiv data...")
    arxiv_train_full = load_from_disk(args.arxiv_train_path)
    arxiv_val_full = load_from_disk(args.arxiv_val_path)
    
    print(f"  Train set: {len(arxiv_train_full)} samples")
    print(f"  Val set: {len(arxiv_val_full)} samples")
    
    # Few-shot数据来源
    if args.use_train_for_fewshot:
        few_shot_source = arxiv_train_full
        print(f"  Using arxiv_train for few-shot ({args.few_shot_k} samples)")
    else:
        few_shot_source = arxiv_val_full
        print(f"  Using arxiv_val for few-shot ({args.few_shot_k} samples)")
    
    # 采样few-shot数据
    few_shot_k = min(args.few_shot_k, len(few_shot_source))
    few_shot_indices = random.sample(range(len(few_shot_source)), few_shot_k)
    few_shot_ds = few_shot_source.select(few_shot_indices)
    
    # 测试数据始终用validation
    test_ds = arxiv_val_full
    
    print(f"  Few-shot: {len(few_shot_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)
    
    print("Tokenizing...")
    tokenized_few_shot = few_shot_ds.map(tokenize_function, batched=True,
                                         remove_columns=few_shot_ds.column_names,
                                         num_proc=4, desc="Tokenizing few-shot")
    tokenized_test = test_ds.map(tokenize_function, batched=True,
                                 remove_columns=test_ds.column_names,
                                 num_proc=4, desc="Tokenizing test")
    
    # 构建模型配置
    if args.model_size == "300M":
        cur_d, cur_h, cur_l, cur_mlp = 1024, 16, 16, 8
    elif args.model_size == "60M":
        cur_d, cur_h, cur_l, cur_mlp = 512, 8, 8, 8
    else:
        cur_d, cur_h, cur_l, cur_mlp = 256, 8, 8, 8
    
    use_rope = args.pe_type in ["rope", "rope_yarn", "hipe", "hipe_yarn"]
    use_hipe = args.pe_type in ["hipe", "hipe_yarn"]
    
    cfg = ModelConfig(
        d_model=cur_d, n_heads=cur_h, n_layers=cur_l, mlp_ratio=cur_mlp,
        max_sequence_length=args.base_len,
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
    
    # 加载模型
    print(f"Loading model from: {args.model_path}")
    model = OLMo(cfg).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded. Parameters: {count_parameters(model)/1e6:.1f}M")
    
    # 准备结果存储
    results = {
        "config": vars(args),
        "few_shot": {},
        "base_eval": {},
        "extrap_eval": {},
    }
    
    # Collate函数
    def collate_fn_base(batch):
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        data = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=50256)
        block_size = args.base_len + 1
        if data.shape[1] >= block_size:
            data = data[:, :block_size]
        else:
            pad_len = block_size - data.shape[1]
            data = torch.nn.functional.pad(data, (0, pad_len), value=50256)
        return data[:, :-1].contiguous(), data[:, 1:].contiguous()
    
    # ==================== 1. Few-Shot适应 ====================
    print(f"\n{'='*60}")
    print(f"1. Few-Shot Adaptation ({few_shot_k} samples, {args.few_shot_steps} steps)")
    print(f"{'='*60}")
    
    few_shot_loader = DataLoader(tokenized_few_shot, batch_size=4,
                                 shuffle=True, collate_fn=collate_fn_base)
    
    # 适应前评估
    print("Evaluating before adaptation...")
    loss_before, ppl_before, _ = evaluate_perplexity(
        model, DataLoader(tokenized_test, batch_size=4, collate_fn=collate_fn_base),
        vocab_size, device, args.eval_batches
    )
    results["few_shot"]["before"] = {"loss": loss_before, "ppl": ppl_before}
    print(f"Before adapt | Loss: {loss_before:.4f} | PPL: {ppl_before:.2f}")
    
    # 进行适应
    if args.few_shot_steps > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.few_shot_lr)
        few_shot_adaptation(model, few_shot_loader, optimizer,
                           args.few_shot_steps, vocab_size, device)
        
        # 适应后评估（base长度）
        print("Evaluating after adaptation (base length)...")
        loss_after, ppl_after, _ = evaluate_perplexity(
            model, DataLoader(tokenized_test, batch_size=4, collate_fn=collate_fn_base),
            vocab_size, device, args.eval_batches
        )
        results["few_shot"]["after"] = {"loss": loss_after, "ppl": ppl_after}
        results["base_eval"]["after_adapt"] = {"length": args.base_len, "loss": loss_after, "ppl": ppl_after}
        print(f"After adapt | Loss: {loss_after:.4f} | PPL: {ppl_after:.2f}")
        print(f"Improvement: {((ppl_before - ppl_after) / ppl_before * 100):+.2f}%")
    
    # ==================== 2. 外推评估（启用YaRN）====================
    print(f"\n{'='*60}")
    print(f"2. Extrapolation Evaluation with YaRN")
    print(f"{'='*60}")
    
    for test_len in args.test_lengths:
        print(f"\n--- Testing length: {test_len} ---")
        
        # 配置YaRN
        if args.pe_type in ["rope_yarn", "hipe_yarn"]:
            model = configure_yarn_for_length(model, args.base_len, test_len, device)
            print(f"YaRN configured: {args.base_len} -> {test_len}")
        else:
            model.config.max_sequence_length = test_len
            print(f"Direct extrapolation (no YaRN): length={test_len}")
        
        # 准备数据
        def collate_fn_extrap(batch):
            input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
            data = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=50256)
            block_size = test_len + 1
            if data.shape[1] >= block_size:
                data = data[:, :block_size]
            else:
                pad_len = block_size - data.shape[1]
                data = torch.nn.functional.pad(data, (0, pad_len), value=50256)
            return data[:, :-1].contiguous(), data[:, 1:].contiguous()
        
        test_loader = DataLoader(tokenized_test, batch_size=max(1, 4 * args.base_len // test_len),
                                 collate_fn=collate_fn_extrap)
        
        # 评估
        loss, ppl, tokens = evaluate_perplexity(model, test_loader, vocab_size,
                                                 device, args.eval_batches)
        
        results["extrap_eval"][test_len] = {
            "length": test_len,
            "loss": loss,
            "ppl": ppl,
            "tokens_evaluated": tokens
        }
        
        base_ppl = results['base_eval'].get('after_adapt', results.get('few_shot', {}).get('before', {})).get('ppl', ppl)
        ratio = ppl / base_ppl if base_ppl else float('inf')
        print(f"Length {test_len} | Loss: {loss:.4f} | PPL: {ppl:.2f} | Ratio: {ratio:.2f}x")
    
    # ==================== 保存结果 ====================
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.output_file}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Summary (K={few_shot_k}):")
    print(f"{'='*60}")
    base_ppl = results.get('few_shot', {}).get('after', {}).get('ppl') or results.get('few_shot', {}).get('before', {}).get('ppl', 0)
    print(f"Base ({args.base_len}): PPL = {base_ppl:.2f}")
    for length, res in results["extrap_eval"].items():
        ratio = res['ppl'] / base_ppl if base_ppl else float('inf')
        print(f"Extrap ({length}): PPL = {res['ppl']:.2f} (ratio: {ratio:.2f}x)")

if __name__ == "__main__":
    main()
