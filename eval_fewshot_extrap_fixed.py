"""
Few-Shot外推评估脚本 - 修复版
修复YaRN配置bug，使用C4数据进行few-shot
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
    """
    配置YaRN用于指定长度的外推
    【修复】：正确设置所有必要的YaRN参数
    """
    print(f"Configuring YaRN: base={base_len}, target={target_len}")
    
    # 设置配置参数
    model.config.yarn_enabled = True
    model.config.yarn_max_position_embeddings = base_len
    model.config.yarn_target_max_position_embeddings = target_len
    model.config.max_sequence_length = target_len
    
    # 【关键修复】：确保yarn_beta参数存在
    if not hasattr(model.config, 'yarn_beta_slow'):
        model.config.yarn_beta_slow = 1.0  # 默认值
    if not hasattr(model.config, 'yarn_beta_fast'):
        model.config.yarn_beta_fast = 32.0  # 默认值
    
    scale = target_len / base_len
    print(f"  YaRN scale factor: {scale:.2f}x")
    
    # 更新每个block的rotary embedding
    if hasattr(model.transformer, 'blocks'):
        blocks = model.transformer.blocks
    elif hasattr(model.transformer, 'block_groups'):
        blocks = [block for group in model.transformer.block_groups for block in group]
    else:
        blocks = []
        print("  Warning: Could not find transformer blocks!")
    
    mscale_values = []
    for i, block in enumerate(blocks):
        if hasattr(block, 'rotary_emb') and block.rotary_emb is not None:
            # 重新计算inv_freq和mscale
            block.rotary_emb.inv_freq = block.rotary_emb.get_inv_freq(device)
            
            # 【关键修复】：保存并打印mscale值
            if hasattr(block.rotary_emb, '_mscale'):
                mscale_values.append(block.rotary_emb._mscale)
            
            # 清空缓存，强制重新计算位置编码
            if hasattr(block.rotary_emb, '_cache'):
                block.rotary_emb._cache.clear()
            
            # 如果是HIPE，确保scale_factor保持不变（基于base_inv_freq）
            if hasattr(block.rotary_emb, 'base_inv_freq'):
                # 验证：base_inv_freq不应该改变
                pass
    
    if mscale_values:
        print(f"  YaRN mscale range: [{min(mscale_values):.4f}, {max(mscale_values):.4f}]")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Few-Shot Extrapolation Evaluation (Fixed)")
    
    # 模型路径
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="300M",
                        choices=["20M", "60M", "300M"])
    
    # 数据路径 - 使用C4进行few-shot和测试
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="C4数据集路径（包含c4_30M_train/val）")
    parser.add_argument("--local_tokenizer_path", type=str, required=True)
    
    # Few-shot配置
    parser.add_argument("--few_shot_k", type=int, default=2000,
                        help="Few-shot样本数量（从C4采样）")
    parser.add_argument("--few_shot_steps", type=int, default=200,
                        help="Few-shot微调步数")
    parser.add_argument("--few_shot_lr", type=float, default=1e-5,
                        help="Few-shot学习率")
    
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
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="extrap_results_fixed.json")
    parser.add_argument("--eval_batches", type=int, default=None)
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print(f"Few-Shot Extrapolation Evaluation (Fixed)")
    print(f"Data: C4 (same domain as training)")
    print(f"Few-shot K: {args.few_shot_k}")
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
    
    # 加载C4数据（与预训练相同领域）
    print(f"Loading C4 data from: {args.dataset_path}")
    train_path = os.path.join(args.dataset_path, "c4_30M_train")
    val_path = os.path.join(args.dataset_path, "c4_30M_validation")
    
    train_full = load_from_disk(train_path)
    val_full = load_from_disk(val_path)
    
    print(f"  Train: {len(train_full)} samples")
    print(f"  Val: {len(val_full)} samples")
    
    # 从C4训练集采样few-shot数据（与预训练同领域）
    few_shot_k = min(args.few_shot_k, len(train_full))
    few_shot_indices = random.sample(range(len(train_full)), few_shot_k)
    few_shot_ds = train_full.select(few_shot_indices)
    
    # 从C4验证集采样测试数据
    test_size = min(5000, len(val_full))
    test_ds = val_full.select(range(test_size))
    
    print(f"  Few-shot: {len(few_shot_ds)} samples (from C4 train)")
    print(f"  Test: {len(test_ds)} samples (from C4 val)")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)
    
    print("Tokenizing...")
    tokenized_few_shot = few_shot_ds.map(tokenize_function, batched=True,
                                         remove_columns=few_shot_ds.column_names,
                                         num_proc=8, desc="Tokenizing few-shot")
    tokenized_test = test_ds.map(tokenize_function, batched=True,
                                 remove_columns=test_ds.column_names,
                                 num_proc=8, desc="Tokenizing test")
    
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
        yarn_enabled=False,  # 初始禁用YaRN
        use_scaled_rope1=use_hipe,
        scaled_rope_sigma=args.sigma if use_hipe else 1.0,
        rope_scaling_threshold=args.rope_scaling_threshold if use_hipe else -1,
        decay_func=args.decay_func if use_hipe else 'gaussian',
        flash_attention=True,
    )
    
    # 【修复】：添加YaRN参数到config
    cfg.yarn_beta_slow = 1.0
    cfg.yarn_beta_fast = 32.0
    cfg.yarn_max_position_embeddings = args.base_len
    
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
    print(f"   Domain: C4 (same as pretraining)")
    print(f"{'='*60}")
    
    few_shot_loader = DataLoader(tokenized_few_shot, batch_size=8,
                                 shuffle=True, collate_fn=collate_fn_base)
    
    # 适应前评估
    print("Evaluating before adaptation...")
    loss_before, ppl_before, _ = evaluate_perplexity(
        model, DataLoader(tokenized_test, batch_size=8, collate_fn=collate_fn_base),
        vocab_size, device, args.eval_batches
    )
    results["few_shot"]["before"] = {"loss": loss_before, "ppl": ppl_before}
    print(f"Before adapt | Loss: {loss_before:.4f} | PPL: {ppl_before:.2f}")
    
    # 进行适应
    if args.few_shot_steps > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.few_shot_lr)
        few_shot_adaptation(model, few_shot_loader, optimizer,
                           args.few_shot_steps, vocab_size, device)
        
        # 适应后评估
        print("Evaluating after adaptation...")
        loss_after, ppl_after, _ = evaluate_perplexity(
            model, DataLoader(tokenized_test, batch_size=8, collate_fn=collate_fn_base),
            vocab_size, device, args.eval_batches
        )
        results["few_shot"]["after"] = {"loss": loss_after, "ppl": ppl_after}
        results["base_eval"]["after_adapt"] = {"length": args.base_len, "loss": loss_after, "ppl": ppl_after}
        print(f"After adapt | Loss: {loss_after:.4f} | PPL: {ppl_after:.2f}")
        print(f"Improvement: {((ppl_before - ppl_after) / ppl_before * 100):+.2f}%")
    
    # ==================== 2. 外推评估 ====================
    print(f"\n{'='*60}")
    print(f"2. Extrapolation Evaluation")
    print(f"{'='*60}")
    
    for test_len in args.test_lengths:
        print(f"\n--- Testing length: {test_len} ---")
        
        if args.pe_type in ["rope_yarn", "hipe_yarn"]:
            # 使用YaRN进行外推
            model = configure_yarn_for_length(model, args.base_len, test_len, device)
        else:
            # 直接外推（无YaRN）
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
        
        # 动态调整batch size
        batch_size = max(1, 8 * args.base_len // test_len)
        test_loader = DataLoader(tokenized_test, batch_size=batch_size,
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
    print(f"Summary ({args.pe_type}, K={few_shot_k}):")
    print(f"{'='*60}")
    base_ppl = results.get('few_shot', {}).get('after', {}).get('ppl') or results.get('few_shot', {}).get('before', {}).get('ppl', 0)
    print(f"Base ({args.base_len}): PPL = {base_ppl:.2f}")
    for length, res in results["extrap_eval"].items():
        ratio = res['ppl'] / base_ppl if base_ppl else float('inf')
        print(f"Extrap ({length}): PPL = {res['ppl']:.2f} (ratio: {ratio:.2f}x)")
    print("="*60)

if __name__ == "__main__":
    main()
