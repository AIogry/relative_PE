"""
Few-Shot外推评估脚本 - ArXiv版本
使用ArXiv数据进行few-shot和测试（与C4不同领域）
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

def main():
    parser = argparse.ArgumentParser(description="Few-Shot Extrapolation Evaluation (ArXiv)")
    
    # 模型路径
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="300M",
                        choices=["20M", "60M", "300M"])
    
    # ArXiv数据路径
    parser.add_argument("--arxiv_train_path", type=str, 
                        default="/data/qijunrong/03-proj/PE/arxiv_data/arxiv_train")
    parser.add_argument("--arxiv_val_path", type=str,
                        default="/data/qijunrong/03-proj/PE/arxiv_data/arxiv_validation")
    parser.add_argument("--local_tokenizer_path", type=str, required=True)
    
    # Few-shot配置
    parser.add_argument("--few_shot_k", type=int, default=256,
                        help="从ArXiv train采样的样本数")
    parser.add_argument("--few_shot_steps", type=int, default=50)
    parser.add_argument("--few_shot_lr", type=float, default=5e-6)
    
    # 【关键参数】是否在外推长度下先训练
    parser.add_argument("--adapt_on_extrap", action="store_true",
                        help="在外推长度下进行few-shot训练（而非base长度）")
    
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
    parser.add_argument("--output_file", type=str, default="arxiv_extrap_results.json")
    parser.add_argument("--eval_batches", type=int, default=None)
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print(f"Few-Shot Extrapolation Evaluation (ArXiv)")
    print(f"Data: ArXiv (different domain from C4)")
    print(f"Few-shot K: {args.few_shot_k}")
    if args.adapt_on_extrap:
        print(f"Mode: Adapt on EXTRAP length (not base length)")
    else:
        print(f"Mode: Adapt on BASE length ({args.base_len})")
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
    train_full = load_from_disk(args.arxiv_train_path)
    val_full = load_from_disk(args.arxiv_val_path)
    
    print(f"  Train: {len(train_full)} samples")
    print(f"  Val: {len(val_full)} samples")
    
    # 从ArXiv train采样few-shot数据
    few_shot_k = min(args.few_shot_k, len(train_full))
    few_shot_indices = random.sample(range(len(train_full)), few_shot_k)
    few_shot_ds = train_full.select(few_shot_indices)
    
    # 测试数据用ArXiv validation
    test_ds = val_full
    
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
    
    # Collate函数（base长度）
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
    
    # ==================== 1. Base长度评估（不训练）====================
    print(f"\n{'='*60}")
    print(f"1. Base Length ({args.base_len}) Evaluation - No Adaptation")
    print(f"{'='*60}")
    
    print("Evaluating on base length (no adaptation)...")
    loss_base, ppl_base, _ = evaluate_perplexity(
        model, DataLoader(tokenized_test, batch_size=4, collate_fn=collate_fn_base),
        vocab_size, device, args.eval_batches
    )
    results["base_eval"]["no_adapt"] = {"length": args.base_len, "loss": loss_base, "ppl": ppl_base}
    print(f"Base length | Loss: {loss_base:.4f} | PPL: {ppl_base:.2f}")
    
    # ==================== 2. Few-Shot Adaptation（可选）====================
    if args.few_shot_steps > 0:
        print(f"\n{'='*60}")
        if args.adapt_on_extrap:
            print(f"2. Few-Shot Adaptation on EXTRAP Length")
        else:
            print(f"2. Few-Shot Adaptation on BASE Length ({args.base_len})")
        print(f"{'='*60}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.few_shot_lr)
        
        if args.adapt_on_extrap:
            # 在每个外推长度下分别训练（更复杂）
            print("Note: Will adapt on each extrap length separately")
            # 这里先跳过，在下面的循环中处理
        else:
            # 在base长度下训练
            few_shot_loader = DataLoader(tokenized_few_shot, batch_size=4,
                                         shuffle=True, collate_fn=collate_fn_base)
            few_shot_adaptation(model, few_shot_loader, optimizer,
                               args.few_shot_steps, vocab_size, device)
            
            # 训练后评估base长度
            print("Evaluating after adaptation (base length)...")
            loss_after, ppl_after, _ = evaluate_perplexity(
                model, DataLoader(tokenized_test, batch_size=4, collate_fn=collate_fn_base),
                vocab_size, device, args.eval_batches
            )
            results["few_shot"]["after"] = {"loss": loss_after, "ppl": ppl_after}
            results["base_eval"]["after_adapt"] = {"length": args.base_len, "loss": loss_after, "ppl": ppl_after}
            print(f"After adapt | Loss: {loss_after:.4f} | PPL: {ppl_after:.2f}")
            print(f"Improvement: {((ppl_base - ppl_after) / ppl_base * 100):+.2f}%")
    
    # ==================== 3. 外推评估 ====================
    print(f"\n{'='*60}")
    print(f"3. Extrapolation Evaluation")
    print(f"{'='*60}")
    
    for test_len in args.test_lengths:
        print(f"\n--- Testing length: {test_len} ---")
        
        if args.pe_type in ["rope_yarn", "hipe_yarn"]:
            model = configure_yarn_for_length(model, args.base_len, test_len, device)
        else:
            model.config.max_sequence_length = test_len
            print(f"Direct extrapolation (no YaRN): length={test_len}")
        
        # 【关键】如果指定了在外推长度下adapt
        if args.adapt_on_extrap and args.few_shot_steps > 0:
            print(f"Adapting on length {test_len}...")
            
            def collate_fn_adapt(batch):
                input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
                data = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=50256)
                block_size = test_len + 1
                if data.shape[1] >= block_size:
                    data = data[:, :block_size]
                else:
                    pad_len = block_size - data.shape[1]
                    data = torch.nn.functional.pad(data, (0, pad_len), value=50256)
                return data[:, :-1].contiguous(), data[:, 1:].contiguous()
            
            few_shot_loader = DataLoader(tokenized_few_shot, batch_size=4,
                                         shuffle=True, collate_fn=collate_fn_adapt)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.few_shot_lr)
            few_shot_adaptation(model, few_shot_loader, optimizer,
                               args.few_shot_steps, vocab_size, device)
        
        # 准备测试数据
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
        
        batch_size = max(1, 4 * args.base_len // test_len)
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
        
        # 计算ratio（相对于base）
        base_ppl = results['base_eval'].get('after_adapt', 
                                            results['base_eval'].get('no_adapt', {})).get('ppl', ppl)
        ratio = ppl / base_ppl if base_ppl else float('inf')
        print(f"Length {test_len} | Loss: {loss:.4f} | PPL: {ppl:.2f} | Ratio: {ratio:.2f}x")
    
    # ==================== 保存结果 ====================
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.output_file}")
    
    print(f"\n{'='*60}")
    print(f"Summary ({args.pe_type}, K={few_shot_k}):")
    print(f"{'='*60}")
    base_ppl = results.get('base_eval', {}).get('after_adapt', {}).get('ppl') or \
               results.get('base_eval', {}).get('no_adapt', {}).get('ppl', 0)
    print(f"Base ({args.base_len}): PPL = {base_ppl:.2f}")
    for length, res in results["extrap_eval"].items():
        ratio = res['ppl'] / base_ppl if base_ppl else float('inf')
        print(f"Extrap ({length}): PPL = {res['ppl']:.2f} (ratio: {ratio:.2f}x)")
    print("="*60)

if __name__ == "__main__":
    main()
