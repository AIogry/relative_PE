import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import argparse
import json
import os
import copy
import random
import numpy as np
from datetime import datetime
from torch.amp import autocast

# 引入 OLMo 组件
from OLMo.olmo.model import OLMo
from OLMo.olmo.config import ModelConfig

# --- 工具函数: 固定随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f">>> Random Seed set to: {seed}")

# --- Dataset 1: 标准关联回忆 (Standard) ---
class AssociativeRecallDataset(IterableDataset):
    def __init__(self, vocab_size=1000, seq_len=64, num_pairs=8):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        assert vocab_size >= num_pairs, "Vocab size must be larger than num_pairs!"

    def __iter__(self):
        # === [新增] 多进程种子处理 ===
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # 这里的 seed 由 PyTorch 主进程种子 + worker_id 自动派生
            seed = worker_info.seed % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        # ===========================

        while True:
            keys = torch.randperm(self.vocab_size)[:self.num_pairs]
            vals = torch.randint(0, self.vocab_size, (self.num_pairs,))
            sequence = []
            for k, v in zip(keys, vals):
                sequence.extend([k, v])
            query_idx = torch.randint(0, self.num_pairs, (1,)).item()
            sequence.append(keys[query_idx])
            sequence.append(vals[query_idx])
            input_ids = torch.tensor(sequence, dtype=torch.long)
            if len(input_ids) < self.seq_len:
                padding = torch.zeros(self.seq_len - len(input_ids), dtype=torch.long)
                input_ids = torch.cat([input_ids, padding])
            yield {"input_ids": input_ids[:self.seq_len]}

# --- Dataset 2: 块复制任务 (Block) ---
class BlockRecallDataset(IterableDataset):
    def __init__(self, vocab_size=100, seq_len=128, num_pairs=4, block_size=3):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.block_size = block_size
        assert vocab_size >= num_pairs, "Vocab too small!"

    def __iter__(self):
        # === [新增] 多进程种子处理 ===
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = worker_info.seed % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        # ===========================

        while True:
            keys = torch.randperm(self.vocab_size)[:self.num_pairs]
            sequence = []
            loss_mask = torch.zeros(self.seq_len, dtype=torch.float)
            query_idx = torch.randint(0, self.num_pairs, (1,)).item()
            target_key = keys[query_idx]
            for i, key in enumerate(keys):
                sequence.append(key)
                block_vals = torch.randint(0, self.vocab_size, (self.block_size,)).tolist()
                sequence.extend(block_vals)
                if i == query_idx:
                    target_block = block_vals
            q_start_idx = len(sequence)
            sequence.append(target_key)
            sequence.extend(target_block)
            input_ids = torch.tensor(sequence, dtype=torch.long)
            
            mask_start = q_start_idx
            mask_end = mask_start + self.block_size
            if mask_end > self.seq_len: mask_end = self.seq_len
            
            if len(input_ids) < self.seq_len:
                padding = torch.zeros(self.seq_len - len(input_ids), dtype=torch.long)
                input_ids = torch.cat([input_ids, padding])
            else:
                input_ids = input_ids[:self.seq_len]
            
            if mask_start < self.seq_len:
                loss_mask[mask_start:mask_end] = 1.0
            yield {"input_ids": input_ids, "loss_mask": loss_mask}

# --- Dataset 3: 位置抖动任务 (Jitter) ---
class JitteredRecallDataset(IterableDataset):
    def __init__(self, vocab_size=100, seq_len=128, num_pairs=8, max_jitter=5):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.max_jitter = max_jitter

    def __iter__(self):
        # === [新增] 多进程种子处理 ===
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = worker_info.seed % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        # ===========================

        while True:
            keys = torch.randperm(self.vocab_size)[:self.num_pairs]
            vals = torch.randint(0, self.vocab_size, (self.num_pairs,))
            sequence = []
            query_idx = torch.randint(0, self.num_pairs, (1,)).item()
            target_key = keys[query_idx]
            target_val = vals[query_idx]
            for i, (k, v) in enumerate(zip(keys, vals)):
                sequence.append(k)
                jitter = torch.randint(0, self.max_jitter + 1, (1,)).item()
                if jitter > 0:
                    noise = torch.randint(0, self.vocab_size, (jitter,)).tolist()
                    sequence.extend(noise)
                sequence.append(v)
            q_idx = len(sequence)
            sequence.append(target_key)
            sequence.append(target_val)
            input_ids = torch.tensor(sequence, dtype=torch.long)
            loss_mask = torch.zeros(self.seq_len, dtype=torch.float)
            if len(input_ids) < self.seq_len:
                padding = torch.zeros(self.seq_len - len(input_ids), dtype=torch.long)
                input_ids = torch.cat([input_ids, padding])
            else:
                input_ids = input_ids[:self.seq_len]
            if q_idx < self.seq_len:
                loss_mask[q_idx] = 1.0
            yield {"input_ids": input_ids, "loss_mask": loss_mask}

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Synthetic Induction")
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=6198, help="Random seed")
    
    parser.add_argument("--use_scaled_rope", action="store_true")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--decay_func", type=str, default="gaussian")
    
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_pairs", type=int, default=10)

    parser.add_argument("--task", type=str, default="standard", choices=["standard", "block", "jitter"])
    parser.add_argument("--block_size", type=int, default=3)
    parser.add_argument("--max_jitter", type=int, default=5)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Starting Run: {args.run_id} | Task: {args.task}")
    
    cfg = ModelConfig(
        d_model=512, n_heads=8, n_layers=8, mlp_ratio=4,
        max_sequence_length=args.seq_len, vocab_size=args.vocab_size,
        rope=True, use_scaled_rope1=args.use_scaled_rope,
        scaled_rope_sigma=args.sigma, decay_func=args.decay_func,
        flash_attention=True
    )
    model = OLMo(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.task == "block":
        dataset = BlockRecallDataset(args.vocab_size, args.seq_len, args.num_pairs, args.block_size)
    elif args.task == "jitter":
        dataset = JitteredRecallDataset(args.vocab_size, args.seq_len, args.num_pairs, args.max_jitter)
    else:
        dataset = AssociativeRecallDataset(args.vocab_size, args.seq_len, args.num_pairs)
    
    # [修改点] 开启多进程加速数据生成
    # num_workers=8: 开启8个子进程（建议与分配的CPU数接近）
    # persistent_workers=True: 保持子进程活跃，避免频繁创建销毁开销
    # pin_memory=True: 加速内存到显存的传输
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=8, 
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2
    )
    iter_loader = iter(loader)

    history = {"steps": [], "loss": [], "accuracy": []}
    running_loss = 0.0
    running_acc = 0.0
    log_interval = 50 

    model.train()
    for step in range(args.steps):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True) if "loss_mask" in batch else None

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids)
            
            # Logits: [B, Seq-1, Vocab]
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            
            if loss_mask is not None:
                # === [关键修改] 极大优化显存 ===
                # 1. 截取对应的 Mask
                active_mask = loss_mask[:, :-1].bool() # 转为布尔类型以便索引
                
                # 2. 直接用 Mask 选出需要计算 Loss 的 token
                # 结果形状从 [B, L, V] 变为 [N_active, V]
                # N_active 远小于 B*L (对于 Jitter 任务，缩小了 128 倍)
                selected_logits = logits[active_mask]
                selected_targets = targets[active_mask]
                
                # 3. 只对有效 token 计算 Loss
                if selected_logits.numel() > 0:
                    loss = nn.functional.cross_entropy(selected_logits, selected_targets)
                    
                    # 计算准确率
                    with torch.no_grad():
                        preds = torch.argmax(selected_logits, dim=-1)
                        correct = (preds == selected_targets).float()
                        acc = correct.mean() # 自动对 active 的求平均
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                    acc = 0.0
            else:
                # --- Standard 任务逻辑 ---
                target_idx = args.num_pairs * 2 
                final_logits = logits[:, target_idx, :] 
                final_targets = targets[:, target_idx]
                loss = nn.functional.cross_entropy(final_logits, final_targets)
                preds = torch.argmax(final_logits, dim=-1)
                acc = (preds == final_targets).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc_val = acc.item() if isinstance(acc, torch.Tensor) else acc
        history["steps"].append(step)
        history["loss"].append(loss.item())
        history["accuracy"].append(acc_val) 
        running_loss += loss.item()
        running_acc += acc_val

        if (step + 1) % log_interval == 0:
            print(f"[{args.run_id} | {args.task}] Step {step+1}: Avg Loss {running_loss/log_interval:.4f} | Avg Acc {running_acc/log_interval:.4f}")
            running_loss = 0.0
            running_acc = 0.0
        elif step == 0:
            print(f"[{args.run_id} | {args.task}] Step 0: Loss {loss.item():.4f} | Acc {acc_val:.4f}")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.run_id}_{args.task}_{timestamp_str}.json"
    save_path = os.path.join(args.output_dir, filename)
    with open(save_path, 'w') as f:
        json.dump({"config": vars(args), "history": history, "timestamp": datetime.now().isoformat()}, f, indent=4)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()