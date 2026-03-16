import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import argparse
import json
import os
import random
import numpy as np
from datetime import datetime
from torch.amp import autocast
from OLMo.olmo.model import OLMo
from OLMo.olmo.config import ModelConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class AssociativeRecallDataset(IterableDataset):
    def __init__(self, vocab_size=1000, seq_len=64, num_pairs=8):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_pairs = num_pairs
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = worker_info.seed % (2**32)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        while True:
            keys = torch.randperm(self.vocab_size)[:self.num_pairs]
            vals = torch.randint(0, self.vocab_size, (self.num_pairs,))
            sequence = []
            for k, v in zip(keys, vals): sequence.extend([k, v])
            query_idx = torch.randint(0, self.num_pairs, (1,)).item()
            sequence.append(keys[query_idx]); sequence.append(vals[query_idx])
            input_ids = torch.tensor(sequence, dtype=torch.long)
            if len(input_ids) < self.seq_len:
                padding = torch.zeros(self.seq_len - len(input_ids), dtype=torch.long)
                input_ids = torch.cat([input_ids, padding])
            yield {"input_ids": input_ids[:self.seq_len]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="60M")
    parser.add_argument("--use_scaled_rope", action="store_true")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--rope_scaling_threshold", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=12000) # 12k 步足够
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128) # 加大BS加速
    parser.add_argument("--vocab_size", type=int, default=50) # Exp1 设置
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_pairs", type=int, default=4)
    args = parser.parse_args()
    set_seed(6198)

    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    
    d_model = 256 if args.model_size == "20M" else 512
    mlp_ratio = 8 if args.model_size == "20M" else 4

    cfg = ModelConfig(
        d_model=d_model, n_heads=8, n_layers=8, mlp_ratio=mlp_ratio,
        max_sequence_length=args.seq_len, vocab_size=args.vocab_size,
        rope=True, use_scaled_rope1=args.use_scaled_rope,
        scaled_rope_sigma=args.sigma, 
        rope_scaling_threshold=args.rope_scaling_threshold,
        flash_attention=True
    )
    model = OLMo(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dataset = AssociativeRecallDataset(args.vocab_size, args.seq_len, args.num_pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    iter_loader = iter(loader)

    print(f">>> Start Training {args.run_id}...")
    model.train()
    for step in range(args.steps):
        try: batch = next(iter_loader)
        except StopIteration: iter_loader = iter(loader); batch = next(iter_loader)
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(input_ids=batch["input_ids"].to(device)).logits[:, :-1, :]
            targets = batch["input_ids"][:, 1:].to(device)
            # 只算最后一个 token (Query Value) 的 loss，加速收敛
            idx = args.num_pairs * 2
            loss = nn.functional.cross_entropy(logits[:, idx, :], targets[:, idx])
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 1000 == 0: print(f"Step {step}: Loss {loss.item():.4f}")

    # === 关键：保存模型 ===
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model2.pt"))
    print(">>> Model Saved!")

if __name__ == "__main__":
    main()