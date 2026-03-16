"""
Parallel training script for 30M dataset using 2 GPUs with cached data.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from itertools import islice
import argparse
import os
import math
import json
from datetime import datetime
import random
import numpy as np
import pickle
import sys

# Import necessary components from OLMo and Hugging Face datasets
from OLMo.olmo.config import TrainConfig
from OLMo.olmo.model import OLMo
from OLMo.olmo.optim import build_optimizer, build_scheduler
from OLMo.olmo.tokenizer import Tokenizer
from datasets import load_dataset, interleave_datasets, load_from_disk
from tqdm import tqdm


class CachedDataset(Dataset):
    """Dataset that loads preprocessed data from cache."""
    def __init__(self, cache_path):
        print(f"Loading cached data from: {cache_path}")
        with open(cache_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} sequences from cache")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.data[idx], dtype=torch.long)}


def setup(rank, world_size, master_addr='localhost', master_port='12355'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def evaluate_model(model, eval_loader, cfg, device, max_eval_steps, rank):
    """Evaluates the model and returns perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for step, batch in enumerate(islice(eval_loader, max_eval_steps)):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids[:, 1:].contiguous()
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=cfg.model.pad_token_id, reduction='sum')
            total_loss += loss.item()
            total_tokens += (labels != cfg.model.pad_token_id).sum().item()
    
    # Synchronize across all processes
    total_loss_tensor = torch.tensor([total_loss, total_tokens], device=device)
    dist.all_reduce(total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
    
    total_loss_all = total_loss_tensor[0].item()
    total_tokens_all = total_loss_tensor[1].item()
    
    if total_tokens_all == 0: 
        perplexity = float('inf')
    else:
        average_loss = total_loss_all / total_tokens_all
        perplexity = math.exp(average_loss)
    
    return perplexity


def run_experiment_ddp(rank, world_size, cfg, max_steps, save_interval, results_filename, 
                      train_cache_path, val_cache_path, max_eval_steps=200):
    """
    Runs a single training and evaluation experiment for a given configuration using DDP.
    """
    print(f"Running DDP process on rank {rank} of {world_size}")
    
    # Setup the distributed environment
    setup(rank, world_size)
    
    print(f"Starting Experiment for: {cfg.run_name} on GPU {rank}")
    print(f"Global Batch Size: {cfg.global_train_batch_size}, Micro Batch Size: {cfg.device_train_microbatch_size}")
    print(f"Using Position Embedding: {'RoPE' if cfg.model.rope else 'FoPE'}")
    if cfg.model.rope:
        print(f"use_scaledRoPE: {cfg.model.use_scaled_rope}, use_scaledrotaryembedding: {cfg.model.use_scaledrotaryembedding}, sin weight: {cfg.model.sin_lambda}, cos weight: {cfg.model.cos_sigma}, scaled_rope_sigma: {cfg.model.scaled_rope_sigma}, decay_func: {cfg.model.decay_func}")
    
    # --- Model and Tokenizer Initialization ---
    device = torch.device(f"cuda:{rank}")
    model = OLMo(cfg.model)
    model.to(device)
    
    # Load tokenizer only on rank 0
    if rank == 0:
        tokenizer = Tokenizer.from_pretrained(
            "allenai/olmo-1b",
            eos_token_id=cfg.model.eos_token_id,
            pad_token_id=cfg.model.pad_token_id
        )
        print("Loaded tokenizer from Hugging Face: allenai/olmo-1b")
    else:
        tokenizer = None
    
    # Synchronize tokenizer loading
    dist.barrier()
    
    # --- Data Loading with Caching ---
    print(f"Loading data from cache on rank {rank}...")
    train_dataset = CachedDataset(train_cache_path)
    val_dataset = CachedDataset(val_cache_path)
    
    def collate_fn(batch):
        input_ids_list = [item['input_ids'] for item in batch]
        return {"input_ids": torch.stack(input_ids_list)}
    
    # Create data loaders with distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.device_train_microbatch_size, 
        collate_fn=collate_fn,
        sampler=train_sampler,
        shuffle=False  # Shuffle is handled by DistributedSampler
    )
    eval_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.device_train_microbatch_size, 
        collate_fn=collate_fn,
        sampler=val_sampler,
        shuffle=False
    )
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # --- Optimizer and Scheduler ---
    optimizer = build_optimizer(cfg, model.module)  # Use the original model for optimizer
    scheduler = build_scheduler(cfg)
    max_grad_norm = getattr(cfg.optimizer, 'max_grad_norm', 1.0)

    # --- Training & Evaluation Loop ---
    results = {}
    save_dir = f"./checkpoints/{cfg.run_name}"
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    # Synchronize across processes
    dist.barrier()
    
    print(f"Starting training for {cfg.run_name} on rank {rank}...")
    
    for step, batch in enumerate(train_loader):
        if step >= max_steps:
            break
            
        model.train()
        input_ids = batch["input_ids"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=cfg.model.pad_token_id)
        
        # Scale loss by world size for proper averaging
        loss = loss / world_size
        loss.backward()
        
        # All-reduce gradients across all processes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Sync learning rate across processes
        current_lr = scheduler.get_lr(cfg.optimizer.learning_rate, step, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        if (step + 1) % 100 == 0 and rank == 0:
            print(f"Step [{step+1}/{max_steps}], Loss: {loss.item() * world_size:.4f}")

        if (step + 1) % save_interval == 0 and rank == 0:
            # Only rank 0 evaluates and saves
            perplexity = evaluate_model(model.module, eval_loader, cfg, device, max_eval_steps, rank)
            results[step + 1] = perplexity
            print(f"Step [{step+1}], Perplexity: {perplexity:.4f}")
            
            checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step+1}.pt")
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            # Save results incrementally
            with open(results_filename, "w") as f:
                json.dump({cfg.run_name: results}, f, indent=4)
            print(f"Intermediate results saved to {results_filename}")
    
    if rank == 0:
        print(f"\nTraining finished for {cfg.run_name}. Final results saved to {results_filename}")
    
    cleanup()


def main():
    # --- Set up argument parser ---
    parser = argparse.ArgumentParser(description="Train a single OLMo model with a given configuration using 2 GPUs.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file (e.g., ./configs/olmo-60m-simplified.yaml)")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps for the experiment.")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval for saving checkpoints and evaluating the model.")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval.")
    parser.add_argument("--seed", type=int, default=6198, help="Random seed.")
    # --- Arguments to override config file settings ---
    parser.add_argument("--position_embedding", type=str, choices=['rope', 'fope'], default="rope", help="Override the position embedding method.")
    parser.add_argument("--use_scaled_rope", action="store_true", help="Enable Scaled RoPE.")
    parser.add_argument("--sin_lambda", type=float, default=None, help="Lambda for sin scaling in Scaled RoPE.")
    parser.add_argument("--cos_sigma", type=float, default=None, help="Sigma for cos scaling in Scaled RoPE.")
    # for shujun's scaled rotary embedding
    parser.add_argument("--use_scaledrotaryembedding", action="store_true", help="Enable shujun's Scaled RoPE.")
    parser.add_argument("--scaled_rope_sigma",type=float,default=None,help = "Place field size for scaledRoPE")
    parser.add_argument("--sigmas", nargs='+', type=float, default=None, help="List of sigma values to scan.")
    parser.add_argument("--decay_func", type=str, choices=['exp', 'gaussian','power'], default=None, help="Distance decay function for scaledRoPE")

    parser.add_argument("--run_name", type=str, default="olmo-60m-train", help="Custom name for the run. Overrides the name in the config.")
    parser.add_argument("--batch_size", type=int, default=32, help="Global training batch size. Overrides the config file setting for all experiments.")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Device micro batch size. If not set, defaults to the global batch size.")
    
    # Add cache path arguments
    parser.add_argument("--train_cache_path", type=str, required=True, help="Path to cached training data")
    parser.add_argument("--val_cache_path", type=str, required=True, help="Path to cached validation data")
    
    args = parser.parse_args()

    # --- Remove randomness for reproducibility ---
    seed = 6198
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # --- Load and configure the experiment ---
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at '{args.config}'")
        return

    cfg = TrainConfig.load(args.config)
    if args.batch_size is not None:
        cfg.global_train_batch_size = args.batch_size
        print(f"Overriding global batch size for all experiments: {cfg.global_train_batch_size}")

    if args.micro_batch_size is not None:
        cfg.device_train_microbatch_size = args.micro_batch_size
        print(f"Overriding micro batch size for all experiments: {cfg.device_train_microbatch_size}")
    else:
        cfg.device_train_microbatch_size = cfg.global_train_batch_size
    # Override config with command-line arguments if provided
    if args.run_name:
        cfg.run_name = args.run_name
    
    if args.position_embedding == 'fope':
        cfg.model.fope = True
        cfg.model.rope = False
        if args.fope_d is not None:
            cfg.model.fope_freq_d = args.fope_d
        if args.fope_sigma is not None:
            cfg.model.fope_freq_var_sigma = args.fope_sigma
    elif args.position_embedding == 'rope':
        cfg.model.fope = False
        cfg.model.rope = True
        if args.use_scaled_rope:
            cfg.model.use_scaled_rope = True
            cfg.model.use_scaledrotaryembedding = False  # 注意
            cfg.model.sin_lambda = args.sin_lambda
            cfg.model.cos_sigma = args.cos_sigma
        elif args.use_scaledrotaryembedding:        # shujun zhou
            cfg.model.use_scaledrotaryembedding = True
            cfg.model.use_scaled_rope = False       # 注意
            if args.sigmas is not None:
                assert len(args.sigmas) == cfg.model.n_heads, f"scaled rope sigmas should have the same length as n_heads: {cfg.model.n_heads}" 
                cfg.model.scaled_rope_sigmas = args.sigmas
            else:
                cfg.model.scaled_rope_sigma = args.scaled_rope_sigma
            if args.decay_func is not None:
                cfg.model.decay_func = args.decay_func
        else:
            cfg.model.use_scaled_rope = False
            cfg.model.sin_lambda = None
            cfg.model.cos_sigma = None

    # --- Prepare results file ---
    results_dir = "training_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_filename = os.path.join(results_dir, f"{cfg.run_name}_{timestamp}.json")

    # --- Run the distributed experiment ---
    world_size = 2  # Use 2 GPUs
    mp.spawn(
        run_experiment_ddp,
        args=(world_size, cfg, args.max_steps, args.save_interval, results_filename, 
              args.train_cache_path, args.val_cache_path),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()