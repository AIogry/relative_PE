"""
modify based on the train_single_model.
Support DDP with 2 GPUs.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from itertools import islice
import argparse
import os
import math
import json
from datetime import datetime
import random
import numpy as np
import torch.distributed as dist

# Import necessary components from OLMo and Hugging Face datasets
from OLMo.olmo.config import TrainConfig
from OLMo.olmo.model import OLMo
from OLMo.olmo.optim import build_optimizer, build_scheduler
from OLMo.olmo.tokenizer import Tokenizer
from datasets import load_dataset, interleave_datasets, load_from_disk


def setup_ddp():
    """Initialize DDP environment."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


def run_experiment(cfg: TrainConfig, max_steps: int, save_interval: int, use_streaming: bool):
    """
    Runs a single training and evaluation experiment for a given configuration.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    if is_main_process():
        print("\n" + "="*80)
        print(f"Starting Experiment for: {cfg.run_name}")
        print(f"Global Batch Size: {cfg.global_train_batch_size}, Micro Batch Size: {cfg.device_train_microbatch_size}")
        print(f"Using Position Embedding: {'RoPE' if cfg.model.rope else 'FoPE'}")
        if cfg.model.rope:
            print(f"use_scaledRoPE: {cfg.model.use_scaled_rope}, use_scaledrotaryembedding: {cfg.model.use_scaledrotaryembedding}")
            print(f"decay_func: {cfg.model.decay_func}")
        print(f"Dataset Streaming: {use_streaming}")
        print(f"World size: {world_size}, Using GPUs: {world_size}")
        print("="*80)

    # --- Model and Tokenizer Initialization ---
    model = OLMo(cfg.model)
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    if is_main_process():
        try:
            tokenizer = Tokenizer.from_pretrained(
                "allenai/olmo-1b",
                eos_token_id=cfg.model.eos_token_id,
                pad_token_id=cfg.model.pad_token_id
            )
            print("Loaded tokenizer from Hugging Face: allenai/olmo-1b")
        except Exception as e:
            print(f"Failed to load HF tokenizer: {e}")
            raise
    else:
        tokenizer = Tokenizer.from_pretrained(
            "allenai/olmo-1b",
            eos_token_id=cfg.model.eos_token_id,
            pad_token_id=cfg.model.pad_token_id
        )

    # --- Data Loading ---
    train_full = load_from_disk("/data/qijunrong/03-proj/PE/c4_train_shards")
    val_full = load_from_disk("/data/qijunrong/03-proj/PE/c4_val_shards")



    #train_size = min(500_000, len(train_full))
    #val_size = min(25_000, len(val_full))

    hf_train_datasets = [train_full.select(range(1000))]
    eval_dataset = val_full.select(range(1000))

    def tokenize_and_chunk(examples):
        all_token_ids = []
        for text in examples.get("text", []):
            if text:
                all_token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        chunk_size = cfg.model.max_sequence_length
        total_tokens = (len(all_token_ids) // chunk_size) * chunk_size
        return {"input_ids": [all_token_ids[i:i+chunk_size] for i in range(0, total_tokens, chunk_size)]}

    processed_train_datasets = [
        ds.map(tokenize_and_chunk, batched=True, batch_size=1000, remove_columns=ds.column_names)
        for ds in hf_train_datasets
    ]
    interleaved_train_dataset = interleave_datasets(processed_train_datasets)
    processed_eval_dataset = eval_dataset.map(tokenize_and_chunk, batched=True, batch_size=1000, remove_columns=eval_dataset.column_names)

    def collate_fn(batch):
        input_ids_list = [item['input_ids'] for item in batch]
        return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long)}

    # Use DistributedSampler for training
    train_sampler = DistributedSampler(
        interleaved_train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        interleaved_train_dataset,
        batch_size=cfg.device_train_microbatch_size,
        collate_fn=collate_fn,
        sampler=train_sampler
    )

    # Validation: only main process or all? (here: all, but you can skip on non-main)
    eval_loader = DataLoader(
        processed_eval_dataset,
        batch_size=cfg.device_train_microbatch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    # --- Optimizer and Scheduler ---
    optimizer = build_optimizer(cfg, model.module)  # use .module for DDP
    scheduler = build_scheduler(cfg)
    max_grad_norm = getattr(cfg.optimizer, 'max_grad_norm', 1.0)

    # --- Training & Evaluation Loop ---
    results = {}
    save_dir = f"/data/qijunrong/03-proj/PE/checkpoints/{cfg.run_name}"
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    for step, batch in enumerate(islice(train_loader, max_steps)):
        model.train()
        input_ids = batch["input_ids"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=cfg.model.pad_token_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        current_lr = scheduler.get_lr(cfg.optimizer.learning_rate, step, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        if is_main_process() and (step + 1) % 100 == 0:
            print(f"Step [{step+1}/{max_steps}], Loss: {loss.item():.4f}")

        if (step + 1) % save_interval == 0:
            # Evaluate on all ranks (or only main); here we do on all for simplicity
            perplexity = evaluate_model(model, eval_loader, cfg, device, max_eval_steps=200)
            
            # Gather perplexity from all ranks (optional: average)
            perplexity_tensor = torch.tensor(perplexity, device=device)
            dist.all_reduce(perplexity_tensor, op=dist.ReduceOp.SUM)
            avg_perplexity = perplexity_tensor.item() / world_size

            if is_main_process():
                results[step + 1] = avg_perplexity
                print(f"Step [{step+1}], Avg Perplexity: {avg_perplexity:.4f}")
                
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step+1}.pt")
                torch.save(model.module.state_dict(), checkpoint_path)  # save .module
                print(f"Checkpoint saved to {checkpoint_path}")
                
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                results_filename = os.path.join("training_results", f"{cfg.run_name}_{timestamp}.json")

                os.makedirs("training_results", exist_ok=True)
                with open(results_filename, "w") as f:
                    json.dump({cfg.run_name: results}, f, indent=4)
                print(f"Intermediate results saved to {results_filename}")
    
    if is_main_process():
        print(f"\nTraining finished for {cfg.run_name}.")

    dist.barrier()  # sync before exit
    return


def evaluate_model(model, eval_loader, cfg, device, max_eval_steps):
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
    
    if total_tokens == 0:
        return float('inf')
    average_loss = total_loss / total_tokens
    return math.exp(average_loss)


def main():
    # --- Set up argument parser ---
    parser = argparse.ArgumentParser(description="Train OLMo with DDP.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--position_embedding", type=str, choices=['rope', 'fope'], default="rope")
    parser.add_argument("--use_scaled_rope", action="store_true")
    parser.add_argument("--sin_lambda", type=float, default=None)
    parser.add_argument("--cos_sigma", type=float, default=None)
    parser.add_argument("--use_scaledrotaryembedding", action="store_true")
    parser.add_argument("--scaled_rope_sigma", type=float, default=None)
    parser.add_argument("--sigmas", nargs='+', type=float, default=None)
    parser.add_argument("--decay_func", type=str, choices=['exp', 'gaussian','power'], default=None)
    parser.add_argument("--run_name", type=str, default="olmo-60m-ddp")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--streaming", action='store_true')
    args = parser.parse_args()

    # --- DDP setup ---
    setup_ddp()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # --- Reproducibility (set on each process) ---
    seed = args.seed + rank  # avoid same seed on all GPUs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # --- Load config ---
    if not os.path.exists(args.config):
        if is_main_process():
            print(f"Error: Config not found: {args.config}")
        cleanup_ddp()
        return

    cfg = TrainConfig.load(args.config)

    # Adjust batch sizes for DDP
    if args.batch_size is not None:
        cfg.global_train_batch_size = args.batch_size
    if args.micro_batch_size is not None:
        cfg.device_train_microbatch_size = args.micro_batch_size
    else:
        cfg.device_train_microbatch_size = cfg.global_train_batch_size // world_size

    if is_main_process():
        print(f"Global batch size: {cfg.global_train_batch_size}")
        print(f"Per-GPU micro batch size: {cfg.device_train_microbatch_size}")

    # Override model config
    if args.run_name:
        cfg.run_name = args.run_name + f"_ddp{world_size}"

    if args.position_embedding == 'fope':
        cfg.model.fope = True
        cfg.model.rope = False
    elif args.position_embedding == 'rope':
        cfg.model.fope = False
        cfg.model.rope = True
        if args.use_scaled_rope:
            cfg.model.use_scaled_rope = True
            cfg.model.use_scaledrotaryembedding = False
            cfg.model.sin_lambda = args.sin_lambda
            cfg.model.cos_sigma = args.cos_sigma
        elif args.use_scaledrotaryembedding:
            cfg.model.use_scaledrotaryembedding = True
            cfg.model.use_scaled_rope = False
            if args.sigmas is not None:
                assert len(args.sigmas) == cfg.model.n_heads, f"sigmas length {len(args.sigmas)} != n_heads {cfg.model.n_heads}"
                cfg.model.scaled_rope_sigmas = args.sigmas
            else:
                cfg.model.scaled_rope_sigma = args.scaled_rope_sigma
            if args.decay_func is not None:
                cfg.model.decay_func = args.decay_func

    cfg.model.init_device = f"cuda:{local_rank}"

    # Run experiment
    run_experiment(cfg, args.max_steps, args.save_interval, args.streaming)

    cleanup_ddp()


if __name__ == "__main__":
    main()