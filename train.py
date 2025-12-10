"""
modify based on the train_single_model / 50W_train_1sigma.py / 50W_train_multisigma.py
add train_size / val_size / max_sequence_length varibale

modfiy based on train.py
add train_max_sequence_length / val_max_sequence_length
add eval_interval, change save_interval to only save model in the finishing time
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import islice
import argparse
import os
import math
import json
from datetime import datetime
import random
import numpy as np

# Import necessary components from OLMo and Hugging Face datasets
from OLMo.olmo.config import TrainConfig
from OLMo.olmo.model import OLMo
from OLMo.olmo.optim import build_optimizer, build_scheduler
from OLMo.olmo.tokenizer import Tokenizer
from datasets import load_dataset, interleave_datasets, load_from_disk



def run_experiment(cfg: TrainConfig, max_steps: int, log_interval: int, eval_interval : int, save_interval: int, device: torch.device, results_filename: str, use_streaming: bool):
    """
    Runs a single training and evaluation experiment for a given configuration.
    """
    print("\n" + "="*80)
    print(f"Starting Experiment for: {cfg.run_name}")
    print(f"Global Batch Size: {cfg.global_train_batch_size}, Micro Batch Size: {cfg.device_train_microbatch_size}")
    print(f"Using Position Embedding: {'RoPE' if cfg.model.rope else 'FoPE'}")
    if cfg.model.rope:
        print(f"use_scaled_rope1: {cfg.model.use_scaled_rope1}, use_scaled_rope2: {cfg.model.use_scaled_rope2}, sin weight: {cfg.model.sin_lambda}, cos weight: {cfg.model.cos_sigma}, scaled_rope_sigma: {cfg.model.scaled_rope_sigma}, decay_func: {cfg.model.decay_func}")
    print(f"Dataset Streaming: {use_streaming}")
    print("="*80)



    # --- Model and Tokenizer Initialization ---
    model = OLMo(cfg.model)
    model.to(device)
    # export HF_HUB_OFFLINE=1, tokenizer load offline
    try:
        tokenizer = Tokenizer.from_pretrained(
            "allenai/olmo-1b",
            eos_token_id=cfg.model.eos_token_id,
            pad_token_id=cfg.model.pad_token_id
        )
        print("Loaded tokenizer from Hugging Face: allenai/olmo-1b")
    except Exception as e:
        print(f"Failed to load HF tokenizer: {e}")
        print("You may need to download tokenizer.json manually.")
        raise

    # --- Data Loading (using the same strategy as scan_sigma_aligned.py) ---
    # dataset_names = ['allenai/c4']
    # hf_train_datasets = [load_dataset(name, 'en', streaming=use_streaming, split='train[:50000]', trust_remote_code=True) for name in dataset_names]
    # eval_dataset = load_dataset('allenai/c4', 'en', streaming=use_streaming, split='validation[:5000]', trust_remote_code=True)

    # Loaded from local disk, completely offline
    # dataset
    # hf_train_datasets = [load_from_disk("/data/qijunrong/03-proj/PE/c4_train_shards")]
    # eval_dataset = load_from_disk("/data/qijunrong/03-proj/PE/c4_val_shards")

    # load full dataset
    train_full = load_from_disk("/data/qijunrong/03-proj/PE/c4_30M_train")
    val_full = load_from_disk("/data/qijunrong/03-proj/PE/c4_30M_validation")


    train_size = min(cfg.train_size, len(train_full))
    val_size = min(cfg.val_size, len(val_full))
    # train_size = 100000  # 50000 100000
    # val_size = 10000

    hf_train_datasets = [train_full.select(range(train_size))]
    eval_dataset = val_full.select(range(val_size))

    train_chunk_size = cfg.model.max_sequence_length

    train_chunk_size = cfg.model.max_sequence_length
    val_chunk_size = getattr(cfg, 'val_max_sequence_length', train_chunk_size)

    def make_chunk_fn(chunk_size):
        def tokenize_and_chunk(examples):
            all_token_ids = []
            for text in examples.get("text", []):
                if text:
                    all_token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
            total_tokens = (len(all_token_ids) // chunk_size) * chunk_size
            return {"input_ids": [all_token_ids[i:i+chunk_size] for i in range(0, total_tokens, chunk_size)]}
        return tokenize_and_chunk

    tokenize_train = make_chunk_fn(train_chunk_size)
    tokenize_val = make_chunk_fn(val_chunk_size)

    processed_train_datasets = [
        ds.map(tokenize_train, batched=True, batch_size=1000, remove_columns=ds.column_names)
        for ds in hf_train_datasets
    ]
    interleaved_train_dataset = interleave_datasets(processed_train_datasets)
    processed_eval_dataset = eval_dataset.map(tokenize_val, batched=True, batch_size=1000, remove_columns=eval_dataset.column_names)

    def collate_fn(batch):
        input_ids_list = [item['input_ids'] for item in batch]
        return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long)}

    train_loader = DataLoader(interleaved_train_dataset, batch_size=cfg.device_train_microbatch_size, collate_fn=collate_fn)
    eval_loader = DataLoader(processed_eval_dataset, batch_size=cfg.device_train_microbatch_size, collate_fn=collate_fn)

    # --- Optimizer and Scheduler ---
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg)
    max_grad_norm = getattr(cfg.optimizer, 'max_grad_norm', 1.0)    # 

    # --- Training & Evaluation Loop ---
    results = {}
    save_dir = f"/data/qijunrong/03-proj/PE/checkpoints/{cfg.run_name}"
    os.makedirs(save_dir, exist_ok=True)

    # --- evaluate and log ---
    def _evaluate_and_log(step):
        perplexity = evaluate_model(model, eval_loader, cfg, device, max_eval_steps=200)
        results[step] = perplexity
        print(f"Step [{step}], Perplexity: {perplexity:.4f}")
        # Save results immediately
        with open(results_filename, "w") as f:
            json.dump({cfg.run_name: results}, f, indent=4)
        print(f"Intermediate results saved to {results_filename}")

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

        if (step + 1) % log_interval == 0:
            print(f"Step [{step+1}/{max_steps}], Loss: {loss.item():.4f}")

        if (step + 1) % eval_interval == 0:
            _evaluate_and_log(step + 1)

        if (step + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # if (step + 1) % save_interval == 0:

        # --> if eval_interval
        #    perplexity = evaluate_model(model, eval_loader, cfg, device, max_eval_steps=200)
        #    results[step + 1] = perplexity
        #    print(f"Step [{step+1}], Perplexity: {perplexity:.4f}")
            
        # --> if save_interval
        #    checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step+1}.pt")
        #    torch.save(model.state_dict(), checkpoint_path)
        #    print(f"Checkpoint saved to {checkpoint_path}")

        # --> _evaluate_and_log function
            # Save results incrementally
        #    with open(results_filename, "w") as f:
        #        json.dump({cfg.run_name: results}, f, indent=4)
        #    print(f"Intermediate results saved to {results_filename}")
    
    if max_steps % eval_interval != 0:
        _evaluate_and_log(max_steps)

    # final checkpoint
    final_ckpt_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Final model saved to {final_ckpt_path}")

    print(f"\nTraining finished for {cfg.run_name}. Final results saved to {results_filename}")
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
    
    if total_tokens == 0: return float('inf')
    average_loss = total_loss / total_tokens
    return math.exp(average_loss)

def main():
    # --- Set up argument parser ---
    parser = argparse.ArgumentParser(description="Train a single OLMo model with a given configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file (e.g., ./configs/olmo-60m-simplified.yaml)")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps for the experiment.")
    parser.add_argument("--save_interval", type=int, default=10000, help="Interval for saving checkpoints and evaluating the model.")
    parser.add_argument("--train_max_sequence_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--val_max_sequence_length", type=int, default=None, help="Maximum sequence length for validation (default: same as train)")

    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval.")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Interval (in steps) to evaluate and log perplexity.")
    parser.add_argument("--seed", type=int, default=6198, help="Random seed.")
    parser.add_argument("--train_size", type=int, default=1000, help="Number of training samples to use (e.g., 5_000_000 for 5M)")
    parser.add_argument("--val_size", type=int, default=1000, help="Number of validation samples to use")

    # --- Arguments to override config file settings ---
    parser.add_argument("--position_embedding", type=str, choices=['rope', 'fope'], default="rope", help="Override the position embedding method.")
    parser.add_argument("--use_scaled_rope2", action="store_true", help="Enable two weights Scaled RoPE.")
    parser.add_argument("--sin_lambda", type=float, default=None, help="Lambda for sin scaling in Scaled RoPE.")
    parser.add_argument("--cos_sigma", type=float, default=None, help="Sigma for cos scaling in Scaled RoPE.")

    # for shujun's scaled rotary embedding
    parser.add_argument("--use_scaled_rope1", action="store_true", help="Enable one weight Scaled RoPE.")
    parser.add_argument("--scaled_rope_sigma",type=float,default=None,help = "Place field size for scaledRoPE")
    parser.add_argument("--sigmas", nargs='+', type=float, default=None, help="List of sigma values to scan.")
    parser.add_argument("--decay_func", type=str, choices=['exp', 'gaussian','power'], default=None, help="Distance decay function for scaledRoPE")

    parser.add_argument("--yarn_enabled", action="store_true", help="Enable YaRN for context window extension.")
    parser.add_argument("--yarn_max_position_embeddings", type=int, default=4096, help="Original training context length (L_base).")
    parser.add_argument("--yarn_target_max_position_embeddings", type=int, default=None, help="Target context length for fine-tuning (L_target). If not set, inferred from max_sequence_length.")
    parser.add_argument("--yarn_beta_fast", type=float, default=32.0, help="β in YaRN ramp function (high-frequency cutoff).")
    parser.add_argument("--yarn_beta_slow", type=float, default=1.0, help="α in YaRN ramp function (low-frequency start).")
    parser.add_argument("--yarn_dynamic_scaling", action="store_true", help="Use dynamic scaling at inference time (no fine-tuning needed).")
    
    # parser.add_argument("--fope_sigma", type=float, default=None, help="Set the sigma value for FoPE frequency variance.")
    # parser.add_argument("--fope_d", type=int, default=None, help="Set the number of frequencies (D) for FoPE.")
    parser.add_argument("--run_name", type=str, default="olmo-60m-train", help="Custom name for the run. Overrides the name in the config.")
    parser.add_argument("--batch_size", type=int, default=32, help="Global training batch size. Overrides the config file setting for all experiments.")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Device micro batch size. If not set, defaults to the global batch size.")
    parser.add_argument(
        "--streaming",
        action='store_true',
        help="以流式模式加载数据集。如果包含此开关，则 streaming=True，否则为 False。"
    )
    args = parser.parse_args()

    # --- Remove randomness for reproducibility ---
    seed = args.seed
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
    
    if args.train_size:
        cfg.train_size = args.train_size
    if args.val_size:
        cfg.val_size = args.val_size

    if args.train_max_sequence_length:
        cfg.model.max_sequence_length = args.train_max_sequence_length
    # if args.val_max_sequence_length:
    #    cfg.model.val_max_sequence_length = args.val_max_sequence_length

    # evaluate length extension
    cfg.val_max_sequence_length = args.val_max_sequence_length if args.val_max_sequence_length is not None else args.train_max_sequence_length

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
        if args.use_scaled_rope2:       # 使用2个权重同时缩放
            cfg.model.use_scaled_rope2 = True
            cfg.model.use_scaled_rope1 = False  # 注意
            cfg.model.sin_lambda = args.sin_lambda
            cfg.model.cos_sigma = args.cos_sigma
        elif args.use_scaled_rope1:        # shujun zhou，使用1个权重放缩
            cfg.model.use_scaled_rope1 = True
            cfg.model.use_scaled_rope2 = False       # 注意
            if args.sigmas is not None:
                assert len(args.sigmas) == cfg.model.n_heads, f"scaled rope sigmas should have the same length as n_heads: {cfg.model.n_heads}" 
                cfg.model.scaled_rope_sigmas = args.sigmas
            else:
                cfg.model.scaled_rope_sigma = args.scaled_rope_sigma
            if args.decay_func is not None:
                cfg.model.decay_func = args.decay_func
        else:
            cfg.model.use_scaled_rope1 = False
            cfg.model.use_scaled_rope2=False
            cfg.model.sin_lambda = None
            cfg.model.cos_sigma = None

    if args.yarn_enabled:
        cfg.model.yarn_enabled = True
        cfg.model.yarn_max_position_embeddings = args.yarn_max_position_embeddings
        if args.yarn_target_max_position_embeddings is not None:
            cfg.model.yarn_target_max_position_embeddings = args.yarn_target_max_position_embeddings
        else:
            # Default: use current max_sequence_length as target
            cfg.model.yarn_target_max_position_embeddings = cfg.model.max_sequence_length
        cfg.model.yarn_beta_fast = args.yarn_beta_fast
        cfg.model.yarn_beta_slow = args.yarn_beta_slow
        cfg.model.yarn_dynamic_scaling = args.yarn_dynamic_scaling
    else:
        cfg.model.yarn_enabled = False


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.model.init_device = str(device)
    
    # --- Prepare results file ---
    results_dir = "training_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_filename = os.path.join(results_dir, f"{cfg.run_name}_{timestamp}.json")

    # --- Run the single experiment ---
    run_experiment(cfg, args.max_steps, args.log_interval, args.eval_interval, args.save_interval, device, results_filename, args.streaming)

if __name__ == "__main__":
    main()
