"""
SST-2 (Stanford Sentiment Treebank) Fine-tuning Script for OLMo with HIPE
支持 LoRA 微调和 Few-shot 学习

Usage:
    # Full fine-tuning on SST-2 with LoRA (推荐)
    python finetune_sst2.py \
        --base_model_path checkpoints/c4_300M/model.pt \
        --output_dir results/sst2_hipe \
        --use_scaled_rope --learnable_sigma \
        --use_lora --lora_rank 8 --lora_alpha 32 \
        --few_shot -1

    # Few-shot (100 samples)
    python finetune_sst2.py \
        --base_model_path checkpoints/c4_300M/model.pt \
        --output_dir results/sst2_hipe_100 \
        --use_scaled_rope --learnable_sigma \
        --use_lora --lora_rank 8 \
        --few_shot 100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import argparse
import os
import math
import random
import numpy as np
import wandb
import subprocess
import sys
import json
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Imports from OLMo
from OLMo.olmo.config import ModelConfig
from OLMo.olmo.model import OLMo


def set_seed(seed: int):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_info() -> Dict[str, str]:
    """获取当前代码仓库的Git信息"""
    git_info = {}
    try:
        git_info["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        git_info["short_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        git_info["is_dirty"] = len(git_status) > 0
        git_info["dirty_files"] = git_status if git_info["is_dirty"] else "None"
    except Exception as e:
        git_info["error"] = str(e)
        git_info["commit_hash"] = "unknown"
        git_info["short_commit"] = "unknown"
    return git_info


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层实现
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, lora_alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        # LoRA 可训练参数
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # 初始化: A 用高斯, B 用零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output


class OLMoForSequenceClassification(nn.Module):
    """
    OLMo + 分类头，用于SST-2等文本分类任务
    支持 LoRA 微调
    """
    def __init__(
        self, 
        config: ModelConfig, 
        num_labels: int = 2,
        lora_rank: int = 0,
        lora_alpha: float = 16,
        lora_target_modules: Optional[List[str]] = None,
        freeze_base_model: bool = True,
        freeze_sigma: bool = False,
    ):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.lora_rank = lora_rank
        
        # 加载基础 OLMo 模型
        self.olmo = OLMo(config)
        
        # 分类头
        self.classifier = nn.Linear(config.d_model, num_labels)
        
        # 初始化分类头 - 使用更保守的初始化
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
        
        # 设置参数冻结策略
        self._setup_parameter_freezing(freeze_base_model, freeze_sigma)
        
        # 设置 LoRA
        if lora_rank > 0:
            # OLMo结构: transformer.blocks.{i}.att_proj (QKV合并) 和 attn_out
            target_modules = lora_target_modules or ["att_proj", "attn_out", "ff_proj", "ff_out"]
            self._setup_lora(lora_rank, lora_alpha, target_modules)
    
    def _setup_parameter_freezing(self, freeze_base_model: bool, freeze_sigma: bool):
        """设置参数冻结策略 - 基础模型参数默认冻结"""
        if freeze_base_model:
            for name, param in self.olmo.named_parameters():
                param.requires_grad = False
        
        # 分类头始终可训练
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def setup_learnable_sigma(self, freeze_sigma: bool = False):
        """
        在加载预训练权重后调用，解冻实际存在的sigma参数
        注意：rope_scaling_threshold 已在创建模型时正确设置
        """
        if freeze_sigma:
            return
        
        # 解冻实际存在的sigma参数
        sigma_count = 0
        for name, param in self.olmo.named_parameters():
            if "sigma" in name and param is not None:
                param.requires_grad = True
                sigma_count += 1
                print(f"[Unfrozen] Sigma param: {name}")
        
        if sigma_count > 0:
            print(f"[Sigma] Total {sigma_count} sigma parameters unfrozen")
        else:
            print("[Sigma] No sigma parameters found in model (using fixed RoPE)")
    
    def _setup_lora(self, rank: int, alpha: float, target_modules: List[str]):
        """为指定模块添加 LoRA"""
        self.lora_layers = nn.ModuleDict()
        lora_count = 0
        
        for name, module in self.olmo.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    lora = LoRALayer(
                        module.in_features, 
                        module.out_features, 
                        rank=rank, 
                        lora_alpha=alpha
                    )
                    self.lora_layers[name.replace(".", "_")] = lora
                    for param in module.parameters():
                        param.requires_grad = False
                    lora_count += 1
        
        print(f"[LoRA] Added to {lora_count} layers with rank={rank}, alpha={alpha}")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        outputs = self.olmo(input_ids=input_ids, output_hidden_states=True)
        
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            raise ValueError("hidden_states is None.")
        
        # Mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            mean_hidden = sum_hidden / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        else:
            mean_hidden = hidden_states.mean(dim=1)
        
        logits = self.classifier(mean_hidden)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return logits, loss
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """获取可训练参数的统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        lora_params = 0
        if hasattr(self, 'lora_layers'):
            lora_params = sum(p.numel() for p in self.lora_layers.parameters())
        sigma_params = sum(p.numel() for n, p in self.named_parameters() if "sigma" in n and p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
            "trainable_pct": 100 * trainable_params / total_params,
            "classifier": classifier_params,
            "lora": lora_params,
            "sigma": sigma_params,
        }


def load_sst2_dataset(tokenizer, max_length: int = 128, few_shot: int = -1, local_path: str = None):
    """
    加载 SST-2 数据集
    
    SST-2是二分类情感分析任务：
    - 0: Negative (负面)
    - 1: Positive (正面)
    """
    import os
    
    # 加载数据集
    if local_path and os.path.exists(local_path):
        print(f"Loading SST-2 from local path: {local_path}")
        dataset = {
            'train': load_from_disk(os.path.join(local_path, 'train')),
            'validation': load_from_disk(os.path.join(local_path, 'validation')),
            'test': load_from_disk(os.path.join(local_path, 'test'))
        }
        print(f"  Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    else:
        if local_path:
            print(f"Local path not found: {local_path}, downloading from HuggingFace...")
        else:
            print("Loading SST-2 from HuggingFace...")
        dataset = load_dataset("glue", "sst2")
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[Tokenizer] Set pad_token to eos_token: {tokenizer.pad_token}")
    
    def preprocess_function(examples):
        """预处理函数"""
        result = tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors=None,
        )
        result["labels"] = examples["label"]
        return result
    
    # 应用预处理
    print(">>> Preprocessing datasets...")
    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train"
    )
    val_dataset = dataset["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        desc="Tokenizing validation"
    )
    test_dataset = dataset["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["test"].column_names,
        desc="Tokenizing test"
    )
    
    # 转换为 PyTorch 格式
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Few-shot 采样 - 分层采样
    if few_shot > 0 and few_shot < len(train_dataset):
        labels = np.array(train_dataset["labels"])
        indices_0 = np.where(labels == 0)[0]  # Negative
        indices_1 = np.where(labels == 1)[0]  # Positive
        
        # SST-2分布: 约50/50
        n_1 = int(few_shot * len(indices_1) / len(labels))
        n_0 = few_shot - n_1
        
        n_0 = min(n_0, len(indices_0))
        n_1 = min(n_1, len(indices_1))
        
        if n_0 + n_1 < few_shot:
            remaining = few_shot - n_0 - n_1
            if n_0 < len(indices_0):
                n_0 = min(n_0 + remaining, len(indices_0))
            else:
                n_1 = min(n_1 + remaining, len(indices_1))
        
        sampled_indices = np.concatenate([
            np.random.choice(indices_0, n_0, replace=False),
            np.random.choice(indices_1, n_1, replace=False)
        ])
        np.random.shuffle(sampled_indices)
        
        train_dataset = Subset(train_dataset, sampled_indices)
        print(f"[Few-shot] Sampled {len(sampled_indices)} examples ({n_0} negative, {n_1} positive)")
    
    return train_dataset, val_dataset, test_dataset


def evaluate(model, dataloader, device, use_amp: bool = True) -> Dict[str, float]:
    """
    评估模型性能
    
    SST-2主要指标：Accuracy (因为类别平衡)
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(input_ids, attention_mask, labels)
            else:
                logits, loss = model(input_ids, attention_mask, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # SST-2主要指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    avg_loss = total_loss / len(dataloader)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "loss": avg_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="SST-2 Fine-tuning for OLMo with HIPE")
    
    # 模型参数
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="预训练模型路径 (.pt 文件)")
    parser.add_argument("--model_size", type=str, default="300M", 
                        choices=["20M", "60M", "300M"])
    parser.add_argument("--local_tokenizer_path", type=str, 
                        default="./wikitext/tokenizer",
                        help="分词器路径")
    
    # HIPE 参数
    parser.add_argument("--use_scaled_rope", action="store_true",
                        help="使用 HIPE (ScaledRoPE)")
    parser.add_argument("--sigma", type=float, default=200.0,
                        help="HIPE sigma 初始值")
    parser.add_argument("--learnable_sigma", action="store_true",
                        help="sigma 是否可学习")
    parser.add_argument("--rope_scaling_threshold", type=int, default=-1,
                        help="分层阈值，-1 表示全局使用")
    parser.add_argument("--decay_func", type=str, default="gaussian",
                        choices=["gaussian", "exp", "power", "segmented"],
                        help="衰减函数类型")
    
    # LoRA 参数
    parser.add_argument("--use_lora", action="store_true",
                        help="是否使用 LoRA (强烈推荐)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=float, default=16,
                        help="LoRA alpha 缩放因子")
    parser.add_argument("--lora_target", nargs="+", 
                        default=["att_proj", "attn_out", "ff_proj", "ff_out"],
                        help="LoRA 目标模块 (OLMo使用: att_proj, attn_out, ff_proj, ff_out)")
    
    # 微调策略
    parser.add_argument("--freeze_base", action="store_true", default=True,
                        help="冻结基础模型权重")
    parser.add_argument("--freeze_sigma", action="store_true", default=False,
                        help="冻结 sigma 参数")
    parser.add_argument("--sigma_lr", type=float, default=None,
                        help="sigma 参数的专用学习率")
    
    # 数据参数
    parser.add_argument("--sst2_data_path", type=str, default=None,
                        help="SST-2 数据集本地路径")
    parser.add_argument("--few_shot", type=int, default=-1,
                        help="Few-shot 样本数，-1 表示使用全部数据")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="SST-2通常5个epoch足够")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="学习率")
    parser.add_argument("--classifier_lr", type=float, default=None,
                        help="分类头专用学习率")
    parser.add_argument("--lora_lr", type=float, default=None,
                        help="LoRA专用学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="梯度累积步数")
    
    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="早停耐心值 (按eval次数)")
    parser.add_argument("--early_stopping_delta", type=float, default=0.001,
                        help="早停最小改善阈值")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="每隔多少 step 评估一次 (与eval_interval_samples二选一)")
    parser.add_argument("--eval_interval_samples", type=int, default=None,
                        help="每隔多少样本评估一次 (如1000表示每处理1000个样本评估)")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="每隔多少 step 保存 checkpoint")
    
    # wandb 参数
    parser.add_argument("--wandb_mode", type=str, default="offline",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="HIPE-SST2")
    parser.add_argument("--wandb_dir", type=str, default=None,
                        help="WandB 日志目录")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Git 信息
    git_info = get_git_info()
    
    # 初始化 wandb
    run_name = args.run_name or f"sst2_{args.model_size}_{'hipe' if args.use_scaled_rope else 'rope'}"
    if args.few_shot > 0:
        run_name += f"_shot{args.few_shot}"
    if args.use_lora:
        run_name += f"_lora{args.lora_rank}"
    
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            **vars(args),
            **git_info,
        },
        mode=args.wandb_mode,
        dir=args.wandb_dir if args.wandb_dir else args.output_dir,
    )
    
    # 加载分词器
    print(f"Loading tokenizer from {args.local_tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer_path, local_files_only=True)
    except:
        from OLMo.olmo.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_pretrained(args.local_tokenizer_path)
    
    # 加载 SST-2 数据
    print("Loading SST-2 dataset...")
    train_dataset, val_dataset, test_dataset = load_sst2_dataset(
        tokenizer, 
        max_length=args.max_length,
        few_shot=args.few_shot,
        local_path=args.sst2_data_path
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 模型配置
    print("Building model...")
    if args.model_size == "20M":
        cur_d, cur_h, cur_l, cur_mlp = 256, 8, 8, 8
    elif args.model_size == "60M":
        cur_d, cur_h, cur_l, cur_mlp = 512, 8, 8, 8
    elif args.model_size == "300M":
        cur_d, cur_h, cur_l, cur_mlp = 1024, 16, 16, 8
    
    raw_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    vocab_size = ((raw_vocab_size + 63) // 64) * 64
    
    use_flash_attn = (device.type == "cuda")
    if not use_flash_attn:
        print("⚠️  CUDA not available, disabling FlashAttention")
    
    cfg = ModelConfig(
        d_model=cur_d,
        n_heads=cur_h,
        n_layers=cur_l,
        mlp_ratio=cur_mlp,
        max_sequence_length=args.max_length,
        vocab_size=vocab_size,
        embedding_size=vocab_size,
        init_std=0.02,
        rope=True,
        use_scaled_rope1=args.use_scaled_rope,
        scaled_rope_sigma=args.sigma,
        rope_scaling_threshold=args.rope_scaling_threshold,
        learnable_sigma=args.learnable_sigma,  # 创建时启用，但加载后会删除多余的sigma
        decay_func=args.decay_func,
        flash_attention=use_flash_attn,
        include_bias=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
    )
    
    # 构建模型
    model = OLMoForSequenceClassification(
        config=cfg,
        num_labels=2,
        lora_rank=args.lora_rank if args.use_lora else 0,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target,
        freeze_base_model=args.freeze_base,
        freeze_sigma=args.freeze_sigma,
    )
    
    # 加载预训练权重
    if args.base_model_path and os.path.exists(args.base_model_path):
        print(f"Loading pretrained weights from {args.base_model_path}...")
        try:
            state_dict = torch.load(args.base_model_path, map_location="cpu", weights_only=False)
            model.olmo.load_state_dict(state_dict, strict=False)
            print("✓ Pretrained weights loaded successfully.")
            
            # 关键：加载权重后，根据实际存在的sigma参数设置可学习性
            if args.learnable_sigma and not args.freeze_sigma:
                print(">>> Setting up learnable sigma parameters...")
                model.setup_learnable_sigma(freeze_sigma=False)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load pretrained weights: {e}")
            print("   Continuing with random initialization...")
    
    model = model.to(device)
    
    # 打印可训练参数信息
    param_stats = model.get_trainable_parameters()
    print(f"\nParameter Statistics:")
    print(f"  Total: {param_stats['total']:,}")
    print(f"  Trainable: {param_stats['trainable']:,} ({param_stats['trainable_pct']:.2f}%)")
    print(f"  Frozen: {param_stats['frozen']:,}")
    print(f"  - Classifier: {param_stats['classifier']:,}")
    print(f"  - LoRA: {param_stats['lora']:,}")
    print(f"  - Sigma: {param_stats['sigma']:,}")
    
    wandb.config.update(param_stats, allow_val_change=True)
    
    # 优化器设置
    param_groups = []
    
    # 分类头参数
    classifier_lr = args.classifier_lr or args.lr
    classifier_params = list(model.classifier.parameters())
    if classifier_params:
        param_groups.append({
            "params": classifier_params,
            "lr": classifier_lr,
            "name": "classifier"
        })
        print(f"Classifier learning rate: {classifier_lr}")
    
    # LoRA 参数
    if args.use_lora and hasattr(model, 'lora_layers'):
        lora_lr = args.lora_lr or args.lr
        lora_params = list(model.lora_layers.parameters())
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": lora_lr,
                "name": "lora"
            })
            print(f"LoRA learning rate: {lora_lr}")
    
    # Sigma 参数
    sigma_params = [p for n, p in model.named_parameters() if "sigma" in n and p.requires_grad]
    if sigma_params:
        sigma_lr = args.sigma_lr or args.lr
        param_groups.append({
            "params": sigma_params,
            "lr": sigma_lr,
            "name": "sigma"
        })
        print(f"Sigma learning rate: {sigma_lr}")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # 学习率调度
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\nTraining for {args.num_epochs} epochs ({len(train_loader)} steps/epoch)")
    print(f"Total steps (with grad accum): {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # 训练循环
    global_step = 0
    best_acc = 0.0  # SST-2用accuracy作为主要指标
    best_checkpoint_path = None
    evals_without_improvement = 0
    
    # Sample Efficiency Tracking: 记录达到特定准确率所需的步数和样本数
    # 根据shot设置动态调整目标阈值
    if args.few_shot > 0 and args.few_shot < 1000:
        # 小样本情况下降低目标
        acc_thresholds = [0.60, 0.65, 0.70, 0.75]
    elif args.few_shot >= 1000 and args.few_shot < 5000:
        acc_thresholds = [0.70, 0.75, 0.80]
    elif args.few_shot >= 5000:
        # 大样本 (5000, 10000)
        acc_thresholds = [0.75, 0.80, 0.85]
    else:
        # full数据集：从50%开始记录，更全面地追踪收敛过程
        acc_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    
    sample_efficiency = {
        "thresholds": acc_thresholds,
        "results": {}  # {threshold: {"step": int, "samples": int, "epoch": float}}
    }
    samples_processed = 0  # 已处理的样本数
    last_eval_samples = 0  # 上次评估时的样本数（用于基于样本的评估间隔）
    
    log_file = open(os.path.join(args.output_dir, "training_log.txt"), "w")
    log_file.write("step,epoch,train_loss,eval_acc,eval_f1,learning_rate,samples_processed\n")
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        accumulated_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 统计样本数
            batch_size_actual = input_ids.size(0)
            samples_processed += batch_size_actual
            
            # 前向
            if use_flash_attn:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(input_ids, attention_mask, labels)
            else:
                logits, loss = model(input_ids, attention_mask, labels)
            
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                epoch_loss += accumulated_loss
                
                pbar.set_postfix({
                    "loss": f"{accumulated_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                if global_step % 10 == 0:
                    wandb.log({
                        "train/loss": accumulated_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/step": global_step,
                        "train/epoch": epoch + 1,
                    })
                
                accumulated_loss = 0.0
                
                # 评估条件：基于step或基于样本数
                should_eval = False
                if args.eval_interval_samples is not None:
                    # 基于样本数的评估
                    if samples_processed - last_eval_samples >= args.eval_interval_samples:
                        should_eval = True
                else:
                    # 基于step的评估
                    if global_step % args.eval_interval == 0:
                        should_eval = True
                
                # 评估
                if should_eval:
                    eval_results = evaluate(model, val_loader, device, use_amp=use_flash_attn)
                    
                    print(f"\n[Step {global_step}] Eval: Acc={eval_results['accuracy']:.4f}, F1={eval_results['f1']:.4f}")
                    
                    wandb.log({
                        "eval/accuracy": eval_results["accuracy"],
                        "eval/f1": eval_results["f1"],
                        "eval/precision": eval_results["precision"],
                        "eval/recall": eval_results["recall"],
                        "eval/loss": eval_results["loss"],
                        "eval/step": global_step,
                    })
                    
                    log_file.write(f"{global_step},{epoch+1},{accumulated_loss},{eval_results['accuracy']:.4f},{eval_results['f1']:.4f},{scheduler.get_last_lr()[0]:.6f},{samples_processed}\n")
                    log_file.flush()
                    
                    # Sample Efficiency Tracking: 检查是否达到新的阈值
                    current_acc = eval_results["accuracy"]
                    for threshold in acc_thresholds:
                        thresh_str = f"{threshold:.2f}"
                        if thresh_str not in sample_efficiency["results"]:
                            if current_acc >= threshold:
                                sample_efficiency["results"][thresh_str] = {
                                    "step": global_step,
                                    "samples": samples_processed,
                                    "epoch": epoch + 1 + batch_idx / len(train_loader),
                                    "accuracy": current_acc
                                }
                                print(f"  -> [Sample Efficiency] Reached {threshold*100:.0f}% accuracy at step {global_step} ({samples_processed} samples)")
                    
                    # 保存最佳模型 (基于accuracy)
                    if current_acc > best_acc + args.early_stopping_delta:
                        best_acc = current_acc
                        best_checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
                        torch.save({
                            "step": global_step,
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "accuracy": best_acc,
                            "f1": eval_results["f1"],
                            "args": vars(args),
                        }, best_checkpoint_path)
                        print(f"  -> New best model saved! (Acc: {best_acc:.4f})")
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1
                        print(f"  -> No improvement ({evals_without_improvement}/{args.early_stopping_patience})")
                    
                    # 更新上次评估的样本数（用于基于样本的评估间隔）
                    if args.eval_interval_samples is not None:
                        last_eval_samples = samples_processed
                    
                    # Early stopping
                    if evals_without_improvement >= args.early_stopping_patience:
                        print(f"\n⚠️  Early stopping triggered")
                        break
                    
                    model.train()
        
        avg_epoch_loss = epoch_loss / max(1, len(train_loader) // args.gradient_accumulation_steps)
        print(f"Epoch {epoch+1} finished. Average loss: {avg_epoch_loss:.4f}")
        
        if evals_without_improvement >= args.early_stopping_patience:
            break
    
    # 最终评估
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from step {checkpoint['step']} (epoch {checkpoint['epoch']})")
    
    final_results = evaluate(model, val_loader, device, use_amp=use_flash_attn)
    print(f"Best Accuracy: {final_results['accuracy']:.4f}")
    print(f"Best F1: {final_results['f1']:.4f}")
    print(f"Precision: {final_results['precision']:.4f}")
    print(f"Recall: {final_results['recall']:.4f}")
    
    # 保存最终结果
    # 补充未达到的阈值信息
    for threshold in acc_thresholds:
        thresh_str = f"{threshold:.2f}"
        if thresh_str not in sample_efficiency["results"]:
            sample_efficiency["results"][thresh_str] = {
                "step": None,
                "samples": None,
                "epoch": None,
                "accuracy": None,
                "status": "not_reached"
            }
    
    with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
        json.dump({
            "best_accuracy": final_results['accuracy'],
            "best_f1": final_results['f1'],
            "best_precision": final_results['precision'],
            "best_recall": final_results['recall'],
            "best_val_loss": final_results['loss'],
            "trainable_params": param_stats['trainable'],
            "trainable_pct": param_stats['trainable_pct'],
            "sample_efficiency": sample_efficiency,  # 新增：样本效率指标
            "total_samples_processed": samples_processed,  # 总处理样本数
            "args": vars(args),
        }, f, indent=2)
    
    # 打印 Sample Efficiency 摘要
    print("\n" + "="*50)
    print("Sample Efficiency Summary")
    print("="*50)
    print(f"{'Target Acc':<12} {'Step':<8} {'Samples':<10} {'Epoch':<8}")
    print("-"*50)
    for threshold in acc_thresholds:
        thresh_str = f"{threshold:.2f}"
        result = sample_efficiency["results"][thresh_str]
        if result.get("step") is not None:
            print(f"{threshold*100:>6.0f}%      {result['step']:<8} {result['samples']:<10} {result['epoch']:.2f}")
        else:
            print(f"{threshold*100:>6f}%      {'N/A':<8} {'N/A':<10} {'N/A'}")
    print("="*50)
    
    wandb.log({
        "final/accuracy": final_results["accuracy"],
        "final/f1": final_results["f1"],
        "final/precision": final_results["precision"],
        "final/recall": final_results["recall"],
        "final/best_accuracy": best_acc,
    })
    
    log_file.close()
    wandb.finish()
    
    print(f"\n✓ Results saved to {args.output_dir}")
    
    return final_results['accuracy']


if __name__ == "__main__":
    acc = main()
    sys.exit(0)
