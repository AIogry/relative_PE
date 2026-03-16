import sys
from types import ModuleType
import torch
import torch.nn as nn
import math
import os
import argparse

# =============================================================================
# [CRITICAL FIX] Mock 缺失的 olmo_data 模块
# 原因: OLMo 源码依赖 olmo_data 包来解析数据路径，但我们在本地环境不需要它。
# 方案: 在导入 OLMo 之前，先在内存中创建一个假的 olmo_data 模块欺骗 Python。
# =============================================================================
try:
    import olmo_data
except ImportError:
    # 1. 创建假的 olmo_data 模块
    dummy_pkg = ModuleType("olmo_data")
    sys.modules["olmo_data"] = dummy_pkg
    
    # 2. 创建假的 olmo_data.data 子模块
    dummy_data = ModuleType("olmo_data.data")
    sys.modules["olmo_data.data"] = dummy_data
    
    # 3. 模拟 get_data_path 函数 (直接返回原路径即可)
    def dummy_get_data_path(path):
        return str(path)
    
    dummy_data.get_data_path = dummy_get_data_path
    
    print(">>> [System] Successfully mocked 'olmo_data' module to bypass ImportError.")
# =============================================================================

from transformers import AutoTokenizer
# 现在可以安全导入 OLMo 了
from OLMo.olmo.config import ModelConfig
from OLMo.olmo.model import OLMo, OLMoSequentialBlock

# === 配置 (请检查这里的路径是否正确) ===
# 您的 Checkpoint 路径 (确保文件存在)
CHECKPOINT_PATH = "/data/qijunrong/03-proj/PE/checkpoints_variable_len/baseline_nope_20M_L512_20260127_005110/model.pt"
TOKENIZER_PATH = "/data/qijunrong/03-proj/PE/wikitext/tokenizer"
MODEL_SIZE = "20M"
SEQ_LEN = 512
OUTPUT_FILE = "exp1_attention_data_layers.pt"

# === 1. 定义捕获注意力的 Patch 函数 ===
def patched_scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    """
    替换原版 Flash Attention 的手动实现，用于捕获权重。
    """
    # OLMo 的 q, k, v shape 通常是 [B, T, n_heads, d_head] 或者已经被 transpose 过了
    # 根据 model.py 的 attention 函数:
    # q: [B, n_heads, T, d_head]
    
    # 确保维度符合预期
    if q.dim() == 4:
        B, n_heads, T, d_head = q.shape
        # 1. 手动计算 Attention Scores: (Q @ K^T) / sqrt(d)
        # q: [B, H, T, D], k.transpose: [B, H, D, T] -> scores: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
        
        # 2. 应用 Mask (Causal)
        if is_causal:
            # 创建下三角 Mask
            mask = torch.tril(torch.ones((T, T), device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        
        if attn_mask is not None:
            scores = scores + attn_mask

        # 3. Softmax 获取权重
        attn_weights = torch.softmax(scores, dim=-1)
        
        # === 关键：保存权重 ===
        # 我们利用 self (Block实例) 来暂存数据
        if not hasattr(self, '_captured_list'):
            self._captured_list = []
        # detach 并转到 CPU 以节省显存
        self._captured_list.append(attn_weights.detach().cpu())

        # 4. 计算输出
        output = torch.matmul(attn_weights, v)
        return output
    else:
        # Fallback 如果维度不对 (极少情况)
        return F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)

def run_capture():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载 Tokenizer
    print(f"Loading Tokenizer from {TOKENIZER_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    except:
        from OLMo.olmo.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_pretrained(TOKENIZER_PATH, eos_token_id=50256, pad_token_id=50256)
        
    raw_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    vocab_size = ((raw_vocab_size + 63) // 64) * 64

    # 2. 配置模型
    d_model = 256 if MODEL_SIZE == "20M" else 512
    
    # ⚠️ 注意：这里必须与您训练时的配置完全一致
    # 既然您加载的是 baseline_nope，那么 nope=True, rope=False
    cfg = ModelConfig(
        d_model=d_model, n_heads=8, n_layers=8, mlp_ratio=8,
        max_sequence_length=SEQ_LEN, vocab_size=vocab_size, embedding_size=vocab_size,
        rope=False, fope=False, alibi=False, yarn_enabled=False, 
        nope=True, # <--- 对应 NoPE Checkpoint
        flash_attention=False # 强制关闭 Flash Attention 以便我们 Patch
    )
    
    print("Initializing Model...")
    model = OLMo(cfg).to(device)
    
    # 3. 加载权重
    print(f"Loading Checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint file not found: {CHECKPOINT_PATH}")
        return

    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Checkpoint loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
        
    model.eval()

    # 4. === Monkey Patch ===
    print("Patching Attention mechanism...")
    # 替换类方法，这样所有实例都会受影响
    OLMoSequentialBlock._scaled_dot_product_attention = patched_scaled_dot_product_attention

    # 5. 构造 Associative Recall 输入
    # 格式: A B C ... A -> 预测 B
    # 这里的 ID 是随便选的，只要在该 Tokenizer 的词表范围内即可 (WikiText vocab ~50k)
    kv_pairs = [(100, 200), (300, 400), (500, 600), (700, 800)] 
    input_ids_list = []
    for k, v in kv_pairs:
        input_ids_list.extend([k, v])
    # 重复第一个 Key 作为 Query (Trigger)
    input_ids_list.append(kv_pairs[0][0]) 
    
    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
    print(f"Input Sequence Length: {input_ids.shape[1]}")

    # 6. 前向传播
    print("Running forward pass...")
    with torch.no_grad():
        # 清空之前的捕获
        for block in model.transformer.blocks:
            if hasattr(block, '_captured_list'):
                block._captured_list = []
                
        outputs = model(input_ids)
    
    # 7. 收集所有层的 Attention
    all_layers_attn = []
    captured_count = 0
    
    print("\nCollecting attention maps...")
    for i, block in enumerate(model.transformer.blocks):
        # 检查是否有捕获到数据
        if hasattr(block, '_captured_list') and len(block._captured_list) > 0:
            # 取最后一次调用的结果 (shape: [B, n_heads, T, T])
            attn = block._captured_list[-1]
            all_layers_attn.append(attn)
            print(f"  - Layer {i}: {attn.shape}")
            captured_count += 1
        else:
            print(f"  - Layer {i}: No attention captured (Warning)")

    if captured_count == 0:
        print("❌ No attention maps captured! Check if the model is actually using the patched method.")
        return

    # 堆叠: [Layers, B, Heads, T, T]
    stacked_attn = torch.stack(all_layers_attn)
    print(f"Final Stacked Shape: {stacked_attn.shape}")
    
    # 8. 保存
    save_data = {
        "input_ids": input_ids.cpu(),
        "results": {
            "NoPE": stacked_attn 
        }
    }
    torch.save(save_data, OUTPUT_FILE)
    print(f"\n✅ Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_capture()