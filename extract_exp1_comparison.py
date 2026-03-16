import torch
import os
import sys
import math
import torch.nn.functional as F
from types import ModuleType

# =============================================================================
# 1. Mock olmo_data
# =============================================================================
def dummy_func(*args, **kwargs): return str(args[0]) if args else None
if "olmo_data" not in sys.modules:
    d = ModuleType("olmo_data"); d.get_data_path = dummy_func; d.is_data_file = lambda x: True
    sys.modules["olmo_data"] = d
    sys.modules["olmo_data.data"] = d
    sys.modules["olmo_data"].data = d

# =============================================================================
# 2. 导入 model2
# =============================================================================
# 自动修复路径
current_dir = os.getcwd()
olmo_path = os.path.join(current_dir, "OLMo")
if olmo_path not in sys.path: sys.path.append(olmo_path)

try:
    from olmo.model2 import OLMo, OLMoSequentialBlock
    from olmo.config import ModelConfig
    print(">>> 成功加载 OLMo (model2)")
except ImportError:
    sys.exit("❌ 错误: 找不到 model2.py")

# =============================================================================
# 3. 强制 Patch 逻辑 (关键!)
# =============================================================================
# 全局变量用来存权重
CAPTURED_ATTN = {} # { "RoPE": [Layer0, Layer1...], "Bio": [...] }
CURRENT_MODEL_NAME = ""

def forced_attention_patch(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    # 1. 计算 Scores
    B, n_heads, T, d_head = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
    
    if is_causal:
        mask = torch.tril(torch.ones((T, T), device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    if attn_mask is not None:
        scores = scores + attn_mask

    # 2. Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 3. 强制保存 (直接存到全局列表)
    # 转到 CPU 以防爆显存
    w = attn_weights.detach().cpu()
    
    if CURRENT_MODEL_NAME not in CAPTURED_ATTN:
        CAPTURED_ATTN[CURRENT_MODEL_NAME] = []
        
    CAPTURED_ATTN[CURRENT_MODEL_NAME].append(w)

    # 4. Output
    output = torch.matmul(attn_weights, v)
    return output

# 应用 Patch
print(">>> 应用强制 Attention Patch...")
OLMoSequentialBlock._scaled_dot_product_attention = forced_attention_patch


# =============================================================================
# 4. 提取主逻辑
# =============================================================================
PATHS = {
    "RoPE": "./checkpoints_viz/baseline/model2.pt",
    "Bio-Gradient": "./checkpoints_viz/bio_gradient/model2.pt"
}

SEQ_LEN = 512 # 确保这个长度和画图脚本一致
VOCAB_SIZE = 50304

def extract():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")

    # 构造序列
    seq = []
    for i in range((SEQ_LEN - 2) // 2):
        k = (i % 100) + 100
        v = (i % 100) + 200
        seq.extend([k, v])
    while len(seq) < SEQ_LEN - 1: seq.append(0)
    seq.append(seq[0]) 
    
    input_ids = torch.tensor([seq[:SEQ_LEN]], dtype=torch.long).to(device)

    final_results = {}

    for name, path in PATHS.items():
        print(f"\n>>> 处理: {name}")
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            continue
            
        # 设置全局标记，告诉 Patch 函数当前是谁
        global CURRENT_MODEL_NAME
        CURRENT_MODEL_NAME = name
        CAPTURED_ATTN[name] = [] # 清空列表
        
        is_bio = (name == "Bio-Gradient")
        
        # 配置
        cfg = ModelConfig(
            d_model=256,      # <--- 改为 512 (匹配 Checkpoint)
            n_heads=8, 
            n_layers=8, 
            mlp_ratio=8,      # <--- 改为 4 (因为 512 * 4 = 2048)
            max_sequence_length=SEQ_LEN, 
            vocab_size=VOCAB_SIZE, 
            embedding_size=VOCAB_SIZE,
            rope=True, 
            use_scaled_rope1=is_bio,
            scaled_rope_sigma=100.0 if is_bio else 1.0,
            rope_scaling_threshold=2 if is_bio else -1,
            flash_attention=False 
        )
        
        try:
            model = OLMo(cfg)
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            print("   ✅ 模型加载成功")
            
            with torch.no_grad():
                model(input_ids)
            
            # 检查抓到了多少
            captured = CAPTURED_ATTN[name]
            # 注意: Patch 会被调用多次 (每层一次)，所以列表里是 [Layer0, Layer1, ...]
            # 如果是多头并行，captured 长度应该是 n_layers
            
            # 因为 OLMoSequentialBlock 是串行的，我们期望列表长度 = n_layers
            # 如果长度 > n_layers，说明可能有些不是 block 调用的，或者 batch>1
            # 这里我们只取最后 n_layers 个 (假设一次 forward)
            
            if len(captured) >= 8:
                # 截取最后8个 (对应8层)
                layers_data = captured[-8:] 
                stacked = torch.stack(layers_data) # [8, B, H, T, T]
                final_results[name] = stacked
                print(f"   ✅ 成功捕获! Shape: {stacked.shape}")
            else:
                print(f"   ❌ 捕获数量异常: {len(captured)} (预期 >= 8)")

        except Exception as e:
            print(f"   ❌ 运行异常: {e}")
            import traceback
            traceback.print_exc()

    # 保存
    if len(final_results) > 0:
        save_data = {"results": final_results, "input_ids": input_ids.cpu()}
        torch.save(save_data, "exp1_comparison_data2.pt")
        print(f"\n>>> 🎉 提取完成! 文件大小: {os.path.getsize('exp1_comparison_data2.pt') / 1024 / 1024:.2f} MB")
        print(f"    Keys: {list(final_results.keys())}")
    else:
        print("\n❌ 依然没有提取到数据")

if __name__ == "__main__":
    extract()