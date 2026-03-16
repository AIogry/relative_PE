import torch
import os
import sys

# 🔴 引用 model2
try:
    from OLMo.olmo.model2 import OLMo
    from OLMo.olmo.config import ModelConfig
    print(">>> 成功加载 model2")
except ImportError:
    sys.exit("❌ 错误: 找不到 model2.py")

# === 配置路径 ===
PATHS = {
    "Baseline": "./checkpoints_viz/baseline/model.pt",
    "Bio-Gradient": "./checkpoints_viz/bio_gradient/model.pt"
}

SEQ_LEN = 64
VOCAB_SIZE = 50

def extract():
    # 1. 设置设备 (解决 RuntimeError: cuda:0 and cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")

    # 2. 构造数据并移动到 GPU
    seq = [5, 10, 6, 11, 7, 12, 8, 13] 
    seq = seq + [0] * (SEQ_LEN - len(seq))
    seq[-1] = 5 
    
    input_ids = torch.tensor([seq], dtype=torch.long).to(device) # <--- 关键修改
    
    save_data = {"results": {}, "input_ids": input_ids.cpu()} # 保存时转回 cpu

    for name, path in PATHS.items():
        if not os.path.exists(path):
            print(f"❌ 跳过: {path} 不存在")
            continue
            
        print(f">>> 正在处理: {name} ...")
        
        # 3. 加载模型
        cfg = ModelConfig(
            d_model=512, n_heads=8, n_layers=8, mlp_ratio=4,
            max_sequence_length=SEQ_LEN, vocab_size=VOCAB_SIZE,
            rope=True, 
            flash_attention=False, # 必须关闭
            use_scaled_rope1=(name!="Baseline"),
            scaled_rope_sigma=200.0 if name!="Baseline" else 1.0,
            rope_scaling_threshold=3 if name!="Baseline" else -1,
            init_device="cuda" # 显式指定初始化设备
        )
        
        model = OLMo(cfg)
        
        # 加载权重
        state_dict = torch.load(path, map_location=device) # 加载到 GPU
        model.load_state_dict(state_dict, strict=False)
        model.to(device) # 双重保险
        model.eval()
        
        # 4. 提取
        with torch.no_grad():
            model(input_ids) 
            
            try:
                # 寻找 Layer 5
                block = model.transformer.blocks[5]
                # 兼容 att / attention 命名
                attn_module = block.att if hasattr(block, 'att') else block.attn_out # 这里OlmoBlock通常没有att属性，而是直接在forward里调用self.attention
                
                # Wait, OLMoBlock structure in model2.py:
                # 它没有 self.att 成员变量！它是在 forward 里调用的 self.attention 方法。
                # 所以我们不能通过 block.att 访问。
                # === [更正策略] ===
                # 我们修改的是 OLMoBlock 类里的 _scaled_dot_product_attention 方法
                # 所以权重被保存在了 block 实例本身上 (self.last_attn_weights)
                
                if hasattr(block, 'last_attn_weights'):
                    weights = block.last_attn_weights # 已经在 model2 中 detach().cpu() 了
                    save_data["results"][name] = weights
                    print(f"   ✅ 提取成功! Shape: {weights.shape}")
                else:
                    print(f"   ❌ 失败: block[5] 中没有 last_attn_weights。请检查 model2.py 修改是否生效。")
            except Exception as e:
                print(f"   ❌ 异常: {e}")

    torch.save(save_data, "exp1_attention_data.pt")
    print("\n>>> 🎉 完成! 请下载 exp1_attention_data.pt")

if __name__ == "__main__":
    extract()