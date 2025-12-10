import torch
import pytest
from model2 import ModelConfig, RotaryEmbedding, ScaledRoPE, BufferCache

# 生成通用模型配置（可根据实际需求调整参数）
def get_test_config(d_model=512, n_heads=8, max_seq_len=1024):
    config = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        rope_theta=10000.0,
        rope_full_precision=False,
        max_sequence_length=max_seq_len,
        layer_norm_eps=1e-5,
        include_bias=True
    )
    return config

# 测试1：ScaledRoPE初始化与参数正确性
def test_scaled_rope_initialization():
    config = get_test_config()
    cache = BufferCache()
    scaled_rope = ScaledRoPE(config, lambda_val=10.0, sigma=20.0)
    
    # 验证基础参数
    assert scaled_rope.d_head == config.d_model // config.n_heads, "head维度计算错误"
    assert scaled_rope.d_half == scaled_rope.d_head // 2, "半维度计算错误"
    
    # 验证权重参数（w_sin和w_cos）
    assert hasattr(scaled_rope, "w_sin"), "未定义w_sin参数"
    assert hasattr(scaled_rope, "w_cos"), "未定义w_cos参数"
    assert scaled_rope.w_sin.shape == (scaled_rope.d_half,), "w_sin形状错误"
    assert scaled_rope.w_cos.shape == (scaled_rope.d_half,), "w_cos形状错误"
    assert torch.all(scaled_rope.w_sin >= 0) and torch.all(scaled_rope.w_sin <= 1), "w_sin应在[0,1]区间"
    assert torch.all(scaled_rope.w_cos >= 0) and torch.all(scaled_rope.w_cos <= 1), "w_cos应在[0,1]区间"

# 测试2：ScaledRoPE前向传播形状正确性
def test_scaled_rope_forward_shape():
    config = get_test_config()
    cache = BufferCache()
    scaled_rope = ScaledRoPE(config)
    
    # 生成测试输入（batch_size=2, n_heads=8, seq_len=10, head_dim=64）
    batch_size = 2
    seq_len_q = 10  # query序列长度
    seq_len_k = 15  # key序列长度（模拟生成式任务中Q/K长度不同的场景）
    head_dim = config.d_model // config.n_heads
    q = torch.randn(batch_size, config.n_heads, seq_len_q, head_dim)
    k = torch.randn(batch_size, config.n_heads, seq_len_k, head_dim)
    
    # 前向传播
    q_rot, k_rot = scaled_rope(q, k)
    
    # 验证输出形状：Q旋转后增加key长度维度，K形状不变
    assert q_rot.shape == (batch_size, config.n_heads, seq_len_q, seq_len_k, head_dim), \
        f"Q旋转后形状错误，实际{q_rot.shape}，预期{(batch_size, config.n_heads, seq_len_q, seq_len_k, head_dim)}"
    assert k_rot.shape == k.shape, f"K形状应不变，实际{k_rot.shape}，预期{k.shape}"

# 测试3：相对位置编码逻辑正确性（对比理论计算）
def test_relative_rotary_embedding():
    config = get_test_config(d_model=128, n_heads=2)  # 小维度便于手动验证
    scaled_rope = ScaledRoPE(config)
    q_len, k_len = 2, 3  # 短序列便于计算
    
    # 获取ScaledRoPE计算的相对位置编码
    pos_sin, pos_cos = scaled_rope.get_relative_rotary_embedding(q_len, k_len, device=torch.device("cpu"))
    
    # 手动计算相对位置差异（q_pos - k_pos）
    q_pos = torch.arange(q_len)  # [0,1]
    k_pos = torch.arange(k_len)  # [0,1,2]
    relative_diffs = q_pos.view(-1, 1) - k_pos.view(1, -1)  # 形状(2,3): [[0-0, 0-1, 0-2], [1-0, 1-1, 1-2]]
    
    # 手动计算频率（inv_freq * 相对位置差）
    inv_freq = scaled_rope.get_inv_freq(torch.device("cpu"))  # 形状(d_half,)
    manual_freqs = torch.einsum("ij, d -> ijd", relative_diffs, inv_freq)
    manual_sin = manual_freqs.sin()[None, None, :, :, :]  # 增加前两维（1,1,...）
    manual_cos = manual_freqs.cos()[None, None, :, :, :]
    
    # 验证ScaledRoPE计算结果与手动计算一致
    assert torch.allclose(pos_sin, manual_sin, atol=1e-6), "相对位置sin计算错误"
    assert torch.allclose(pos_cos, manual_cos, atol=1e-6), "相对位置cos计算错误"

# 测试4：数值稳定性（极端输入无NaN/Inf）
def test_scaled_rope_numerical_stability():
    config = get_test_config()
    scaled_rope = ScaledRoPE(config)
    
    # 测试极端输入：全零、全一、大值
    test_cases = [
        (torch.zeros(2, 8, 10, 64), torch.ones(2, 8, 10, 64)),
        (torch.randn(2, 8, 10, 64) * 1000, torch.randn(2, 8, 10, 64) * 1000),  # 大值输入
        (torch.ones(2, 8, 1, 64), torch.randn(2, 8, 100, 64))  # Q序列长度为1（生成起始）
    ]
    
    for q, k in test_cases:
        q_rot, k_rot = scaled_rope(q, k)
        assert not torch.isnan(q_rot).any(), "Q旋转结果含NaN"
        assert not torch.isinf(q_rot).any(), "Q旋转结果含Inf"
        assert not torch.isnan(k_rot).any(), "K旋转结果含NaN"

# 测试5：与原始RotaryEmbedding的兼容性（当lambda和sigma取极端值时逼近原始行为）
def test_scaled_rope_compatibility_with_original():
    config = get_test_config()
    cache = BufferCache()
    original_rope = RotaryEmbedding(config, cache)
    
    # 当lambda→∞（w_sin→1）且sigma→0（w_cos→1），ScaledRoPE应逼近原始RoPE
    scaled_rope = ScaledRoPE(config, lambda_val=1e6, sigma=1e-6)
    
    # 生成输入（Q/K长度相同，避免原始RoPE的相对位置逻辑差异）
    q = torch.randn(1, 8, 5, 64)
    k = torch.randn(1, 8, 5, 64)  # Q/K长度相同
    
    # 原始RoPE输出
    q_orig, k_orig = original_rope(q, k)
    
    # ScaledRoPE输出（需调整形状以匹配原始RoPE的输出维度）
    q_scaled, k_scaled = scaled_rope(q, k)
    q_scaled = q_scaled[:, :, torch.arange(5), torch.arange(5), :]  # 取对角线（对应原始RoPE的绝对位置）
    
    # 验证逼近性（允许一定误差，因ScaledRoPE是相对位置编码）
    assert torch.allclose(q_scaled, q_orig, atol=1e-3), "极端参数下ScaledRoPE未逼近原始RoPE"

# 测试6：不同数据类型兼容性（float16/float32）
def test_scaled_rope_dtype_compatibility():
    config = get_test_config()
    scaled_rope = ScaledRoPE(config)
    
    # 测试半精度输入
    q_fp16 = torch.randn(2, 8, 10, 64, dtype=torch.float16)
    k_fp16 = torch.randn(2, 8, 10, 64, dtype=torch.float16)
    q_rot_fp16, k_rot_fp16 = scaled_rope(q_fp16, k_fp16)
    assert q_rot_fp16.dtype == torch.float16, "半精度输入输出类型不匹配"
    
    # 测试单精度输入
    q_fp32 = torch.randn(2, 8, 10, 64, dtype=torch.float32)
    k_fp32 = torch.randn(2, 8, 10, 64, dtype=torch.float32)
    q_rot_fp32, k_rot_fp32 = scaled_rope(q_fp32, k_fp32)
    assert q_rot_fp32.dtype == torch.float32, "单精度输入输出类型不匹配"

if __name__ == "__main__":
    pytest.main(["-v", "test_rope.py"])  #  verbose模式运行所有测试