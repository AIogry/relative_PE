"""
HIPE + YaRN 集成模块
解决频率-权重耦合冲突问题
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

from .model import RotaryEmbedding, _yarn_ramp, _yarn_get_mscale, _non_meta_init_device
from .config import ModelConfig


class ScaledRotaryEmbeddingYaRN(RotaryEmbedding):
    """
    YaRN-aware ScaledRoPE (HIPE)
    
    核心改进：解耦旋转频率和权重计算频率
    - inv_freq: 用于 RoPE 旋转，可被 YaRN 动态调整
    - inv_freq_base: 用于 HIPE 权重计算，固定不变
    
    这样 YaRN 可以实现长度外推，而 HIPE 的生物启发权重保持其设计意图
    """
    
    def __init__(
        self, 
        config: ModelConfig, 
        cache, 
        sigma: float = 1.0, 
        layer_index: Optional[int] = None
    ):
        # 关键：在调用父类之前先保存配置和参数
        self._sigma_init = sigma
        self._layer_index = layer_index
        self.config = config
        
        # 调用父类初始化（这会设置 inv_freq，可能包含 YaRN 调整）
        super().__init__(config, cache)
        
        # 初始化分层控制
        self.use_scaling = True
        scaling_threshold = getattr(config, "rope_scaling_threshold", -1)
        if scaling_threshold >= 0 and layer_index is not None:
            if layer_index <= scaling_threshold:
                self.use_scaling = False
                print(f"Layer {layer_index}: Scaling disabled (standard RoPE)")
        
        # 可学习 sigma 设置
        self.is_learnable = getattr(config, "learnable_sigma", False)
        
        # 关键：保存基准频率（不被 YaRN 影响）
        device = _non_meta_init_device(config)
        self._register_base_frequency(device)
        
        # 初始化 HIPE 权重
        if self.use_scaling:
            self._initialize_hipe_weights(device)
    
    def _register_base_frequency(self, device: torch.device):
        """
        注册基准频率，这是 HIPE 权重计算的基础
        不随 YaRN 的动态缩放而改变
        """
        dim = self.config.d_model // self.config.n_heads
        i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        
        # 基准频率：标准的 RoPE 频率计算
        inv_freq_base = 1.0 / (self.config.rope_theta ** (i / dim))
        
        self.register_buffer('inv_freq_base', inv_freq_base)
        
        # 同时保存基准序列长度
        self.base_seq_len = getattr(self.config, 'yarn_max_position_embeddings', 
                                    self.config.max_sequence_length)
    
    def _initialize_hipe_weights(self, device: torch.device):
        """初始化 HIPE 权重（基于基准频率）"""
        dim = self.config.d_model // self.config.n_heads
        n_heads = self.config.n_heads
        
        if self.is_learnable:
            # 可学习 sigma：创建参数，权重动态计算
            initial_sigmas = torch.ones(n_heads) * self._sigma_init
            self.sigma_param = nn.Parameter(initial_sigmas)
            print(f"Layer {self._layer_index}: Learnable sigma initialized at {self._sigma_init}")
        else:
            # 固定 sigma：预计算权重
            self._compute_static_weights(device)
    
    def _compute_static_weights(self, device: torch.device):
        """计算静态 HIPE 权重"""
        n_heads = self.config.n_heads
        
        if isinstance(self._sigma_init, (list, tuple)):
            sigmas = list(self._sigma_init)
        else:
            sigmas = [self._sigma_init] * n_heads
        
        sigmas_tensor = torch.tensor(sigmas, device=device, dtype=torch.float).view(n_heads, 1)
        
        # 关键：使用 inv_freq_base 而非 inv_freq
        freqs = self.inv_freq_base.view(1, -1)
        
        # 计算权重
        decay_func = getattr(self.config, 'decay_func', 'gaussian')
        
        if decay_func == 'gaussian':
            scale = torch.exp(-sigmas_tensor**2 * freqs**2 / 2) * freqs
        elif decay_func == 'exp':
            scale = (1/sigmas_tensor)**2 / ((1/sigmas_tensor)**2 + freqs**2) * freqs
        elif decay_func == 'power':
            scale = torch.exp(-sigmas_tensor * freqs) * freqs
        elif decay_func == 'segmented':
            order = getattr(self.config, 'decay_order', 8)
            scale = (1.0 / (1.0 + (sigmas_tensor * freqs) ** order)) * freqs
        else:
            scale = torch.exp(-sigmas_tensor**2 * freqs**2 / 2) * freqs
        
        # 处理和归一化
        scale = torch.sqrt(scale)
        scale_full = torch.cat((scale, scale), dim=-1)
        correction_factor = torch.rsqrt(torch.mean(scale_full**2))
        scale_full = scale_full * correction_factor
        
        self.register_buffer('hipe_scale_factor', scale_full)
    
    def _compute_dynamic_weights(self, device: torch.device) -> torch.Tensor:
        """
        动态计算权重（用于可学习 sigma）
        仍然基于 inv_freq_base 而非 inv_freq
        """
        current_sigmas = torch.clamp(self.sigma_param, min=1e-3).to(device)
        current_sigmas = current_sigmas.view(self.config.n_heads, 1)
        
        # 关键：始终使用 inv_freq_base
        freqs = self.inv_freq_base.view(1, -1)
        
        decay_func = getattr(self.config, 'decay_func', 'gaussian')
        
        if decay_func == 'gaussian':
            scale = torch.exp(-current_sigmas**2 * freqs**2 / 2) * freqs
        elif decay_func == 'exp':
            scale = (1/current_sigmas)**2 / ((1/current_sigmas)**2 + freqs**2) * freqs
        elif decay_func == 'power':
            scale = torch.exp(-current_sigmas * freqs) * freqs
        elif decay_func == 'segmented':
            order = getattr(self.config, 'decay_order', 8)
            scale = (1.0 / (1.0 + (current_sigmas * freqs) ** order)) * freqs
        else:
            scale = torch.exp(-current_sigmas**2 * freqs**2 / 2) * freqs
        
        scale = torch.clamp(scale, min=1e-10)
        scale = torch.sqrt(scale)
        scale_full = torch.cat((scale, scale), dim=-1)
        
        # RMS 归一化
        mean_square = torch.mean(scale_full**2, dim=-1, keepdim=True)
        mean_square = torch.clamp(mean_square, min=1e-10)
        correction_factor = torch.rsqrt(mean_square)
        scale_full = scale_full * correction_factor
        scale_full = torch.nan_to_num(scale_full, nan=1.0, posinf=1.0, neginf=1.0)
        
        return scale_full.view(1, self.config.n_heads, 1, -1)
    
    def _get_scale_factor(self, device: torch.device) -> torch.Tensor:
        """
        获取 HIPE 缩放因子
        
        关键改进：始终基于 inv_freq_base 计算/获取权重，
        不受当前 inv_freq（可能被 YaRN 修改）的影响
        """
        if not self.use_scaling:
            dim = self.config.d_model // self.config.n_heads
            return torch.ones(1, self.config.n_heads, 1, dim, device=device)
        
        if self.is_learnable:
            return self._compute_dynamic_weights(device)
        else:
            return self.hipe_scale_factor.view(1, self.config.n_heads, 1, -1).to(device)
    
    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t):
        """
        应用 RoPE + HIPE
        
        步骤：
        1. 用 HIPE 权重缩放输入（基于原始频率）
        2. 用可能受 YaRN 影响的旋转编码进行旋转
        """
        # 1. HIPE 缩放（基于原始频率的权重）
        scale_factor = self._get_scale_factor(t.device)
        t_scaled = t * scale_factor
        
        # 2. 标准 RoPE 旋转（可能使用 YaRN 调整后的频率）
        return super().apply_rotary_pos_emb(pos_sin, pos_cos, t_scaled)
    
    def get_sigma_values(self) -> Optional[torch.Tensor]:
        """获取当前 sigma 值（用于监控）"""
        if self.is_learnable:
            return torch.clamp(self.sigma_param, min=1e-3)
        return None


class YaRNHipeConfig:
    """
    YaRN + HIPE 的配置辅助类
    
    使用示例：
        config = ModelConfig(...)
        
        # 启用 YaRN + HIPE
        config.yarn_enabled = True
        config.yarn_max_position_embeddings = 512  # 训练长度
        config.yarn_target_max_position_embeddings = 2048  # 目标长度
        config.use_scaled_rope1 = True
        config.scaled_rope_sigma = 200.0
        
        # 使用 YaRN-aware HIPE
        from OLMo.olmo.hipe_yarn import ScaledRotaryEmbeddingYaRN
        rope_emb = ScaledRotaryEmbeddingYaRN(config, cache, sigma=200.0)
    """
    
    @staticmethod
    def create_model_config(
        base_seq_len: int = 512,
        target_seq_len: int = 2048,
        sigma: float = 200.0,
        use_learnable_sigma: bool = False,
        **kwargs
    ):
        """
        创建兼容 YaRN + HIPE 的模型配置
        
        Args:
            base_seq_len: 基础训练序列长度（YaRN L_base）
            target_seq_len: 目标外推序列长度（YaRN L_target）
            sigma: HIPE sigma 初始值
            use_learnable_sigma: 是否使用可学习 sigma
        """
        from .config import ModelConfig
        
        config = ModelConfig(
            max_sequence_length=target_seq_len,  # 设置为目标长度
            yarn_enabled=True,
            yarn_max_position_embeddings=base_seq_len,
            yarn_target_max_position_embeddings=target_seq_len,
            yarn_beta_slow=1.0,
            yarn_beta_fast=32.0,
            use_scaled_rope1=True,
            scaled_rope_sigma=sigma,
            learnable_sigma=use_learnable_sigma,
            **kwargs
        )
        
        return config


def patch_model_for_yarn_hipe(model: nn.Module):
    """
    将已有模型的 ScaledRotaryEmbedding 替换为 YaRN-aware 版本
    
    用于在加载预训练模型后，动态切换到 YaRN + HIPE 模式
    """
    from .model import OLMoBlock, OLMoSequentialBlock
    
    patched_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (OLMoBlock, OLMoSequentialBlock)):
            if hasattr(module, 'rotary_emb'):
                old_rope = module.rotary_emb
                
                # 检查是否是 ScaledRotaryEmbedding
                if hasattr(old_rope, 'sigma_param') or hasattr(old_rope, 'scale_factor'):
                    # 获取当前配置
                    config = old_rope.config
                    cache = old_rope._cache if hasattr(old_rope, '_cache') else old_rope._RotaryEmbedding__cache
                    
                    # 提取当前 sigma 值
                    if hasattr(old_rope, 'sigma_param'):
                        sigma = old_rope.sigma_param.data.mean().item()
                    elif hasattr(old_rope, 'scale_factor'):
                        sigma = getattr(config, 'scaled_rope_sigma', 200.0)
                    else:
                        sigma = 200.0
                    
                    # 提取层索引
                    layer_index = None
                    if hasattr(module, 'layer_id'):
                        layer_index = module.layer_id
                    
                    # 创建新的 YaRN-aware 实例
                    new_rope = ScaledRotaryEmbeddingYaRN(
                        config=config,
                        cache=cache,
                        sigma=sigma,
                        layer_index=layer_index
                    )
                    
                    # 替换
                    module.rotary_emb = new_rope
                    patched_count += 1
    
    print(f"Patched {patched_count} layers with YaRN-aware HIPE")
    return model
