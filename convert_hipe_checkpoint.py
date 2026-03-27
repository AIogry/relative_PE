#!/usr/bin/env python3
"""
HIPE 模型权重转换工具
用于在不同版本间转换 checkpoint
"""

import torch
import argparse
from pathlib import Path


def convert_fix_to_learnable(input_path: str, output_path: str, sigma: float = 200.0):
    """
    将 fix_sigma HIPE 模型转换为 learnable_sigma 格式
    
    fix_sigma: 有 'scale_factor' buffer
    learnable: 需要 'sigma_param' parameter
    
    转换逻辑：从 scale_factor 反推出 sigma 值
    """
    print(f"Loading checkpoint from: {input_path}")
    state_dict = torch.load(input_path, map_location='cpu')
    
    new_state_dict = {}
    converted_count = 0
    
    for key, value in state_dict.items():
        # 检查是否是 scale_factor
        if 'scale_factor' in key:
            # 获取 layer 信息
            # key format: transformer.blocks.{layer_id}.rotary_emb.scale_factor
            parts = key.split('.')
            
            # 尝试从 scale_factor 反推 sigma
            # scale_factor = sqrt(exp(-sigma^2 * freq^2 / 2) * freq)
            # 这是一个近似转换，使用默认 sigma
            
            # 创建 sigma_param (每个 head 一个值)
            n_heads = value.shape[0]  # (n_heads, dim)
            sigma_param = torch.ones(n_heads) * sigma
            
            # 替换 key
            new_key = key.replace('scale_factor', 'sigma_param')
            new_state_dict[new_key] = sigma_param
            
            print(f"  Converted: {key} -> {new_key}")
            print(f"    shape: {value.shape} -> {sigma_param.shape}")
            converted_count += 1
        else:
            new_state_dict[key] = value
    
    # 保存新的 checkpoint
    torch.save(new_state_dict, output_path)
    print(f"\n✅ Converted {converted_count} layers")
    print(f"Saved to: {output_path}")


def inspect_hipe_checkpoint(path: str):
    """检查 HIPE checkpoint 的类型"""
    print(f"Inspecting: {path}\n")
    
    state_dict = torch.load(path, map_location='cpu')
    
    has_scale_factor = False
    has_sigma_param = False
    rope_layers = []
    
    for key in state_dict.keys():
        if 'rotary_emb' in key:
            if 'scale_factor' in key:
                has_scale_factor = True
                rope_layers.append(key.replace('.scale_factor', ''))
            elif 'sigma_param' in key:
                has_sigma_param = True
                rope_layers.append(key.replace('.sigma_param', ''))
    
    print(f"Checkpoint type:")
    if has_sigma_param:
        print("  ✓ Learnable sigma HIPE (new version)")
    elif has_scale_factor:
        print("  ⚠ Fixed sigma HIPE (old version)")
        print(f"  Found {len(rope_layers)} HIPE layers with scale_factor")
    else:
        print("  ✓ Standard RoPE (no HIPE)")
    
    print(f"\nHIPE layers:")
    for layer in rope_layers[:5]:  # 只显示前5个
        print(f"  - {layer}")
    if len(rope_layers) > 5:
        print(f"  ... and {len(rope_layers) - 5} more")


def main():
    parser = argparse.ArgumentParser(description="HIPE checkpoint conversion tool")
    parser.add_argument("command", choices=["inspect", "convert"], 
                        help="Command: inspect or convert")
    parser.add_argument("--input", type=str, required=True,
                        help="Input checkpoint path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output checkpoint path (for convert)")
    parser.add_argument("--sigma", type=float, default=200.0,
                        help="Default sigma value for conversion")
    
    args = parser.parse_args()
    
    if args.command == "inspect":
        inspect_hipe_checkpoint(args.input)
    elif args.command == "convert":
        if not args.output:
            # 自动生成输出路径
            input_path = Path(args.input)
            args.output = str(input_path.parent / f"{input_path.stem}_learnable{input_path.suffix}")
        convert_fix_to_learnable(args.input, args.output, args.sigma)


if __name__ == "__main__":
    main()
