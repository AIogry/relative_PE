"""
实验结果分析和可视化脚本
"""
import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_results(results_dir):
    """加载所有结果文件"""
    results = []
    pattern = os.path.join(results_dir, "*_extrap.json")
    
    for filepath in glob.glob(pattern):
        with open(filepath, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results

def extract_metrics(result):
    """从结果中提取关键指标"""
    config = result['config']
    pe_type = config['pe_type']
    sigma = config['sigma']
    base_len = config['base_len']
    
    metrics = {
        'pe_type': pe_type,
        'sigma': sigma,
        'base_len': base_len,
        'model_path': config['model_path'],
    }
    
    # 基础评估
    if 'after_adapt' in result['base_eval']:
        base_ppl = result['base_eval']['after_adapt']['ppl']
    else:
        base_ppl = result['base_eval']['no_adapt']['ppl']
    
    metrics['base_ppl'] = base_ppl
    
    # 外推评估
    for length, data in result['extrap_eval'].items():
        metrics[f'ppl_{length}'] = data['ppl']
        metrics[f'ratio_{length}'] = data['ppl'] / base_ppl
    
    return metrics

def create_comparison_table(results):
    """创建对比表格"""
    metrics_list = [extract_metrics(r) for r in results]
    df = pd.DataFrame(metrics_list)
    
    # 排序
    pe_order = ['rope', 'rope_yarn', 'hipe', 'hipe_yarn']
    df['pe_type'] = pd.Categorical(df['pe_type'], categories=pe_order, ordered=True)
    df = df.sort_values('pe_type')
    
    return df

def plot_extrapolation_comparison(df, save_path=None):
    """绘制外推对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 获取测试长度
    length_cols = [c for c in df.columns if c.startswith('ppl_')]
    lengths = sorted([int(c.split('_')[1]) for c in length_cols])
    
    pe_types = df['pe_type'].unique()
    colors = {'rope': '#1f77b4', 'rope_yarn': '#ff7f0e', 
              'hipe': '#2ca02c', 'hipe_yarn': '#d62728'}
    labels = {'rope': 'RoPE', 'rope_yarn': 'RoPE+YaRN',
              'hipe': 'HIPE', 'hipe_yarn': 'HIPE+YaRN (NEW)'}
    
    # 左图：绝对PPL
    ax1 = axes[0]
    for pe in pe_types:
        row = df[df['pe_type'] == pe].iloc[0]
        ppls = [row['base_ppl']] + [row[f'ppl_{l}'] for l in lengths]
        x_vals = [row['base_len']] + lengths
        ax1.plot(x_vals, ppls, marker='o', label=labels.get(pe, pe),
                color=colors.get(pe, 'gray'), linewidth=2, markersize=8)
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Perplexity', fontsize=12)
    ax1.set_title('Perplexity vs Sequence Length', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 右图：相对比例
    ax2 = axes[1]
    for pe in pe_types:
        row = df[df['pe_type'] == pe].iloc[0]
        ratios = [1.0] + [row[f'ratio_{l}'] for l in lengths]
        x_vals = [row['base_len']] + lengths
        ax2.plot(x_vals, ratios, marker='s', label=labels.get(pe, pe),
                color=colors.get(pe, 'gray'), linewidth=2, markersize=8)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('PPL Ratio (vs Base Length)', fontsize=12)
    ax2.set_title('Extrapolation Degradation', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def print_summary_table(df):
    """打印摘要表格"""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    # 获取测试长度
    length_cols = [c for c in df.columns if c.startswith('ppl_')]
    lengths = sorted([int(c.split('_')[1]) for c in length_cols])
    
    # 打印表头
    header = f"{'Method':<20} {'Base PPL':>10}"
    for l in lengths:
        header += f" {f'PPL@{l}':>12} {f'Ratio@{l}':>10}"
    print(header)
    print("-"*80)
    
    # 打印每行
    pe_labels = {'rope': 'RoPE', 'rope_yarn': 'RoPE+YaRN',
                 'hipe': 'HIPE', 'hipe_yarn': 'HIPE+YaRN'}
    
    for _, row in df.iterrows():
        pe = row['pe_type']
        label = pe_labels.get(pe, pe)
        line = f"{label:<20} {row['base_ppl']:>10.2f}"
        
        for l in lengths:
            ppl = row[f'ppl_{l}']
            ratio = row[f'ratio_{l}']
            line += f" {ppl:>12.2f} {ratio:>10.2f}"
        
        print(line)
    
    print("="*80)
    
    # 计算改进幅度
    if 'hipe_yarn' in df['pe_type'].values and 'rope' in df['pe_type'].values:
        print("\nIMPROVEMENT OVER BASELINE (RoPE):")
        print("-"*80)
        
        rope_row = df[df['pe_type'] == 'rope'].iloc[0]
        hipe_yarn_row = df[df['pe_type'] == 'hipe_yarn'].iloc[0]
        
        for l in lengths:
            rope_ppl = rope_row[f'ppl_{l}']
            hipe_ppl = hipe_yarn_row[f'ppl_{l}']
            improvement = (rope_ppl - hipe_ppl) / rope_ppl * 100
            print(f"  Length {l}: {improvement:+.2f}% (RoPE: {rope_ppl:.2f} -> HIPE+YaRN: {hipe_ppl:.2f})")
        
        print("="*80)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing result JSON files")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_plot", type=str, default=None)
    args = parser.parse_args()
    
    # 加载结果
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} result files")
    
    if len(results) == 0:
        print("No results found!")
        return
    
    # 创建对比表
    df = create_comparison_table(results)
    
    # 打印摘要
    print_summary_table(df)
    
    # 保存CSV
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nCSV saved to: {args.output_csv}")
    
    # 绘制图表
    if args.output_plot:
        plot_extrapolation_comparison(df, args.output_plot)
    else:
        plot_extrapolation_comparison(df)

if __name__ == "__main__":
    main()
