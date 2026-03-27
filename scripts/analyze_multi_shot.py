"""
多shot大小实验结果分析
"""
import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_multi_shot_results(results_dir, pe_type):
    """加载多shot实验结果"""
    pattern = os.path.join(results_dir, f"{pe_type}_K*.json")
    files = glob.glob(pattern)
    
    results = []
    for filepath in sorted(files):
        with open(filepath, 'r') as f:
            data = json.load(f)
            # 提取K值
            k = data['config']['few_shot_k']
            results.append((k, data))
    
    return sorted(results, key=lambda x: x[0])

def extract_metrics(data):
    """提取关键指标"""
    metrics = {
        'K': data['config']['few_shot_k'],
        'steps': data['config']['few_shot_steps'],
        'lr': data['config']['few_shot_lr'],
    }
    
    # Few-shot适应效果
    if 'few_shot' in data:
        if 'before' in data['few_shot']:
            metrics['ppl_before'] = data['few_shot']['before']['ppl']
        if 'after' in data['few_shot']:
            metrics['ppl_after'] = data['few_shot']['after']['ppl']
            if 'ppl_before' in metrics:
                metrics['improvement_pct'] = (
                    (metrics['ppl_before'] - metrics['ppl_after']) / metrics['ppl_before'] * 100
                )
    
    # 外推结果
    base_ppl = metrics.get('ppl_after', metrics.get('ppl_before', None))
    if base_ppl:
        for length, res in data.get('extrap_eval', {}).items():
            metrics[f'ppl_{length}'] = res['ppl']
            metrics[f'ratio_{length}'] = res['ppl'] / base_ppl
    
    return metrics

def plot_multi_shot_comparison(results_list, pe_type, save_path=None):
    """绘制多shot对比图"""
    # 提取数据
    df = pd.DataFrame([extract_metrics(data) for _, data in results_list])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    K_values = df['K'].values
    
    # 图1: Few-shot适应效果
    ax1 = axes[0, 0]
    if 'ppl_before' in df.columns and 'ppl_after' in df.columns:
        x = np.arange(len(K_values))
        width = 0.35
        ax1.bar(x - width/2, df['ppl_before'], width, label='Before Adapt', alpha=0.8)
        ax1.bar(x + width/2, df['ppl_after'], width, label='After Adapt', alpha=0.8)
        ax1.set_xlabel('Few-shot K')
        ax1.set_ylabel('Perplexity (Base Length)')
        ax1.set_title('Few-shot Adaptation Effect')
        ax1.set_xticks(x)
        ax1.set_xticklabels(K_values)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 图2: 适应提升百分比
    ax2 = axes[0, 1]
    if 'improvement_pct' in df.columns:
        ax2.bar(K_values, df['improvement_pct'], color='green', alpha=0.7)
        ax2.set_xlabel('Few-shot K')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Domain Adaptation Improvement')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
    
    # 图3: 不同长度下的PPL
    ax3 = axes[1, 0]
    length_cols = [c for c in df.columns if c.startswith('ppl_')]
    lengths = sorted([int(c.split('_')[1]) for c in length_cols])
    
    for length in lengths:
        col = f'ppl_{length}'
        if col in df.columns:
            ax3.plot(K_values, df[col], marker='o', label=f'Length {length}', linewidth=2)
    
    ax3.set_xlabel('Few-shot K')
    ax3.set_ylabel('Perplexity')
    ax3.set_title('PPL vs Few-shot K (Different Lengths)')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 外推比例
    ax4 = axes[1, 1]
    for length in lengths:
        col = f'ratio_{length}'
        if col in df.columns:
            ax4.plot(K_values, df[col], marker='s', label=f'Length {length}', linewidth=2)
    
    ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Few-shot K')
    ax4.set_ylabel('PPL Ratio (vs Base)')
    ax4.set_title('Extrapolation Ratio vs Few-shot K')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Multi-Shot Analysis: {pe_type}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return df

def print_multi_shot_table(df):
    """打印多shot对比表格"""
    print("\n" + "="*80)
    print("MULTI-SHOT EXPERIMENT RESULTS")
    print("="*80)
    
    # 基础信息
    print("\nBase Information:")
    print(df[['K', 'steps', 'lr']].to_string(index=False))
    
    # Few-shot适应效果
    print("\nFew-shot Adaptation:")
    cols = ['K']
    if 'ppl_before' in df.columns:
        cols.append('ppl_before')
    if 'ppl_after' in df.columns:
        cols.append('ppl_after')
    if 'improvement_pct' in df.columns:
        cols.append('improvement_pct')
    print(df[cols].to_string(index=False))
    
    # 外推结果
    print("\nExtrapolation Results:")
    length_cols = [c for c in df.columns if c.startswith('ppl_') or c.startswith('ratio_')]
    print(df[['K'] + sorted(length_cols)].to_string(index=False))
    
    print("="*80)
    
    # 关键发现
    print("\nKey Findings:")
    if 'improvement_pct' in df.columns:
        best_k = df.loc[df['improvement_pct'].idxmax(), 'K']
        best_improvement = df['improvement_pct'].max()
        print(f"  - Best adaptation: K={best_k} ({best_improvement:+.2f}%)")
    
    # 外推稳定性
    ratio_cols = [c for c in df.columns if c.startswith('ratio_')]
    if ratio_cols:
        max_length = max([int(c.split('_')[1]) for c in ratio_cols])
        ratio_col = f'ratio_{max_length}'
        if ratio_col in df.columns:
            best_extrap_k = df.loc[df[ratio_col].idxmin(), 'K']
            best_ratio = df[ratio_col].min()
            print(f"  - Best extrapolation (length {max_length}): K={best_extrap_k} (ratio={best_ratio:.2f}x)")
    
    # 推荐配置
    print("\nRecommended Configuration:")
    # 综合考虑adaptation improvement和extrapolation ratio
    if 'improvement_pct' in df.columns and ratio_cols:
        # 标准化评分
        df['adapt_score'] = (df['improvement_pct'] - df['improvement_pct'].min()) / (df['improvement_pct'].max() - df['improvement_pct'].min() + 1e-8)
        df['extrap_score'] = 1 - (df[ratio_cols[-1]] - df[ratio_cols[-1]].min()) / (df[ratio_cols[-1]].max() - df[ratio_cols[-1]].min() + 1e-8)
        df['total_score'] = 0.5 * df['adapt_score'] + 0.5 * df['extrap_score']
        
        best_k = df.loc[df['total_score'].idxmax(), 'K']
        print(f"  - Optimal K (balanced): {best_k}")
    
    print("="*80)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--pe_type", type=str, default="hipe_yarn")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_plot", type=str, default=None)
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    print(f"PE Type: {args.pe_type}")
    
    results = load_multi_shot_results(args.results_dir, args.pe_type)
    
    if not results:
        print(f"No results found for {args.pe_type} in {args.results_dir}")
        return
    
    print(f"Found {len(results)} experiments with K values: {[k for k, _ in results]}")
    
    # 分析并绘图
    df = plot_multi_shot_comparison(results, args.pe_type, args.output_plot)
    
    # 打印表格
    print_multi_shot_table(df)
    
    # 保存CSV
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nCSV saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
