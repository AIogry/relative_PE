# 实验设计详细说明

## 📊 数据集规模

### C4预训练数据
- **训练集**: c4_30M_train (~30M条)
- **实际使用**: 5M条 (约1B tokens)
- **验证集**: 10k条

### ArXiv外推评估数据
- **总样本**: 2,000条 (验证集)
- **文本长度**: 平均56k字符，中位数45k字符
- **领域**: 学术计算机科学论文

---

## 🎯 Few-Shot设计原理

### 为什么选128条？

```
理论基础: Brown et al. (2020) GPT-3 few-shot
实践考量:
  - 128条 ≈ 6.4%的验证集，保留93.6%用于测试
  - 每条ArXiv论文平均~10k tokens
  - 128条 ≈ 1.3M tokens，足够捕获领域特征
  - 防止过拟合（256条在小模型上可能过拟合）
```

### 为什么50步？

```
计算:
  - batch_size = 4 (micro) * (512/512) = 4
  - 128条 / 4 = 32 batches per epoch
  - 50步 ≈ 1.6 epoch

对比:
  - 100步 ≈ 3.1 epoch (容易过拟合)
  - 50步 ≈ 1.6 epoch (保守适应)
```

### 为什么学习率5e-6？

```
对比预训练LR=3e-4:
  - 5e-6 / 3e-4 = 1/60
  - 典型的fine-tune LR = pretrain LR / 10 ~ /100
  - 取1/60作为保守值

考虑:
  - 预训练模型已收敛
  - 只需轻微domain adaptation
  - 防止破坏预训练知识
```

---

## 🧪 测试评估设计

### 为什么用全部1744条测试？

```
统计稳定性计算:
  - 假设每个batch处理 ~8k tokens (batch=4, len=2048)
  - 1744条论文平均 ~10k tokens
  - 分组后约 ~2,000个测试样本
  - 总评估tokens: ~16M tokens

置信度:
  - 16M tokens提供的PPL估计误差 < 1%
  - 相比5个batch (~40k tokens)，精度提高400倍
```

### 长度外推测试

| 长度 | 与训练长度比 | 预期PPL增长 | 失败阈值 |
|------|-------------|------------|----------|
| 512 | 1x (baseline) | 1.0x | - |
| 1024 | 2x | < 1.2x | > 1.5x |
| 2048 | 4x | < 1.5x | > 3.0x |
| 4096 | 8x | < 2.0x | > 5.0x |

---

## 🔬 对照组设计

### 4组实验的科学意义

| 组 | 位置编码 | YaRN | 研究问题 |
|---|----------|------|----------|
| 1 | RoPE | ❌ | **基线**: 无外推机制的表现 |
| 2 | RoPE | ✅ | **YaRN单独效果**: 频率插值的价值 |
| 3 | HIPE | ❌ | **HIPE单独效果**: 生物启发的频带分工 |
| 4 | HIPE | ✅ | **组合效果**: 我们的核心贡献 |

### 关键对比

```
问题1: HIPE本身是否有外推能力?
  -> 对比 Group 1 vs Group 3 (都无YaRN)

问题2: YaRN是否有效?
  -> 对比 Group 1 vs Group 2 (RoPE基线 vs RoPE+YaRN)

问题3: HIPE+YaRN是否优于单独使用?
  -> 对比 Group 2 vs Group 4
  -> 对比 Group 3 vs Group 4

问题4: 我们的解耦设计是否成功?
  -> Group 4应优于所有其他组
```

---

## ⚠️ 潜在局限

### 1. 数据量限制
- ArXiv只有2000条，无法做K-fold交叉验证
- 解决方案: 使用不同random seed多次实验

### 2. Domain Gap
- C4(通用网页) -> ArXiv(学术) 跨度较大
- 解决方案: few-shot adaptation是必要的

### 3. 测试长度上限
- ArXiv论文平均45k字符，约15k tokens
- 可以测试到4096，但8192可能样本不足

---

## 📈 成功判定标准

### 主要指标
1. **Extrapolation Ratio** at 4096:
   - ❌ Fail: > 5.0x (相比baseline无改进)
   - ⚠️ OK: 2.0x - 5.0x (有改进但不显著)
   - ✅ Success: < 2.0x (显著改进)
   - 🌟 Excellent: < 1.5x (突破性改进)

2. **Relative Improvement**:
   - HIPE+YaRN vs RoPE+YaRN at 4096: > 10%
   - HIPE+YaRN vs HIPE only at 4096: > 20%

### 次要指标
- 训练稳定性: 无NaN/Inf
- 收敛速度: 与baseline相当
- 内存开销: 与baseline相当

---

## 🔧 超参数敏感性

### 需要调优的参数
1. **HIPE sigma**: 700.0 (基于之前实验)
2. **HIPE threshold**: 7 (前8层标准RoPE)
3. **YaRN alpha/beta**: 使用默认值

### 建议的调优实验（如果时间允许）
```bash
# 不同sigma
for sigma in 500.0 700.0 1000.0; do
    sbatch scripts/run_pretrain_c4.sh hipe_yarn $sigma 42
done

# 不同threshold
for thr in 5 7 9; do
    # 需要修改代码支持
    sbatch scripts/run_pretrain_c4.sh hipe_yarn 700.0 42 --threshold $thr
done
```

---

## 📚 参考

1. YaRN: "YaRN: Efficient Context Window Extension of Large Language Models" (2023)
2. GPT-3: "Language Models are Few-Shot Learners" (Brown et al., 2020)
3. Positional Encoding: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
4. Length Extrapolation: "Exploring Length Generalization in Large Language Models" (2022)
