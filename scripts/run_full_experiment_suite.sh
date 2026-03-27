#!/bin/bash
# 完整的实验套件：预训练 + Few-Shot外推评估
# 运行所有对比实验

set -e

CODE_DIR="/home/qijunrong/03-proj/PE"
SCRIPT_DIR="${CODE_DIR}/scripts"

# 实验配置
SEEDS=(6198)  # 可以添加多个seed用于统计显著性
MODEL_SIZE="300M"
SIGMA=700.0

# 预训练长度和测试长度
TRAIN_LEN=512
TEST_LENS=(1024 2048 4096)

echo "=================================================="
echo "Full Experiment Suite"
echo "Code: ${CODE_DIR}"
echo "Data: /data/qijunrong/03-proj/PE"
echo "Pretrain (C4, ${TRAIN_LEN}) + Few-Shot Extrap (ArXiv, ${TEST_LENS[@]})"
echo "=================================================="

# ==================== 阶段1: 预训练 ====================
echo ""
echo "=================================================="
echo "Phase 1: Pretraining on C4"
echo "=================================================="

# 1.1 标准RoPE基线
echo "Submitting: RoPE baseline..."
sbatch ${SCRIPT_DIR}/run_pretrain_c4.sh rope 0.0 42

# 1.2 RoPE + YaRN
echo "Submitting: RoPE + YaRN..."
sbatch ${SCRIPT_DIR}/run_pretrain_c4.sh rope_yarn 0.0 42

# 1.3 HIPE (无YaRN)
echo "Submitting: HIPE only..."
sbatch ${SCRIPT_DIR}/run_pretrain_c4.sh hipe ${SIGMA} 42

# 1.4 HIPE + YaRN (新方案)
echo "Submitting: HIPE + YaRN..."
sbatch ${SCRIPT_DIR}/run_pretrain_c4.sh hipe_yarn ${SIGMA} 42

echo ""
echo "All pretraining jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo ""
echo "=================================================="
echo "Phase 2: Few-Shot Extrapolation (run after pretraining completes)"
echo "=================================================="
echo ""
echo "After pretraining completes, run evaluation with:"
echo ""
echo "# Example: Evaluate HIPE+YaRN model"
echo "sbatch ${SCRIPT_DIR}/run_fewshot_extrap.sh \\"
echo "  /data/qijunrong/03-proj/PE/checkpoints/pretrain_c4/${MODEL_SIZE}_hipe_yarn_L${TRAIN_LEN}_sig${SIGMA}_s42/model_final.pt \\"
echo "  hipe_yarn ${SIGMA} 42"
echo ""

# 创建评估脚本模板
cat > ${SCRIPT_DIR}/run_all_evaluations_template.sh << EOF
#!/bin/bash
# 评估所有预训练模型（需要在预训练完成后手动运行）

CODE_DIR="/home/qijunrong/03-proj/PE"
SCRIPT_DIR="\${CODE_DIR}/scripts"
MODEL_SIZE="300M"
TRAIN_LEN=512
SIGMA=700.0
SEED=6198

CKPT_ROOT="/data/qijunrong/03-proj/PE/checkpoints/pretrain_c4"

# RoPE baseline
echo "Evaluating RoPE baseline..."
sbatch \${SCRIPT_DIR}/run_fewshot_extrap.sh \\
    "\${CKPT_ROOT}/\${MODEL_SIZE}_rope_L\${TRAIN_LEN}_sig0.0_s\${SEED}/model_final.pt" \\
    rope 0.0 \${SEED}

# RoPE + YaRN
echo "Evaluating RoPE + YaRN..."
sbatch \${SCRIPT_DIR}/run_fewshot_extrap.sh \\
    "\${CKPT_ROOT}/\${MODEL_SIZE}_rope_yarn_L\${TRAIN_LEN}_sig0.0_s\${SEED}/model_final.pt" \\
    rope_yarn 0.0 \${SEED}

# HIPE only
echo "Evaluating HIPE only..."
sbatch \${SCRIPT_DIR}/run_fewshot_extrap.sh \\
    "\${CKPT_ROOT}/\${MODEL_SIZE}_hipe_L\${TRAIN_LEN}_sig\${SIGMA}_s\${SEED}/model_final.pt" \\
    hipe \${SIGMA} \${SEED}

# HIPE + YaRN
echo "Evaluating HIPE + YaRN..."
sbatch \${SCRIPT_DIR}/run_fewshot_extrap.sh \\
    "\${CKPT_ROOT}/\${MODEL_SIZE}_hipe_yarn_L\${TRAIN_LEN}_sig\${SIGMA}_s\${SEED}/model_final.pt" \\
    hipe_yarn \${SIGMA} \${SEED}

echo "All evaluation jobs submitted!"
EOF

chmod +x ${SCRIPT_DIR}/run_all_evaluations_template.sh

echo "Created evaluation template: ${SCRIPT_DIR}/run_all_evaluations_template.sh"
echo ""
echo "=================================================="
echo "Phase 3: Multi-Shot Size Experiments (optional)"
echo "=================================================="
echo ""
echo "After main evaluation, run multi-shot comparison:"
echo "  sbatch ${SCRIPT_DIR}/run_multi_shot_experiments.sh \\"
echo "    /data/qijunrong/03-proj/PE/checkpoints/pretrain_c4/300M_hipe_yarn_L512_sig700.0_s42/model_final.pt \\"
echo "    hipe_yarn 700.0 42"
echo ""
echo "This will test K=16,64,128,256,512 with appropriate steps/lr"
echo ""
echo "=================================================="
echo "Experiment Summary"
echo "=================================================="
echo ""
echo "预训练实验 (C4, 1B tokens):"
echo "  1. RoPE baseline"
echo "  2. RoPE + YaRN"
echo "  3. HIPE (no YaRN)"
echo "  4. HIPE + YaRN (NEW)"
echo ""
echo "Few-Shot外推评估 (ArXiv):"
echo "  - Base length: ${TRAIN_LEN}"
echo "  - Test lengths: ${TEST_LENS[@]}"
echo "  - Few-shot K: 256 samples"
echo "  - Adaptation: 100 steps"
echo ""
echo "对比维度:"
echo "  - PPL on different lengths"
echo "  - Extrapolation ratio (PPL_extrap / PPL_base)"
echo "  - Effect of few-shot adaptation"
echo "=================================================="
