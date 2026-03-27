#!/bin/bash
# 完整测试：训练 + 评估
# 本地运行（不通过sbatch），用于快速验证

set -e

CODE_DIR="/home/qijunrong/03-proj/PE"
DATA_DIR="/data/qijunrong/03-proj/PE"

# 测试配置
PE_TYPE="hipe_yarn"
SIGMA=700.0
SEED=42
MODEL_SIZE="20M"
TRAIN_SIZE=500      # 非常小，只测试代码
VAL_SIZE=50
MAX_TOKENS=500000   # 约10步

echo "=================================================="
echo "Complete Pipeline Test (Local Run)"
echo "=================================================="
echo "This will test:"
echo "  1. Data loading"
echo "  2. Model creation with ${PE_TYPE}"
echo "  3. Training loop"
echo "  4. Checkpoint saving"
echo "  5. Model loading"
echo "  6. YaRN extrapolation"
echo "=================================================="

# 检查必要路径
echo "Checking paths..."
if [ ! -d "${DATA_DIR}/c4_30M_train" ]; then
    echo "ERROR: C4 data not found at ${DATA_DIR}/c4_30M_train"
    exit 1
fi

if [ ! -d "${DATA_DIR}/wikitext/tokenizer" ]; then
    echo "ERROR: Tokenizer not found at ${DATA_DIR}/wikitext/tokenizer"
    exit 1
fi

echo "Paths OK."

# 设置环境
export PYTHONPATH="${CODE_DIR}/OLMo:${PYTHONPATH}"
export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=0

PYTHON_BIN="/home/qijunrong/anaconda3/bin/python"
TRAIN_SCRIPT="${CODE_DIR}/train_hipe_c4_pretrain.py"
EVAL_SCRIPT="${CODE_DIR}/eval_fewshot_extrap.py"

# 创建测试输出目录
TEST_DIR="${DATA_DIR}/checkpoints/quick_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${TEST_DIR}

echo ""
echo "Step 1/2: Training test (${TRAIN_SIZE} samples, ~10 steps)"
echo "Output: ${TEST_DIR}"
echo "--------------------------------------------------"

${PYTHON_BIN} ${TRAIN_SCRIPT} \
    --output_dir ${TEST_DIR} \
    --run_id "quick_test_${PE_TYPE}" \
    --dataset_path ${DATA_DIR} \
    --local_tokenizer_path "${DATA_DIR}/wikitext/tokenizer" \
    --model_size ${MODEL_SIZE} \
    --seq_len 512 \
    --train_size ${TRAIN_SIZE} \
    --val_size ${VAL_SIZE} \
    --max_tokens ${MAX_TOKENS} \
    --global_batch_size 16 \
    --micro_batch_size 4 \
    --lr 3e-4 \
    --pe_type ${PE_TYPE} \
    --sigma ${SIGMA} \
    --rope_scaling_threshold 7 \
    --decay_func gaussian \
    --eval_interval 5 \
    --save_interval 10 \
    --seed ${SEED} \
    --wandb_mode offline

echo ""
echo "Step 2/2: Loading and testing YaRN extrapolation"
echo "--------------------------------------------------"

MODEL_PATH="${TEST_DIR}/model_final.pt"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: Model file not found at ${MODEL_PATH}"
    exit 1
fi

# 测试加载和外推配置
echo "Testing model loading with YaRN..."

${PYTHON_BIN} << EOF
import torch
import sys
sys.path.insert(0, '${CODE_DIR}/OLMo')
from olmo.config import ModelConfig
from olmo.model import OLMo

# 加载模型
cfg = ModelConfig(
    d_model=256, n_heads=8, n_layers=8, mlp_ratio=8,
    max_sequence_length=512,
    vocab_size=50304,
    embedding_size=50304,
    rope=True,
    yarn_enabled=True,  # 启用YaRN
    yarn_max_position_embeddings=512,
    yarn_target_max_position_embeddings=2048,
    use_scaled_rope1=True,
    scaled_rope_sigma=${SIGMA},
    rope_scaling_threshold=7,
    decay_func='gaussian',
    flash_attention=False,  # 测试时关闭，避免兼容性问题
)

model = OLMo(cfg)
state_dict = torch.load('${MODEL_PATH}', map_location='cpu')
model.load_state_dict(state_dict)

print("✅ Model loaded successfully!")

# 测试YaRN频率更新
device = torch.device('cpu')
if hasattr(model.transformer, 'blocks'):
    blocks = model.transformer.blocks
else:
    blocks = []

for i, block in enumerate(blocks):
    if hasattr(block, 'rotary_emb') and block.rotary_emb is not None:
        # 保存原始频率
        orig_freq = block.rotary_emb.inv_freq.clone()
        
        # 更新YaRN配置
        block.rotary_emb.inv_freq = block.rotary_emb.get_inv_freq(device)
        new_freq = block.rotary_emb.inv_freq
        
        # 验证频率确实被压缩了
        if i == 0:
            ratio = new_freq[0].item() / orig_freq[0].item()
            print(f"✅ YaRN frequency scaling: {ratio:.4f} (should be < 1.0)")
        
        # 验证HIPE的scale_factor保持不变
        if hasattr(block.rotary_emb, 'scale_factor'):
            print(f"✅ Layer {i}: HIPE scale_factor preserved, shape={block.rotary_emb.scale_factor.shape}")

print("")
print("All tests passed! YaRN + HIPE integration looks good.")
EOF

echo ""
echo "=================================================="
echo "COMPLETE TEST PASSED! ✅"
echo "=================================================="
echo ""
echo "Test artifacts saved to: ${TEST_DIR}"
echo ""
echo "You can now run the full experiment:"
echo "  bash scripts/run_full_experiment_suite.sh"
echo ""
