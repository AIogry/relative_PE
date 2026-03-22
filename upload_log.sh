#!/bin/bash
set -e  # 遇到错误立即退出，避免批量同步出错

# ===================== 配置区（已适配旧版WandB） =====================
ROOT_CHECKPOINT_DIR="/data/qijunrong/03-proj/PE/checkpoints_exp2/wiki20-60"
EXP_TYPE="base"
# 旧版WandB参数：--clean-force（替代--force）、--include-globs（替代--include）
FORCE_SYNC="--clean-force"
INCLUDE_GLOBS="--include-globs files/*"
# ===================== 无需修改以下内容 =====================

# 1. 检查WandB登录状态
echo ">>> 检查WandB登录状态..."
if ! wandb login --check > /dev/null 2>&1; then
    echo ">>> 请先登录WandB："
    wandb login
fi

# 2. 检查protobuf版本（避免兼容性问题）
REQUIRED_PROTOBUF="4.25.5"
CURRENT_PROTOBUF=$(python -c "import google.protobuf; print(google.protobuf.__version__)" 2>/dev/null || echo "not installed")
if [ "$CURRENT_PROTOBUF" != "$REQUIRED_PROTOBUF" ]; then
    echo ">>> 安装兼容的protobuf版本 ($REQUIRED_PROTOBUF)..."
    pip install protobuf==$REQUIRED_PROTOBUF -q
fi

# 3. 遍历所有seed目录下的wandb离线目录
echo -e "\n>>> 开始遍历${EXP_TYPE}实验的log.txt文件..."
find "${ROOT_CHECKPOINT_DIR}/${EXP_TYPE}" -path "*/seed_*/wandb/offline-run-*" -type d | grep -v -E "files|logs|tmp" | while read -r run_dir; do
    # 向上回溯找到seed目录（log.txt所在目录）
    seed_dir=$(dirname $(dirname "$run_dir"))
    log_file="${seed_dir}/log.txt"
    
    # 检查log.txt是否存在
    if [ ! -f "$log_file" ]; then
        echo ">>> [跳过] $log_file 不存在，跳过该实验"
        continue
    fi
    
    # 打印当前同步信息
    echo -e "\n========================================"
    echo ">>> 处理实验：$seed_dir"
    echo ">>> WandB离线目录：$run_dir"
    echo ">>> 待上传日志：$log_file"
    
    # 4. 创建WandB的files目录（确保存在）
    wandb_files_dir="${run_dir}/files"
    mkdir -p "$wandb_files_dir"
    
    # 5. 复制log.txt到WandB files目录（重命名避免冲突）
    log_filename=$(basename "$seed_dir")_log.txt  # 例如：seed_7_log.txt
    cp "$log_file" "${wandb_files_dir}/${log_filename}"
    echo ">>> 已复制log.txt到：${wandb_files_dir}/${log_filename}"
    
    # 6. 同步该WandB离线目录到云端（适配旧版参数）
    echo ">>> 开始同步到WandB云端..."
    wandb sync $FORCE_SYNC $INCLUDE_GLOBS "$run_dir"
    
    # 7. 验证同步结果
    if [ $? -eq 0 ]; then
        echo ">>> ✅ $seed_dir 同步成功！"
    else
        echo ">>> ❌ $seed_dir 同步失败，请检查日志"
    fi
done

echo -e "\n========================================"
echo ">>> 所有${EXP_TYPE}实验的log.txt上传完成！"
echo ">>> 登录 https://wandb.ai 查看对应实验的「Files」标签页"