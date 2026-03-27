#!/bin/bash
# SST-2数据集下载脚本 - 下载到 /data/qijunrong/03-proj/PE/sst2_data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/data/qijunrong/03-proj/PE/sst2_data"

echo ">>> Starting SST-2 dataset download..."
echo ">>> Target directory: $DATA_DIR"

cd "$SCRIPT_DIR"

# 运行下载脚本
python download_sst2.py --output_dir "$DATA_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo ">>> ✓ SST-2 dataset downloaded successfully!"
    echo ">>> Location: $DATA_DIR"
    ls -la "$DATA_DIR"
else
    echo ">>> ✗ Download failed!"
    exit 1
fi
