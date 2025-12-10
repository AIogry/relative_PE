# build_c4_dataset.py
import os
import json
import gzip
from datasets import Dataset

# ========================
# 配置区（按需修改）
# ========================
# 分片所在目录（如果是当前目录，就用 "."）
SHARD_DIR = "/data/qijunrong/03-proj/PE/c4_raw_shards"

# 输出数据集目录名
OUTPUT_DATASET_DIR = "/data/qijunrong/03-proj/PE/c4_train_shards"

# ========================
# 加载训练分片
# ========================
def load_c4_shards(shard_dir):
    shard_files = sorted([
        f for f in os.listdir(shard_dir)
        if f.startswith("c4-train") and f.endswith(".json.gz")
    ])
    if not shard_files:
        raise FileNotFoundError(f"在 {shard_dir} 中未找到 c4-train.*.json.gz 文件！")
    
    print(f"发现 {len(shard_files)} 个训练分片: {shard_files[:3]}{'...' if len(shard_files) > 3 else ''}")
    all_texts = []
    for fname in shard_files:
        path = os.path.join(shard_dir, fname)
        print(f"正在加载: {fname}")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        all_texts.append(data["text"])
                    except Exception as e:
                        print(f"警告：跳过无效行 in {fname}: {e}")
    print(f"共加载 {len(all_texts):,} 条训练样本")
    return all_texts

# ========================
# 主流程
# ========================
if __name__ == "__main__":
    train_texts = load_c4_shards(SHARD_DIR)
    
    # 转为 Dataset
    full_dataset = Dataset.from_dict({"text": train_texts})
    
    # 划分：95% 训练，5% 验证
    split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
    
    # 保存
    split_dataset["train"].save_to_disk(OUTPUT_DATASET_DIR)  # 如 c4_train_shards
    split_dataset["test"].save_to_disk(OUTPUT_DATASET_DIR.replace("train", "val"))  # 如 c4_val_shards
    
    print(f"训练集 ({len(split_dataset['train'])} 条) 保存到: {OUTPUT_DATASET_DIR}")
    print(f"验证集 ({len(split_dataset['test'])} 条) 保存到: {OUTPUT_DATASET_DIR.replace('train', 'val')}")