import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from itertools import islice
from tqdm import tqdm
import glob

OUTPUT_PATH = "/data/qijunrong/03-proj/PE/c4_30M_raw"
TARGET_SIZE = 30_000_000
BATCH_SIZE = 500_000

os.environ["HF_HOME"] = "/data/qijunrong/03-proj/PE/hf_cache"

print(f"🔧 修复合并问题...")

# 检查已完成的批次
existing_batches = glob.glob(os.path.join(OUTPUT_PATH, "batch_*"))
print(f"📊 发现批次目录: {len(existing_batches)} 个")

# 正确的合并方法
final_datasets = []
success_count = 0

for i in range(60):  # 检查所有60个批次
    batch_path = os.path.join(OUTPUT_PATH, f"batch_{i:03d}")
    
    if os.path.exists(batch_path):
        try:
            # ✅ 使用正确的加载方法
            batch_ds = load_from_disk(batch_path)
            final_datasets.append(batch_ds)
            success_count += 1
            print(f"✅ 成功加载批次 {i:03d}: {len(batch_ds):,} 条")
        except Exception as e:
            print(f"❌ 无法加载批次 {i:03d}: {e}")
    else:
        print(f"❌ 批次目录不存在: batch_{i:03d}")

if final_datasets:
    print(f"🔗 合并 {len(final_datasets)} 个批次...")
    final_ds = concatenate_datasets(final_datasets)
    final_ds.save_to_disk(OUTPUT_PATH)
    print(f"🎉 合并完成！总样本数: {len(final_ds):,}")
else:
    print("❌ 没有可合并的数据集")