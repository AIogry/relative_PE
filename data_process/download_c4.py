import os
from datasets import load_dataset
from itertools import islice
from tqdm import tqdm
import time

OUTPUT_PATH = "/data/qijunrong/03-proj/PE/c4_30M_raw"
TARGET_SIZE = 30_000_000
BATCH_SIZE = 500_000  # 每批50万条

os.environ["HF_HOME"] = "/data/qijunrong/03-proj/PE/hf_cache"

print(f"🚀 从断点继续下载 C4 (en) 前 {TARGET_SIZE:,} 条文本...")

# 检查已完成的批次
import glob
existing_batches = glob.glob(os.path.join(OUTPUT_PATH, "batch_*"))
existing_batch_nums = []
for batch_path in existing_batches:
    try:
        batch_num = int(os.path.basename(batch_path).split('_')[-1])
        existing_batch_nums.append(batch_num)
    except:
        continue

if existing_batch_nums:
    last_completed_batch = max(existing_batch_nums)
    print(f"📊 发现已完成的批次: {len(existing_batches)} 个")
    print(f"🔄 从批次 {last_completed_batch + 1} 继续下载...")
else:
    last_completed_batch = -1
    print("🆕 未发现已完成的批次，从头开始下载...")

# 流式加载数据集
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

# 计算总批次
total_batches = (TARGET_SIZE + BATCH_SIZE - 1) // BATCH_SIZE

# 确保输出目录存在
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 跳过已完成的批次（添加进度条）
if last_completed_batch >= 0:
    skip_count = (last_completed_batch + 1) * BATCH_SIZE
    print(f"⏩ 跳过前 {skip_count:,} 条数据...")
    
    # 添加跳过进度条
    start_time = time.time()
    skipped_samples = 0
    
    # 方法1：使用tqdm进度条（推荐）
    with tqdm(total=skip_count, desc="跳过进度", unit="样本", unit_scale=True) as pbar:
        for i, sample in enumerate(islice(dataset, skip_count)):
            skipped_samples += 1
            pbar.update(1)
            
            # 每10万条显示一次速度
            if i % 100000 == 0 and i > 0:
                elapsed = time.time() - start_time
                speed = i / elapsed
                remaining_time = (skip_count - i) / speed if speed > 0 else float('inf')
                pbar.set_postfix({
                    '速度': f'{speed:,.0f} 样本/秒',
                    '预计剩余': f'{remaining_time/60:.1f} 分钟'
                })
    
    skip_time = time.time() - start_time
    print(f"✅ 跳过完成！耗时: {skip_time:.1f} 秒, 速度: {skip_count/skip_time:,.0f} 样本/秒")

# 继续处理剩余批次
all_batches = []
start_batch = last_completed_batch + 1

print(f"📥 开始下载剩余批次 {start_batch} 到 {total_batches-1}...")

for batch_num in tqdm(range(start_batch, total_batches), 
                      desc="批次下载", 
                      initial=start_batch, 
                      total=total_batches):
    
    # 处理当前批次
    batch_samples = []
    batch_start_time = time.time()
    
    for sample in islice(dataset, BATCH_SIZE):
        batch_samples.append(sample)
    
    # 保存当前批次
    from datasets import Dataset
    batch_ds = Dataset.from_list(batch_samples)
    batch_path = os.path.join(OUTPUT_PATH, f"batch_{batch_num:03d}")
    batch_ds.save_to_disk(batch_path)
    all_batches.append(batch_ds)
    
    batch_time = time.time() - batch_start_time
    print(f"✅ 批次 {batch_num+1}/{total_batches} 完成: {len(batch_samples):,} 条, 耗时: {batch_time:.1f}秒")

# 合并所有批次到最终数据集
print("🔗 合并所有批次...")
from datasets import concatenate_datasets

# 加载所有已完成的批次
final_datasets = []
for i in range(total_batches):
    batch_path = os.path.join(OUTPUT_PATH, f"batch_{i:03d}")
    if os.path.exists(batch_path):
        try:
            batch_ds = load_dataset(batch_path)             # 这一步应该从本地读取数据
            final_datasets.append(batch_ds)
            print(f"✅ 加载批次 {i:03d}: {len(batch_ds):,} 条")
        except:
            print(f"❌ 无法加载批次 {i:03d}")

if final_datasets:
    final_ds = concatenate_datasets(final_datasets)
    final_ds.save_to_disk(OUTPUT_PATH)
    print(f"✅ 完成！总样本数: {len(final_ds):,}")
else:
    print("❌ 没有可合并的数据集")