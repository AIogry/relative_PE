import os
from datasets import load_dataset
from transformers import AutoTokenizer

# === 路径配置 ===
BASE_DIR = "/data/qijunrong/03-proj/PE/arxiv_data"
HF_CACHE_BASE = "/data/qijunrong/03-proj/PE/hf_cache"
TRAIN_DIR = os.path.join(BASE_DIR, "arxiv_train")
VAL_DIR = os.path.join(BASE_DIR, "arxiv_validation")

def download_via_mirror():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # 强制清理之前的锁文件，防止镜像下载被阻塞
    lock_path = os.path.join(HF_CACHE_BASE, "datasets/togethercomputer__red_pajama-data-1_t/arxiv/1.0.0")
    if os.path.exists(lock_path):
        import shutil
        print(f">>> 清理损坏的索引缓存: {lock_path}")
        shutil.rmtree(lock_path)

    print(f">>> 正在通过国内镜像下载 ArXiv 子集...")
    try:
        # 使用镜像站下载，trust_remote_code 必须开启以解析旧版脚本
        dataset = load_dataset(
            "togethercomputer/RedPajama-Data-1T", 
            "arxiv", 
            cache_dir=HF_CACHE_BASE,
            trust_remote_code=True,
            num_proc=16  # 开启多进程加速下载
        )
        
        # 数据切分
        full_train = dataset["train"]
        print(">>> 正在切分数据集 (2000条验证集)...")
        splits = full_train.train_test_split(test_size=2000, seed=6198)
        
        # 保存到磁盘
        splits["train"].save_to_disk(TRAIN_DIR)
        splits["test"].save_to_disk(VAL_DIR)
        print(f">>> 数据集成功保存至: {BASE_DIR}")
        
    except Exception as e:
        print(f">>> 下载失败: {e}")

if __name__ == "__main__":
    download_via_mirror()