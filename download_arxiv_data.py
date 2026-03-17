import os
# [修改 1] 引入 disable_progress_bar 和 DownloadConfig
from datasets import load_dataset, disable_progress_bar, DownloadConfig
from transformers import AutoTokenizer

# === 路径配置 ===
BASE_DIR = "/data/qijunrong/03-proj/PE/arxiv_data"

# 为了适配 train.py，我们直接分为 train 和 validation 两个文件夹保存
TRAIN_DIR = os.path.join(BASE_DIR, "arxiv_train")
VAL_DIR = os.path.join(BASE_DIR, "arxiv_validation")
LOCAL_TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

def download_and_save():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # [修改 2] 彻底关闭 datasets 库烦人的进度条，保持日志纯净
    # disable_progress_bar()

    print(f">>> 1. Downloading RedPajama-ArXiv Dataset...")
    # print(f">>> (进度条已关闭以防止日志乱码，后台正在全力下载，请耐心等待...)")
    try:
        # [修改 3] 实例化一个下载配置，强制开启最严格的断点续传
        # dl_config = DownloadConfig(resume_download=True)

        # 下载 ArXiv 子集 (开启 32 进程加速)
        dataset = load_dataset(
            "togethercomputer/RedPajama-Data-1T", 
            "arxiv", 
            num_proc=8, 
            trust_remote_code=True,
            # download_config=dl_config  # 传入断点续传配置
        )
        
        # RedPajama 默认都在 train split 里
        full_train = dataset["train"]
        
        print(">>> 正在切分数据集 (抽取 2000 条作为验证集)...")
        # 随机切分，固定 seed 保证可复现
        splits = full_train.train_test_split(test_size=2000, seed=6198)
        
        print(f">>> 正在保存 Train Split 至 {TRAIN_DIR}...")
        splits["train"].save_to_disk(TRAIN_DIR)
        
        print(f">>> 正在保存 Validation Split 至 {VAL_DIR}...")
        splits["test"].save_to_disk(VAL_DIR)
        
        print(">>> 数据集处理与保存成功！")
        
    except Exception as e:
        print(f">>> Error downloading or processing ArXiv: {e}")

    print(f"\n>>> 2. Downloading Tokenizer to {LOCAL_TOKENIZER_DIR}...")
    try:
        # ⚠️ 显式授权运行 OLMo 的自定义代码
        tokenizer = AutoTokenizer.from_pretrained("allenai/olmo-1b", trust_remote_code=True)
        tokenizer.save_pretrained(LOCAL_TOKENIZER_DIR)
        print(f">>> Success! Tokenizer saved to: {LOCAL_TOKENIZER_DIR}")
    except Exception as e:
        print(f">>> Error downloading Tokenizer: {e}")

if __name__ == "__main__":
    download_and_save()