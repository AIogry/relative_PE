import os
import argparse
# [修改 1] 引入 disable_progress_bar 和 DownloadConfig
from datasets import load_dataset, disable_progress_bar, DownloadConfig
from transformers import AutoTokenizer
from release_utils import ensure_dir, repo_path


def download_and_save(base_dir: str, tokenizer_name: str):
    train_dir = os.path.join(base_dir, "arxiv_train")
    val_dir = os.path.join(base_dir, "arxiv_validation")
    local_tokenizer_dir = os.path.join(base_dir, "tokenizer")
    ensure_dir(base_dir)
    
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
        
        print(f">>> 正在保存 Train Split 至 {train_dir}...")
        splits["train"].save_to_disk(train_dir)
        
        print(f">>> 正在保存 Validation Split 至 {val_dir}...")
        splits["test"].save_to_disk(val_dir)
        
        print(">>> 数据集处理与保存成功！")
        
    except Exception as e:
        print(f">>> Error downloading or processing ArXiv: {e}")

    print(f"\n>>> 2. Downloading Tokenizer to {local_tokenizer_dir}...")
    try:
        # ⚠️ 显式授权运行 OLMo 的自定义代码
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tokenizer.save_pretrained(local_tokenizer_dir)
        print(f">>> Success! Tokenizer saved to: {local_tokenizer_dir}")
    except Exception as e:
        print(f">>> Error downloading Tokenizer: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=repo_path("data", "arxiv"))
    parser.add_argument("--tokenizer_name", type=str, default="allenai/olmo-1b")
    args = parser.parse_args()
    download_and_save(args.output_dir, args.tokenizer_name)
