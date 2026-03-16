import os
from datasets import load_dataset
from transformers import AutoTokenizer

# === [关键修改] 设置你的目标路径 ===
# 基础目录
BASE_DIR = "/data/qijunrong/03-proj/PE/pg19"

# 数据集保存路径
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "raw")
# Tokenizer 保存路径
LOCAL_TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

def download_and_save():
    # 确保目录存在
    os.makedirs(BASE_DIR, exist_ok=True)

    print(f">>> 1. Downloading PG-19 Dataset to {LOCAL_DATA_DIR}...")
    print("    (This handles the 'buying ingredients' part - saving raw text)")
    
    # 下载 PG-19 (deepmind/pg19)
    # 注意：PG-19 很大，如果只想快速测试，可以用 "wikitext"
    try:
        dataset = load_dataset("deepmind/pg19", trust_remote_code=True)
        dataset.save_to_disk(LOCAL_DATA_DIR)
        print(f">>> Success! Dataset saved to: {LOCAL_DATA_DIR}")
    except Exception as e:
        print(f">>> Error downloading PG-19: {e}")
        print(">>> Suggestion: Check internet connection or disk space.")

    print(f"\n>>> 2. Downloading Tokenizer to {LOCAL_TOKENIZER_DIR}...")
    # 下载 GPT-NeoX-20B 的 Tokenizer (这也是 OLMo 常用的标准)
    try:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.save_pretrained(LOCAL_TOKENIZER_DIR)
        print(f">>> Success! Tokenizer saved to: {LOCAL_TOKENIZER_DIR}")
    except Exception as e:
        print(f">>> Error downloading Tokenizer: {e}")

if __name__ == "__main__":
    download_and_save()