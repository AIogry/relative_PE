import os
from datasets import load_dataset
from transformers import AutoTokenizer

# === [修改] 路径改名为 wikitext 以示区分 ===
BASE_DIR = "/data/qijunrong/03-proj/PE/wikitext"

# 数据集保存路径
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "raw")
# Tokenizer 保存路径
LOCAL_TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

def download_and_save():
    os.makedirs(BASE_DIR, exist_ok=True)

    print(f">>> 1. Downloading WikiText-103 Dataset to {LOCAL_DATA_DIR}...")
    
    try:
        # === [关键修改] 使用 wikitext-103-v1，这是纯数据格式，不会报错 ===
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        dataset.save_to_disk(LOCAL_DATA_DIR)
        print(f">>> Success! Dataset saved to: {LOCAL_DATA_DIR}")
    except Exception as e:
        print(f">>> Error downloading WikiText: {e}")

    print(f"\n>>> 2. Downloading Tokenizer to {LOCAL_TOKENIZER_DIR}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.save_pretrained(LOCAL_TOKENIZER_DIR)
        print(f">>> Success! Tokenizer saved to: {LOCAL_TOKENIZER_DIR}")
    except Exception as e:
        print(f">>> Error downloading Tokenizer: {e}")

if __name__ == "__main__":
    download_and_save()