# test_tokenizer.py
import os
os.environ["HF_HOME"] = "/data/qijunrong/03-proj/PE/hf_cache"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/olmo-1b")
print("✅ Tokenizer loaded successfully!")
print(tokenizer("Hello world"))