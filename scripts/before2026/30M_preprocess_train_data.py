"""
Preprocess training data with multiple subset sizes.
"""
import os
import pickle
from datasets import load_from_disk
from OLMo.olmo.tokenizer import Tokenizer
from tqdm import tqdm
import random


def preprocess_train_caches(
    train_path, cache_dir="/data/qijunrong/03-proj/PE/preprocessed_data_1024", 
    max_seq_length=1024, train_subset_sizes=[500000, 1000000, 2000000, 5000000, 10000000]
):
    """
    Create multiple cache files for different training subset sizes.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Loading training dataset from: {train_path}")
    train_dataset = load_from_disk(train_path)
    
    print(f"Full train dataset size: {len(train_dataset)}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_pretrained("allenai/olmo-1b")
    
    def tokenize_and_chunk_subset(dataset, name, target_size, full_dataset_size, desc_suffix=""):
        cache_path = os.path.join(cache_dir, f"{name}_size_{target_size}_processed.pkl")
        
        if os.path.exists(cache_path):
            print(f"Cache already exists for {name} with size {target_size}: {cache_path}")
            return cache_path
        
        print(f"Preprocessing {name} dataset (target size: {target_size}){desc_suffix}...")
        processed_data = []
        
        # Get indices to process
        indices = list(range(full_dataset_size))
        if target_size < full_dataset_size:
            random.Random(42).shuffle(indices)
            indices = indices[:target_size]
        else:
            indices = indices[:full_dataset_size]  # Use all available data
        
        for idx in tqdm(indices, desc=f"Tokenizing {name} (size: {target_size})", total=len(indices)):
            example = dataset[idx]
            text = example.get("text", "")
            if text:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                # Chunk into sequences
                chunk_size = max_seq_length
                for start in range(0, len(tokens) - chunk_size + 1, chunk_size):
                    chunk = tokens[start:start + chunk_size]
                    if len(chunk) == chunk_size:  # Only keep full-length sequences
                        processed_data.append(chunk)
        
        print(f"Saving processed {name} data to cache: {cache_path}")
        print(f"Saved {len(processed_data)} sequences for target size {target_size}")
        with open(cache_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        return cache_path
    
    # Create cache for each training subset size
    cache_paths = {}
    for size in train_subset_sizes:
        if size > len(train_dataset):
            print(f"Warning: Requested size {size} exceeds available training data {len(train_dataset)}. Using full dataset.")
            actual_size = len(train_dataset)
        else:
            actual_size = size
            
        train_cache_path = tokenize_and_chunk_subset(
            train_dataset, "train", actual_size, len(train_dataset), 
            desc_suffix=f" (max: {size})"
        )
        cache_paths[actual_size] = train_cache_path
    
    return cache_paths


if __name__ == "__main__":
    train_path = "/data/qijunrong/03-proj/PE/c4_30M_train"
    
    # Create caches for different training sizes
    train_subset_sizes = [500000, 1000000, 2000000, 5000000, 10000000]
    cache_paths = preprocess_train_caches(
        train_path=train_path,
        cache_dir="/data/qijunrong/03-proj/PE/preprocessed_data",
        # max_seq_length=512,
        max_seq_length=1024,
        train_subset_sizes=train_subset_sizes
    )
    
    print("Training caching completed!")
    print("Available training subset caches:")
    for size, path in cache_paths.items():
        print(f"  Size {size}: {path}")