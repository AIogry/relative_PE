"""
Preprocess validation data with 50k samples.
放在03-proj下运行
"""
import os
import pickle
from datasets import load_from_disk
from OLMo.olmo.tokenizer import Tokenizer
from tqdm import tqdm
import random


def preprocess_validation_cache(
    val_path, cache_dir="/data/qijunrong/03-proj/PE/preprocessed_data", 
    max_seq_length=512, val_subset_size=50000
):
    """
    Create cache file for validation data with 50k samples.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Loading validation dataset from: {val_path}")
    val_dataset = load_from_disk(val_path)
    
    print(f"Full validation dataset size: {len(val_dataset)}")
    
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
    
    # Create validation cache with 50k samples
    val_cache_path = tokenize_and_chunk_subset(
        val_dataset, "val", val_subset_size, len(val_dataset), 
        desc_suffix=f" (max: {val_subset_size})"
    )
    
    return val_cache_path


if __name__ == "__main__":
    val_path = "/data/qijunrong/03-proj/PE/c4_30M_validation"
    
    # Create cache for validation set
    val_cache_path = preprocess_validation_cache(
        val_path=val_path,
        cache_dir="/data/qijunrong/03-proj/PE/preprocessed_data",
        max_seq_length=512,
        val_subset_size=50000
    )
    
    print("Validation caching completed!")
    print(f"Validation cache: {val_cache_path}")