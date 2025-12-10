"""
Load C4 dataset from mixed directory containing both arrow files and batch directories.
"""
import os
import glob
from datasets import load_dataset
from tqdm import tqdm

def load_c4_from_mixed_dir(raw_data_path):
    """
    Load C4 dataset from directory containing arrow files and other files/directories.
    """
    print(f"Loading raw data from: {raw_data_path}")
    
    # Get only arrow files, excluding any files in batch directories
    arrow_files = []
    
    # Method 1: Find all .arrow files in the main directory (not in subdirectories)
    for file in os.listdir(raw_data_path):
        if file.endswith('.arrow') and os.path.isfile(os.path.join(raw_data_path, file)):
            arrow_files.append(os.path.join(raw_data_path, file))
    
    print(f"Found {len(arrow_files)} arrow files in main directory")
    
    if not arrow_files:
        # Method 2: If no arrow files in main directory, look for them in batch directories
        print("No arrow files found in main directory, searching in batch directories...")
        for batch_dir in os.listdir(raw_data_path):
            batch_path = os.path.join(raw_data_path, batch_dir)
            if os.path.isdir(batch_path) and batch_dir.startswith('batch_'):
                batch_arrow_files = glob.glob(os.path.join(batch_path, '*.arrow'))
                arrow_files.extend(batch_arrow_files)
        
        print(f"Found {len(arrow_files)} arrow files in batch directories")
    
    if not arrow_files:
        raise ValueError(f"No arrow files found in {raw_data_path}")
    
    # Sort arrow files to ensure consistent ordering
    arrow_files.sort()
    
    print(f"Loading {len(arrow_files)} arrow files...")
    dataset = load_dataset("arrow", data_files=arrow_files, split="train")
    
    return dataset

def split_30m_c4_dataset(raw_data_path, output_base_path, train_ratio=0.90, val_ratio=0.05, test_ratio=0.01):
    """
    Load raw C4 dataset and split into train/validation/test sets.
    """
    # Load dataset using the custom function
    dataset = load_c4_from_mixed_dir(raw_data_path)
    
    print(f"Total dataset size: {len(dataset)}")
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    
    print(f"Target splits - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
    print(f"Actual splits - Train: {train_size}, Validation: {val_size}, Test: {total_size - train_size - val_size}")
    
    # Shuffle the dataset
    import random
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    
    # Split indices
    train_indices = indices[:train_size]
    remaining_indices = indices[train_size:]
    val_indices = remaining_indices[:val_size]
    test_indices = remaining_indices[val_size:]
    
    print(f"Actual split - Train: {len(train_indices)}, Validation: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create datasets
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)
    
    # Create output directories
    train_output = os.path.join(output_base_path, "c4_30M_train")
    val_output = os.path.join(output_base_path, "c4_30M_validation") 
    test_output = os.path.join(output_base_path, "c4_30M_test")
    
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)
    
    # Save datasets
    print(f"Saving train dataset to: {train_output}")
    train_dataset.save_to_disk(train_output)
    print(f"Train dataset saved with {len(train_dataset)} samples")
    
    print(f"Saving validation dataset to: {val_output}")
    val_dataset.save_to_disk(val_output)
    print(f"Validation dataset saved with {len(val_dataset)} samples")
    
    print(f"Saving test dataset to: {test_output}")
    test_dataset.save_to_disk(test_output)
    print(f"Test dataset saved with {len(test_dataset)} samples")
    
    print("Dataset splitting completed!")
    
    return train_output, val_output, test_output

if __name__ == "__main__":
    raw_data_path = "/data/qijunrong/03-proj/PE/c4_30M_raw"
    output_base_path = "/data/qijunrong/03-proj/PE"
    
    train_path, val_path, test_path = split_30m_c4_dataset(
        raw_data_path=raw_data_path,
        output_base_path=output_base_path,
        train_ratio=0.90,
        val_ratio=0.05,
        test_ratio=0.01
    )
    
    print(f"Train dataset: {train_path}")
    print(f"Validation dataset: {val_path}")
    print(f"Test dataset: {test_path}")