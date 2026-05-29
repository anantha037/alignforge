"""
Sanity check for dataset loader and collator.
"""
import logging
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional

import sys
import os
from dataset_loader import DollyDatasetLoader
from data_collator import SFTDataCollator

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting sanity check...")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Handle path regardless of where it's executed
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "configs", "sft_config.yaml")
    
    loader = DollyDatasetLoader(config_path)
    dataset_dict = loader.load_and_prepare(tokenizer)
    
    loader.get_dataset_stats(dataset_dict)
    
    collator = SFTDataCollator(tokenizer, max_length=loader.max_seq_length)
    
    train_ds = dataset_dict["train"]
    samples = [train_ds[i] for i in range(min(4, len(train_ds)))]
    
    batch = collator(samples)
    
    collator.verify_masking(batch)
    
    print("\nSANITY CHECK PASSED")

if __name__ == "__main__":
    main()
