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
from dataset_loader import DollyDatasetLoader, HHRLHFDatasetLoader
from data_collator import SFTDataCollator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_utils import load_quantized_model, get_vram_usage

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
    
    print("\n--- MODEL UTILS TEST ---")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    try:
        model, tok = load_quantized_model(config)
        if torch.cuda.is_available():
            vram = get_vram_usage()
            print(f"VRAM Usage: {vram}")
    except RuntimeError:
        print("GPU NOT AVAILABLE LOCALLY - SKIPPING MODEL LOAD TEST")
    except Exception as e:
        if "GPU NOT AVAILABLE" in str(e) or "CUDA" in str(e) or "bitsandbytes" in str(e):
            print("GPU NOT AVAILABLE LOCALLY - SKIPPING MODEL LOAD TEST")
        else:
            raise e
            
    print("\n--- DPO DATASET TEST ---")
    dpo_config_path = os.path.join(base_dir, "configs", "dpo_config.yaml")
    dpo_loader = HHRLHFDatasetLoader(dpo_config_path)
    dpo_dataset_dict = dpo_loader.load_and_prepare(tokenizer)
    
    dpo_loader.get_dataset_stats(dpo_dataset_dict)
    
    first_sample = dpo_dataset_dict["train"][0]
    print("\nFirst sample prompt (truncated):")
    print(first_sample["prompt"][:300] + ("..." if len(first_sample["prompt"]) > 300 else ""))
    print("\nFirst sample chosen (truncated):")
    print(first_sample["chosen"][:300] + ("..." if len(first_sample["chosen"]) > 300 else ""))
    print("\nFirst sample rejected (truncated):")
    print(first_sample["rejected"][:300] + ("..." if len(first_sample["rejected"]) > 300 else ""))
    
    print("\nDPO DATASET TEST PASSED")

if __name__ == "__main__":
    main()
