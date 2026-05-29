"""
Data collator module for AlignForge.
"""
import logging
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional

logger = logging.getLogger(__name__)

class SFTDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Encode '<|assistant|>' to find it in the input_ids sequence
        self.assistant_tokens = self.tokenizer.encode("<|assistant|>", add_special_tokens=False)
        logger.info(f"Initialized collator. Assistant tokens encoded as: {self.assistant_tokens}")

    def __call__(self, batch: list) -> dict:
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        
        # Get pad token id, handle cases where tokenizer might not have it set
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        # Pad on the right
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=pad_token_id
        )
        
        attention_mask = (padded_input_ids != pad_token_id).long()
        labels = padded_input_ids.clone()
        
        # Mask everything before and including the assistant token sequence
        for i in range(labels.size(0)):
            sample_ids = labels[i].tolist()
            
            assistant_start_idx = -1
            assistant_len = len(self.assistant_tokens)
            
            for j in range(len(sample_ids) - assistant_len + 1):
                if sample_ids[j:j+assistant_len] == self.assistant_tokens:
                    assistant_start_idx = j
                    break
                    
            if assistant_start_idx != -1:
                # Mask up to and including the assistant tokens
                labels[i, :assistant_start_idx + assistant_len] = -100
            else:
                logger.warning(f"Assistant token sequence not found in sample {i}. Masking everything.")
                labels[i, :] = -100
                
        # Set all padding positions in labels to -100
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
    def verify_masking(self, batch_item: dict):
        logger.info("=== Masking Verification ===")
        labels = batch_item["labels"]
        input_ids = batch_item["input_ids"]
        
        if len(labels.shape) > 1:
            labels = labels[0]
            input_ids = input_ids[0]
            
        masked_count = (labels == -100).sum().item()
        unmasked_count = (labels != -100).sum().item()
        
        logger.info(f"Masked tokens (-100): {masked_count}")
        logger.info(f"Unmasked (label) tokens: {unmasked_count}")
        
        unmasked_indices = torch.where(labels != -100)[0]
        if len(unmasked_indices) > 0:
            first_unmasked = unmasked_indices[:20]
            first_unmasked_tokens = input_ids[first_unmasked]
            decoded = self.tokenizer.decode(first_unmasked_tokens)
            logger.info(f"First 20 unmasked tokens decoded: {repr(decoded)}")
        else:
            logger.warning("No unmasked tokens found in this sample!")
