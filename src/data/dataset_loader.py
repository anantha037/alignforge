"""
Dataset loader module for AlignForge.
"""
import logging
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

class DollyDatasetLoader:
    def __init__(self, config_path: str):
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.dataset_name = self.config["dataset"]["name"]
        self.train_split = self.config["dataset"]["train_split"]
        self.max_seq_length = self.config["model"]["max_seq_length"]
        self.max_samples = self.config["dataset"].get("max_samples")
        
    def _format_prompt(self, sample: dict) -> dict:
        instruction = sample.get("instruction", "").strip()
        context = sample.get("context", "").strip()
        response = sample.get("response", "").strip()
        
        if context:
            text = f"<|user|>\n{instruction}\n\n{context}<|end|>\n<|assistant|>\n{response}<|end|>"
        else:
            text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"
            
        sample["text"] = text
        return sample
        
    def load_and_prepare(self, tokenizer: AutoTokenizer) -> dict:
        logger.info(f"Loading dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, split="train")
        
        if self.max_samples is not None:
            dataset = dataset.select(range(min(len(dataset), self.max_samples)))
            
        logger.info("Applying Phi-3 instruction format")
        dataset = dataset.map(self._format_prompt, desc="Formatting prompts")
        
        logger.info("Tokenizing dataset")
        def tokenize_func(examples):
            return tokenizer(examples["text"], truncation=True, max_length=self.max_seq_length)
            
        dataset = dataset.map(tokenize_func, batched=True, desc="Tokenizing")
        
        initial_count = len(dataset)
        
        logger.info("Filtering samples with less than 10 tokens")
        dataset = dataset.filter(lambda x: len(x["input_ids"]) >= 10, desc="Filtering < 10 tokens")
        after_min_filter = len(dataset)
        logger.info(f"Removed {initial_count - after_min_filter} samples with < 10 tokens")
        
        logger.info(f"Filtering samples truncated at max_length ({self.max_seq_length})")
        dataset = dataset.filter(lambda x: len(x["input_ids"]) < self.max_seq_length, desc="Filtering truncated")
        after_max_filter = len(dataset)
        logger.info(f"Removed {after_min_filter - after_max_filter} samples with length >= max_length ({self.max_seq_length})")
        
        logger.info(f"Splitting dataset into train/eval with ratio {self.train_split}")
        split_dataset = dataset.train_test_split(
            train_size=self.train_split, 
            seed=42
        )
        
        return {
            "train": split_dataset["train"],
            "eval": split_dataset["test"]
        }
        
    def get_dataset_stats(self, dataset_dict: dict):
        train_ds = dataset_dict["train"]
        eval_ds = dataset_dict["eval"]
        
        lengths = [len(x) for x in train_ds["input_ids"]] + [len(x) for x in eval_ds["input_ids"]]
        
        logger.info("=== Dataset Statistics ===")
        logger.info(f"Train size: {len(train_ds)}")
        logger.info(f"Eval size: {len(eval_ds)}")
        if lengths:
            logger.info(f"Min length: {np.min(lengths)}")
            logger.info(f"Max length: {np.max(lengths)}")
            logger.info(f"Mean length: {np.mean(lengths):.2f}")
            logger.info(f"50th percentile: {np.percentile(lengths, 50)}")
            logger.info(f"90th percentile: {np.percentile(lengths, 90)}")
            logger.info(f"95th percentile: {np.percentile(lengths, 95)}")
