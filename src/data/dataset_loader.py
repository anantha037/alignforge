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

class HHRLHFDatasetLoader:
    def __init__(self, config_path: str):
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.dataset_name = self.config["dataset"]["name"]
        self.max_samples = self.config["dataset"].get("max_samples")
        self.max_length = self.config["dpo"]["max_length"]
        self.max_prompt_length = self.config["dpo"]["max_prompt_length"]
        
    def _extract_prompt_and_responses(self, sample: dict) -> dict:
        chosen_str = sample.get("chosen", "")
        rejected_str = sample.get("rejected", "")
        
        assistant_idx_chosen = chosen_str.rfind("Assistant:")
        assistant_idx_rejected = rejected_str.rfind("Assistant:")
        
        if assistant_idx_chosen != -1 and assistant_idx_rejected != -1:
            prompt = chosen_str[:assistant_idx_chosen].strip()
            chosen_response = chosen_str[assistant_idx_chosen + len("Assistant:"):].strip()
            rejected_response = rejected_str[assistant_idx_rejected + len("Assistant:"):].strip()
            
            if prompt.startswith("Human:"):
                prompt = prompt[len("Human:"):].strip()
                
            sample["prompt"] = prompt
            sample["chosen_response"] = chosen_response
            sample["rejected_response"] = rejected_response
        else:
            sample["prompt"] = ""
            sample["chosen_response"] = ""
            sample["rejected_response"] = ""
            
        return sample
        
    def _format_for_dpo(self, sample: dict) -> dict:
        prompt = sample.get("prompt", "")
        chosen_response = sample.get("chosen_response", "")
        rejected_response = sample.get("rejected_response", "")
        
        sample["prompt"] = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        sample["chosen"] = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{chosen_response}<|end|>"
        sample["rejected"] = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{rejected_response}<|end|>"
        
        return sample
        
    def load_and_prepare(self, tokenizer) -> dict:
        logger.info(f"Loading dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, split="train")
        
        if self.max_samples is not None:
            dataset = dataset.select(range(min(len(dataset), self.max_samples)))
            
        logger.info("Extracting prompt and responses")
        dataset = dataset.map(self._extract_prompt_and_responses, desc="Extracting texts")
        
        initial_count = len(dataset)
        
        logger.info("Filtering out empty samples")
        dataset = dataset.filter(
            lambda x: bool(x.get("prompt")) and bool(x.get("chosen_response")) and bool(x.get("rejected_response")),
            desc="Filtering empty fields"
        )
        
        filtered_count = len(dataset)
        logger.info(f"Removed {initial_count - filtered_count} samples due to empty fields")
        
        logger.info("Applying Phi-3 DPO formatting")
        dataset = dataset.map(self._format_for_dpo, desc="Formatting for DPO")
        
        logger.info("Splitting dataset into train/eval (95/5)")
        split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
        
        logger.info(f"Train size: {len(split_dataset['train'])}")
        logger.info(f"Eval size: {len(split_dataset['test'])}")
        
        return {
            "train": split_dataset["train"],
            "eval": split_dataset["test"]
        }
        
    def get_dataset_stats(self, dataset_dict: dict):
        train_ds = dataset_dict["train"]
        eval_ds = dataset_dict["eval"]
        
        logger.info("=== DPO Dataset Statistics ===")
        logger.info(f"Train size: {len(train_ds)}")
        logger.info(f"Eval size: {len(eval_ds)}")
        
        prompt_lens = [len(x["prompt"]) for x in train_ds]
        chosen_lens = [len(x["chosen_response"]) for x in train_ds]
        rejected_lens = [len(x["rejected_response"]) for x in train_ds]
        
        diffs = [abs(c - r) for c, r in zip(chosen_lens, rejected_lens)]
        
        if prompt_lens:
            logger.info(f"Average prompt length (chars): {np.mean(prompt_lens):.2f}")
            logger.info(f"Average chosen response length (chars): {np.mean(chosen_lens):.2f}")
            logger.info(f"Average rejected response length (chars): {np.mean(rejected_lens):.2f}")
            logger.info(f"Average length difference (chars): {np.mean(diffs):.2f}")
