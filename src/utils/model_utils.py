"""
Model utilities module for AlignForge.
"""
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pathlib import Path

logger = logging.getLogger(__name__)

def load_quantized_model(config: dict) -> tuple:
    quant_settings = config['quantization']
    
    logger.info("Setting up BitsAndBytesConfig")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_settings.get('load_in_4bit', True),
        bnb_4bit_quant_type=quant_settings.get('bnb_4bit_quant_type', 'nf4'),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=quant_settings.get('bnb_4bit_use_double_quant', True)
    )
    
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.info(f"VRAM usage before loading: {mem_alloc:.2f} GB")
    
    model_name = config['model']['name']
    logger.info(f"Loading model {model_name} with quantization config")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True
    )
    
    model.config.use_cache = False
    
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'
    
    if torch.cuda.is_available():
        mem_alloc_after = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.info(f"VRAM usage after loading: {mem_alloc_after:.2f} GB")
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters count: {trainable_params}")
    
    return model, tokenizer

def apply_lora(model, config: dict):
    lora_settings = config['lora']
    
    logger.info("Preparing model for kbit training")
    model = prepare_model_for_kbit_training(model)
    
    logger.info("Setting up LoraConfig")
    lora_config = LoraConfig(
        r=lora_settings.get('r', 16),
        lora_alpha=lora_settings.get('lora_alpha', 32),
        lora_dropout=lora_settings.get('lora_dropout', 0.05),
        bias=lora_settings.get('bias', 'none'),
        task_type=lora_settings.get('task_type', 'CAUSAL_LM'),
        target_modules=lora_settings.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
    )
    
    logger.info("Applying PEFT wrapper")
    model = get_peft_model(model, lora_config)
    
    logger.info("Trainable parameters summary:")
    model.print_trainable_parameters()
    
    return model

def get_vram_usage() -> dict:
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. VRAM usage is 0.")
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}
        
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free = total - reserved
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free
    }

def save_model(model, tokenizer, output_dir: str, merge_weights: bool = False):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving tokenizer to {output_dir}")
    tokenizer.save_pretrained(output_dir)
    
    if merge_weights:
        logger.info("Merging LoRA weights back into base model...")
        model = model.merge_and_unload()
        logger.info(f"Saving merged model to {output_dir}")
        model.save_pretrained(output_dir)
    else:
        logger.info(f"Saving LoRA adapter weights to {output_dir}")
        model.save_pretrained(output_dir)
        
    def get_size(path):
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024 ** 2)
        
    size_mb = get_size(out_path)
    logger.info(f"Output directory {output_dir} size: {size_mb:.2f} MB")
