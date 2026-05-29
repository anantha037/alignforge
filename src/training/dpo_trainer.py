"""
DPO trainer module for AlignForge.
"""
import logging
import yaml
import torch
import traceback
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from src.data.dataset_loader import HHRLHFDatasetLoader
from src.utils.model_utils import apply_lora, get_vram_usage, save_model

logger = logging.getLogger(__name__)

class DPOAlignmentTrainer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        logging.basicConfig(level=logging.INFO)
        
        self.output_dir = Path(self.config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.ref_model = None
        self.trainer = None

    def setup(self):
        logger.info("Setting up models for DPO...")
        sft_path = self.config['model']['sft_model_path']
        quant_settings = self.config['quantization']
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_settings.get('load_in_4bit', True),
            bnb_4bit_quant_type=quant_settings.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=quant_settings.get('bnb_4bit_use_double_quant', True)
        )
        
        logger.info(f"Loading SFT model from {sft_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            sft_path,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True
        )
        self.model.config.use_cache = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(sft_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'right'
        
        logger.info("Applying LoRA to training model")
        self.model = apply_lora(self.model, self.config)
        
        logger.info(f"Loading reference model from {sft_path}")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            sft_path,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        loader = HHRLHFDatasetLoader(self.config_path)
        self.dataset_dict = loader.load_and_prepare(self.tokenizer)
        
        vram = get_vram_usage()
        logger.info(f"VRAM usage after model setup: {vram}")
        
    def build_trainer(self):
        logger.info("Building DPOConfig and DPOTrainer")
        
        args_dict = {**self.config['training'], **self.config['dpo']}
        
        args = DPOConfig(
            output_dir=args_dict.get("output_dir"),
            num_train_epochs=args_dict.get("num_train_epochs"),
            per_device_train_batch_size=args_dict.get("per_device_train_batch_size"),
            per_device_eval_batch_size=args_dict.get("per_device_eval_batch_size"),
            gradient_accumulation_steps=args_dict.get("gradient_accumulation_steps"),
            gradient_checkpointing=args_dict.get("gradient_checkpointing"),
            learning_rate=args_dict.get("learning_rate"),
            lr_scheduler_type=args_dict.get("lr_scheduler_type"),
            warmup_ratio=args_dict.get("warmup_ratio"),
            fp16=args_dict.get("fp16"),
            logging_steps=args_dict.get("logging_steps"),
            eval_steps=args_dict.get("eval_steps"),
            save_steps=args_dict.get("save_steps"),
            save_total_limit=args_dict.get("save_total_limit"),
            report_to=args_dict.get("report_to"),
            run_name=args_dict.get("run_name"),
            beta=args_dict.get("beta", 0.1),
            loss_type=args_dict.get("loss_type", "sigmoid"),
            max_prompt_length=args_dict.get("max_prompt_length", 512),
            max_length=args_dict.get("max_length", 1024),
            remove_unused_columns=False
        )
        
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=args,
            train_dataset=self.dataset_dict["train"],
            eval_dataset=self.dataset_dict["eval"],
            tokenizer=self.tokenizer
        )
        
        logger.info("DPO Trainer built successfully.")
        
    def train(self):
        vram_before = get_vram_usage()
        logger.info(f"VRAM before training: {vram_before}")
        
        logger.info("Starting DPO training...")
        train_output = self.trainer.train()
        
        metrics = train_output.metrics
        logger.info(f"Final training loss: {metrics.get('train_loss', 'N/A')}")
        
        eval_metrics = self.trainer.evaluate()
        logger.info(f"Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
        
        vram_after = get_vram_usage()
        logger.info(f"VRAM after training: {vram_after}")
        
        return train_output
        
    def save(self, merge_weights: bool = True):
        out_dir = self.config['training']['output_dir']
        logger.info(f"Saving DPO model to {out_dir}")
        save_model(self.model, self.tokenizer, out_dir, merge_weights)
        logger.info(f"Successfully saved to {out_dir}")
        
    def run(self):
        try:
            self.setup()
            self.build_trainer()
            self.train()
            self.save()
        except Exception as e:
            logger.error("Error running DPO pipeline:")
            logger.error(traceback.format_exc())
            raise e
