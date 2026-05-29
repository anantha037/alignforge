"""
SFT trainer module for AlignForge.
"""
import logging
import yaml
import traceback
from pathlib import Path
from transformers import TrainingArguments
from trl import SFTTrainer
from src.data.dataset_loader import DollyDatasetLoader
from src.data.data_collator import SFTDataCollator
from src.utils.model_utils import load_quantized_model, apply_lora, get_vram_usage, save_model

logger = logging.getLogger(__name__)

class QLoRASFTTrainer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        logging.basicConfig(level=logging.INFO)
        
        self.output_dir = Path(self.config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self):
        logger.info("Setting up SFT Trainer...")
        self.model, self.tokenizer = load_quantized_model(self.config)
        
        self.model = apply_lora(self.model, self.config)
        
        vram_usage = get_vram_usage()
        logger.info(f"VRAM after model setup: {vram_usage}")
        
        loader = DollyDatasetLoader(self.config_path)
        self.dataset_dict = loader.load_and_prepare(self.tokenizer)
        
        self.collator = SFTDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.config["model"]["max_seq_length"]
        )
        
        logger.info(f"Setup complete. Train size: {len(self.dataset_dict['train'])}, Eval size: {len(self.dataset_dict['eval'])}")
        
    def build_trainer(self):
        logger.info("Building TrainingArguments and SFTTrainer")
        t_args = self.config['training']
        
        args = TrainingArguments(
            output_dir=t_args["output_dir"],
            num_train_epochs=t_args["num_train_epochs"],
            per_device_train_batch_size=t_args["per_device_train_batch_size"],
            per_device_eval_batch_size=t_args["per_device_eval_batch_size"],
            gradient_accumulation_steps=t_args["gradient_accumulation_steps"],
            gradient_checkpointing=t_args["gradient_checkpointing"],
            learning_rate=t_args["learning_rate"],
            lr_scheduler_type=t_args["lr_scheduler_type"],
            warmup_ratio=t_args["warmup_ratio"],
            weight_decay=t_args["weight_decay"],
            fp16=t_args["fp16"],
            logging_steps=t_args["logging_steps"],
            eval_steps=t_args["eval_steps"],
            save_steps=t_args["save_steps"],
            save_total_limit=t_args["save_total_limit"],
            load_best_model_at_end=t_args.get("load_best_model_at_end", False),
            report_to=t_args["report_to"],
            run_name=t_args["run_name"],
            dataloader_pin_memory=False,
            remove_unused_columns=False
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset_dict["train"],
            eval_dataset=self.dataset_dict["eval"],
            data_collator=self.collator,
            dataset_text_field="text",
        )
        logger.info("Trainer built successfully.")
        
    def train(self):
        vram_before = get_vram_usage()
        logger.info(f"VRAM usage before training: {vram_before}")
        
        logger.info("Starting training...")
        train_output = self.trainer.train()
        
        logger.info("Training complete.")
        metrics = train_output.metrics
        logger.info(f"Final training loss: {metrics.get('train_loss', 'N/A')}")
        
        eval_metrics = self.trainer.evaluate()
        logger.info(f"Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
        
        vram_after = get_vram_usage()
        logger.info(f"VRAM usage after training: {vram_after}")
        
        return train_output
        
    def save(self, merge_weights: bool = True):
        out_dir = self.config['training']['output_dir']
        logger.info(f"Saving model to {out_dir}")
        save_model(self.model, self.tokenizer, out_dir, merge_weights)
        logger.info(f"Successfully saved to {out_dir}")
        
    def run(self):
        try:
            self.setup()
            self.build_trainer()
            self.train()
            self.save()
        except Exception as e:
            logger.error("Error running SFT pipeline:")
            logger.error(traceback.format_exc())
            raise e
