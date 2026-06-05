# AlignForge

AlignForge is a complete, end-to-end machine learning pipeline demonstrating how to fine-tune and align large language models using consumer-grade hardware. By implementing a sequence of Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) on the Phi-3-mini architecture, this project serves as a robust educational blueprint for researchers and developers looking to understand and apply modern LLM alignment techniques practically and efficiently.

## Pipeline Overview

The pipeline is divided into three primary stages:
1. **Base Model Integration:** Establishing the foundation with `microsoft/Phi-3-mini-4k-instruct`, setting up tokenization, and configuring 4-bit quantization.
2. **Supervised Fine-Tuning (SFT) with QLoRA:** Training the base model on the `databricks/databricks-dolly-15k` dataset using QLoRA to adapt the model's conversational and stylistic patterns without updating the full precision weights.
3. **Direct Preference Optimization (DPO):** Aligning the SFT-adapted model's outputs with human preferences using the `Anthropic/hh-rlhf` dataset. A frozen reference model is compared against the active trainable model to optimize response choices.

## Results

An LLM-as-a-judge evaluation (using LLaMA 3.3 70B via Groq on 50 prompts) yielded the following metrics:

| Model | Helpfulness | Safety | Std Dev |
|---|---|---|---|
| Base Phi-3-mini | 4.800 | 5.000 | 0.400 |
| SFT Model | 4.540 | 5.000 | 0.498 |
| DPO Model | 4.380 | 5.000 | 0.797 |

## Key Findings

During the DPO training phase, the primary alignment signal—the **reward margin**—showed significant improvement, increasing from 0.436 to 0.683 (+57%). This demonstrates that the model successfully learned to differentiate and prefer the "chosen" responses over the "rejected" ones. However, training for a single epoch on small datasets (like Dolly-15k) shifts the model's stylistic formatting more than it improves its foundational factual reasoning, leading to a slight drop in absolute helpfulness scores compared to Microsoft's heavily optimized base model. Notably, perfect safety scores were maintained across all stages.

## Repository Structure

```text
alignforge/
├── configs/
│   ├── dpo_config.yaml
│   └── sft_config.yaml
├── notebooks/
│   ├── 01_sft_training.ipynb
│   ├── 02_dpo_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_collator.py
│   │   ├── dataset_loader.py
│   │   └── test_data.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── eval_pipeline.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dpo_trainer.py
│   │   └── sft_trainer.py
│   └── utils/
│       ├── __init__.py
│       └── model_utils.py
├── .gitignore
├── model_card.md
├── README.md
└── requirements.txt
```

## How to Reproduce

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/alignforge.git
   cd alignforge
   ```
2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Notebooks:**
   To execute the pipeline, run the notebooks in sequence (ideally on Kaggle using a Tesla T4 16GB GPU):
   - Run `notebooks/01_sft_training.ipynb`
   - Run `notebooks/02_dpo_training.ipynb`
   - Run `notebooks/03_evaluation.ipynb`

## Models on Hugging Face Hub
- SFT Model: [ananthan7703/alignforge-phi3-sft](https://huggingface.co/ananthan7703/alignforge-phi3-sft)
- DPO Model: [ananthan7703/alignforge-phi3-dpo](https://huggingface.co/ananthan7703/alignforge-phi3-dpo)

## Tech Stack
- Base Model: `microsoft/Phi-3-mini-4k-instruct`
- Libraries: Transformers, PEFT, TRL, W&B
- Techniques: QLoRA, LoRA, DPO
- Evaluation: Groq API (LLaMA 3.3 70B as judge)

## Key Learnings
1. **Masking is Crucial:** Implementing custom label masking in the SFT data collator prevents the model from being penalized for the instruction tokens, ensuring it learns to generate responses rather than regurgitating prompts.
2. **Alignment vs Helpfulness:** DPO successfully aligns model preferences (proven by a 57% reward margin increase) without necessarily improving raw helpfulness. Fine-tuning a highly optimized base model on generic instruct data can actually degrade general task performance if not carefully balanced.
3. **Hardware Constraints:** Orchestrating a DPO pipeline requires keeping both a trainable model and a frozen reference model in VRAM simultaneously. Using 4-bit quantization and aggressive gradient checkpointing is absolutely mandatory to fit this workload onto a single 16GB GPU like the Tesla T4.
