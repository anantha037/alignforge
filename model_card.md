---
language: en
tags:
- phi-3
- qlora
- sft
- dpo
- alignforge
---

# AlignForge Phi-3-mini DPO

## Model Description
AlignForge Phi-3-mini DPO is an aligned version of `microsoft/Phi-3-mini-4k-instruct` using a complete Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) pipeline. The model demonstrates the integration of QLoRA techniques to efficiently shift response styles and align preferences.

## Intended Uses
This model is intended strictly for research and educational purposes. It serves as a demonstration of a full end-to-end SFT to DPO alignment pipeline using limited hardware resources.

## Training Pipeline Overview
1. **SFT Phase:** Fine-tuning with QLoRA on the `databricks/databricks-dolly-15k` dataset.
2. **DPO Phase:** Preference alignment using the `Anthropic/hh-rlhf` dataset.

### SFT Training Details
- **Dataset:** `databricks/databricks-dolly-15k` (13,266 training samples, 1,475 eval samples)
- **Method:** QLoRA 4-bit quantization, LoRA r=16, alpha=32, target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Hyperparameters:** 1 epoch, batch size 2, gradient accumulation 8, learning rate 2e-4, cosine scheduler, warmup ratio 0.03
- **Validation Loss Progression:** 1.320 → 1.312 → 1.306 → 1.305

### DPO Training Details
- **Dataset:** `Anthropic/hh-rlhf` (4,749 training samples, 250 eval samples)
- **Hyperparameters:** 1 epoch, batch size 1, gradient accumulation 8, learning rate 5e-5, beta 0.1, loss type sigmoid
- **Reward Accuracy Progression:** 0.680 → 0.696
- **Reward Margin Progression:** 0.436 → 0.683
- **Final DPO Loss:** 0.6118

## Evaluation Results
Evaluation was conducted using an LLM-as-a-judge (LLaMA 3.3 70B via Groq) across 50 prompts to measure helpfulness and safety.

| Model | Helpfulness (1-5) | Safety (1-5) | Standard Deviation |
|---|---|---|---|
| Base Phi-3-mini | 4.800 | 5.000 | 0.400 |
| SFT Model | 4.540 | 5.000 | 0.498 |
| DPO Model | 4.380 | 5.000 | 0.797 |

## Limitations
A single epoch of SFT on the Dolly-15k dataset primarily acts to shift the conversational and formatting style of the model rather than dramatically improving factual quality or underlying reasoning. Consequently, the raw helpfulness score slightly decreases compared to the highly optimized base model. Furthermore, DPO training substantially improves preference alignment metrics during the training phase (e.g., reward margin increased by 57%), but this alignment improvement does not explicitly translate to higher raw helpfulness scores measured by the judge. All models, however, maintained perfect safety scores throughout the pipeline. 

## Hardware Setup
- NVIDIA Tesla T4 16GB (Kaggle free tier)
