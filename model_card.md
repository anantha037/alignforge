---
language: en
tags:
- phi-3
- qlora
- sft
- dpo
- alignforge
---

# AlignForge Phi-3-mini Model Card

## Model Description
This is an aligned version of `microsoft/Phi-3-mini-4k-instruct` using a full Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) pipeline (AlignForge) with QLoRA.

## Intended Uses
TBD

## Training Data
- SFT: `databricks/databricks-dolly-15k`
- DPO: `Anthropic/hh-rlhf`

## Training Procedure
The model was fine-tuned using QLoRA with 4-bit NormalFloat quantization. The training pipeline consists of an initial SFT phase followed by a DPO phase.

## Evaluation Results
TBD

## Limitations
TBD

## Citation
TBD
