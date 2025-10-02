# Qwen2.5-VL Fine-tuning: Complete VLM Learning Pipeline

A comprehensive tutorial project for understanding and fine-tuning Vision-Language Models (VLMs) using Qwen2.5-VL. This repository contains a complete pipeline from model loading to fine-tuning and evaluation, designed for educational purposes.

## Overview

This project demonstrates:
- Loading and using pre-trained vision-language models
- Efficient fine-tuning with LoRA (Low-Rank Adaptation)
- Creating custom VQA datasets
- Evaluation and comparison of model performance

**Key Achievement**: Fine-tuned a 3.76B parameter model by training only 0.196% of parameters (7.4M) using LoRA.

## Requirements

### Hardware
- GPU with 16GB+ VRAM (tested on Tesla T4)
- 12GB+ system RAM
- ~10GB disk space for model weights

### Software
```bash
transformers>=4.45.0
torch>=2.0.0
accelerate>=0.26.0
peft>=0.8.0
pillow>=9.0.0
datasets>=2.14.0
# Clone repository
git clone https://github.com/yourusername/qwen-vl-finetuning
cd qwen-vl-finetuning

# Install dependencies
pip install -r requirements.txt

# Verify GPU access
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
qwen-vl-finetuning/
├── notebooks/
│   └── complete_pipeline.ipynb    # Full tutorial notebook
├── src/
│   ├── model_loader.py           # Model loading utilities
│   ├── dataset.py                # Custom dataset classes
│   ├── train.py                  # Training configuration
│   └── evaluate.py               # Evaluation scripts
├── data/
│   └── custom_vqa/               # Sample training data
├── outputs/
│   └── checkpoints/              # Saved models
└── README.md

Architecture Overview
Input Image → Vision Encoder (EVA-CLIP)
                    ↓
            Vision Tokens (variable length)
                    ↓
            Adapter Layer (MLP)
                    ↓
    Qwen2 Language Model (3.76B params)
                    ↓
            Generated Text Output
