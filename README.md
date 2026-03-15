# Fine-Tuning-Flan-T5-with-LoRa-PEFT-
LoRA (Low-Rank Adaptation) is a PEFT method that accelerates text summarization by adding tiny, trainable matrices to a frozen base model. Instead of updating millions of parameters, it learns specific summary patterns, reducing VRAM usage while maintaining high-quality, concise outputs.
# Project: Fine-Tuning Flan-T5 with LoRA (PEFT)

This project demonstrates how to use Parameter-Efficient Fine-Tuning (PEFT) 
via the Low-Rank Adaptation (LoRA) method to specialize a Google Flan-T5 model
for a specific text summarization task.

## Prerequisites
- Python 3.x
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- Hugging Face Datasets

## Overview of Stages

### STAGE 1: Environment Setup
Load the necessary libraries and check for GPU availability to ensure 
efficient training.

### STAGE 2: Model & Tokenizer Loading
- Base Model: 'google/flan-t5-base'
- Initialize the tokenizer and the Sequence-to-Sequence Language Model.
- Perform an initial inference test to see the "Base Model" output before tuning.

### STAGE 3: LoRA Configuration
Define the Low-Rank Adaptation (LoRA) settings:
- Rank (r): 8
- Alpha: 32
- Target Modules: q, v (Attention layers)
- Dropout: 0.05
The base model is frozen, and LoRA adapters are injected.

### STAGE 4: Data Preparation- printing the inputs of peft model before finetuning
- Input: A custom prompt (e.g., "summary: In the heart of the golden valley...")
- Tokenization: Prepare the input and target tensors for the model.

### STAGE 5: Training (Fine-Tuning)
- Optimizer: AdamW (learning rate: 1e-3)
- Process: The model undergoes 30 training steps where it learns the 
  specific style and content of the target summary.

### STAGE 6: Final Inference & Comparison
- The model is set to evaluation mode.
- A final generation is performed to compare the "Base Result" with the 
  "Tuned Result."
- (Optional) The LoRA adapters can be merged back into the base model 
  for zero-latency deployment.

## Key Performance Benefits
By using PEFT/LoRA, we only update a tiny fraction (<1%) of the total 
parameters, significantly reducing VRAM requirements while achieving 
specialized performance on the target task.
