# NoneStory - JLPT N1 Learning Story Generator

**Current Status: Paused**  
**Key Words:** LLM x SFT x Language Education
**Key Challenge:** Improving client response speed, especially on mobile  
**What I Learned:**  
1. Text LLMs are more demanding in training cost, inference latency, and quality control  
2. Data cleaning and quality control are critical  
3. Hardware requirements are high (VRAM, stability, training time)

## Overview
This is a customized language model project for JLPT N1 vocabulary learning. A fine-tuned text-to-text model generates modern fiction stories to help users learn and remember N1 vocabulary through reading, combining storytelling with education.

## Demo / Models
- HuggingFace Space: https://huggingface.co/spaces/BBusagi/sft-nonestory-generator
- Model (train002): https://huggingface.co/BBusagi/sft-nonestory-train002
- Model (train001): https://huggingface.co/BBusagi/sft-nonestory-train001

## Environment
- Python 3.10
- Training env: `conda activate rinna-env`
- GPU monitor: `nvidia-smi -l 1`
- Unicode reference: https://blog.cloudnative.co.jp/23733/

## Current Status
- [Bug] Web deployment output not accessible
- [Task] Model output is unstable
- [Task] Long response time on phones / low-end PCs

## System Design
1. Extract JLPT N1 vocabulary
2. Use GPT-4 to generate high-quality corpus
3. Build training dataset
4. Full fine-tuning on rinna-medium

## Development Flow
### Step00 Environment setup & checks
- Script/Tool_check_api_connection.py

### Step01 Prepare corpus
- Script/Step1_generate_json.py
- Script/Tool_convert_jsonl_to_txt.mjs
- Script/Step1_prepare_training_data.py

### Step02 Train model (with checkpoints)
- Script/Tool_check_gpu_status.py
- Script/Step2_train_sft_model.py
- Script/Step2_save_model.py

### Step03 Model evaluation
- Script/Step3_infer_and_evaluate.py
- Script/Tool_run_model_inference.py
- Script/Step3_compare_multiple_models.py

## Dev Log
### Step01: chatGPT + rinna/small
- Environment setup
- OpenAI / rinna tests
- Batch calls and QC (1000 stories)
- Prompt tuning
- Corpus preprocessing and training JSON generation
- SFT training attempt
- Result: workable, style imitation succeeded; length control weak, overall shallow

### Step02: GPT-4 long-form corpus + rinna/medium
- Prompt and batch updates
- BUG: 90,000 enqueued tokens limit (4o -> 4omini)
- Corpus preprocessing and training JSON generation
- SFT training tests
- BUG: Full fine-tuning length inconsistency (parameter cleaning)
- Full fine-tuning (train2)
- BUG: Blue screen / VRAM overflow (changed PyTorch GPU call)
- Checkpointing with multi-threading
- ~90 minutes per run
- checkpoint-2880: strong performance, meets content quality  
  ![](readme/train1(2880).png)
- Resume training
- BUG: lm_head missing weights (pytorch_model.bin)
  - Adjusted save module + short-run training (tried LoRA), still occurred
  - Cause: older resume_from_checkpoint supports .bin only; new format is .safetensors
  - Fix: removed resume_from_checkpoint parameter (resolved)
- Full fine-tuning (train1)
- checkpoint-1500:
  - First test: model rolled back, short outputs, poor quality
  - Second test: strong performance, length improved  
  ![](readme/train2(1500).png)

### Step03: Frontend deployment
- Model packaging repo
- HuggingFace Space setup
- Gradio UI deployment
- Page updates and features
  - History view
  - Multi-model selection
  - Batch generation
  - Error handling
  - Manual delete

## TODO
### Model
- Training monitoring and integrity checks
- Estimate remaining training runs (e.g., "X runs left")
- Log step / duration / loss

### Evaluation
- Add rating system  
  Note: "density scoring" is better as post-generation scoring/filtering, not as a training objective
- Support text formatting (bold, etc.)
- Plot/structure control
- Later: if resources allow, try knowledge distillation + chain optimization for a lightweight model
- Later: build a "Japanese learning LLM" that takes JLPT level and outputs personalized text

### Frontend
- Streaming output (real-time)
- Task queue for multi-user polling
