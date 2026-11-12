# PxCaption (Windows 64-bit, Python 3.13, CUDA)
<BR>
A windows desktop GUI that captions images and video using Qwen2.5/3-VL models. Pluggable backends (vLLM / Transformers) with torch compile, quantization, attention and batching options for fast local inference.
<BR>
<BR>
<div align="center">
  <img src="media/PxCaption.png" alt="PxCaption">
</div>

## Model and Media support

- Models - Qwen2.5/3-VL<BR>
- Image formats - .png, .jpg, .jpeg, .bmp, .gif, .webp<BR>
- Video formats - .mp4, .mov, .avi, .webm, .mkv, .flv<BR>


## Features

- Backend - vLLM, Transformers (With torch compile and disk caching), Transformers.<BR>
- Quantization - vLLM: None / AWQ / GPTQ, Transformers: None / 8-bit / 4-bit NF4 / 4-bit FP4.<BR>
- Attention - FA2, SDPA, Eager.<BR>
- Context - Preset selection with auto-context window resizing (vLLM).<BR>
- KV Cache - FP8 quantization (vLLM).<BR>
- Batching - Batch sizes from 1-16 (vLLM).<BR>
- Resolution - Aspect-ratio-preserving pixel budgets.<BR>


## QoL

- Multi-GPU.
- Model store - Import models into a virtual store for easy management.<BR>
- Model hot-swap — Prevents redundant reloads when only unrelated settings change.<BR>
- Attention fallback chain — FA2 → SDPA → Eager.<BR>
- Auto context window sizing — reads config.json + weight sizes, solves for a safe max_model_len from available VRAM.<BR>
- OOM auto-retry — on CUDA OOM, retries the item at draft resolution; batch OOM falls back to per-item.<BR>
- Batch timeout detection - CUDA call are detected and surfaced instead of freezing the UI.<BR>
- Window state persists between app launches.<BR>


## Usage

- Import a model — Choose a model folder containing config.json + .safetensors files. The app validates the folder (checks for weights, valid JSON, model_index.json for diffusers pipelines).<BR>
- Choose backend & settings — vLLM (fastest for batched work) or Transformers (simpler, more quantization options).<BR>
- Select a media folder — pick a folder of images/videos to caption.<BR>
- Run — captions are written back to the source folder in .txt format.<BR>


## Limitations

- GGUF models not supported.<BR>
- Video captioning (16-frame cap, 2 GB size limit).<BR>


## Requirements

- 64‑bit Windows, Python 3.13, NVIDIA GPU with CUDA support.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Highstate/PxCaption.git PxCaption
   cd PxCaption

2. **Run the installer**

   ```bash
   Install.bat

3. **Run PxCaption**

   ```bash
   PxCaption.bat

## Stack

- Python 3.13, Cuda 13.0, PyTorch 2.14.0, Triton-Windows 3.8.0.post29, vLLM 0.33.0, Flash Attention 2.8.3, PyQt6
