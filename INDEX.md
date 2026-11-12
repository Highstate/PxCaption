# PxCaption — Project Index

Modular PyQt6 desktop app that batch-captions images and videos using local Qwen3-VL models via vLLM (primary) or Transformers (fallback, optionally with torch.compile).

## File Structure

```
C:\AI\PxCaption\
├── PxCaption.py                   ← Main entry point: UI, ModelSession coordinator, workers
├── shared.py                      ← Constants, dataclasses, media utils, auto_max_model_len, OOM detection, logging, run_with_timeout
├── backends/
│   ├── __init__.py                ← Re-exports VLLMBackend, TransformersBackend, BaseBackend
│   ├── base.py                    ← BaseBackend ABC
│   ├── diagnostics.py             ← diagnose_gpu_leak() (VRAM leak diagnostics)
│   ├── vllm_backend.py            ← vLLM logic + media encoding + scoped VRAM cleanup + FP8 KV cache
│   └── transformers_backend.py    ← Transformers fallback + OOM retry + optional torch.compile (Inductor disk cache)
├── Requirements.txt               ← Fully-pinned deps, deduped (incl. 2 custom local wheels)
├── Install.bat                    ← One-shot setup: downloads custom wheels, creates venv, installs all deps
└── PxCaption.bat                  ← Launch script (points to venv)
```

> **User data** (`model_store.json`, `window.state`) is read/written in the project root via `_APP_DIR = os.path.dirname(os.path.abspath(__file__))`. Both writes are atomic (tmp + fsync + `os.replace`). Model weights live in `E:/Models/VLM/` (external).

## File Details

### `PxCaption.py` (~1,700 lines)

| Section | Contents |
|---------|----------|
| **Imports / globals** | `ModelInfo` dataclass imported from `shared`; `MODEL_STORE` loaded from `model_store.json`; `_atomic_write_json()` for user data; `_PrintRedirect` for stdout/stderr → GUI log pane; `faulthandler` for native crash traces. |
| **ModelSession** | Coordinator: delegates to `VLLMBackend` or `TransformersBackend`. Tracks `model_id`, `quant`, `attn_impl`, `kv_fp8`, `use_torch_compile`. `can_reuse()` prevents redundant reloads. `auto_max_model_len()` computes safe vLLM context from VRAM budget (doubled when FP8 KV is on). |
| **ModelLoadWorker** | QThread wrapping a daemon thread with `LOAD_TIMEOUT = 300 s`. Emits `sig_done(ModelInfo fields, backend)` or `sig_error(str)`. Passes `kv_fp8` and `use_torch_compile` to `ModelSession.load()`. |
| **CaptionWorker** | QThread running `CaptionParams` in a daemon thread with `BATCH_TIMEOUT_SECONDS = 600 s`. Yields `CaptionResult` per item via `sig_item`. |
| **UI (`MainWindow`)** | PyQt6 single-window layout. Model groupbox: Model / Backend / Quantization / Attention / Context / FP8 KV Cache / Batch Size. Three backend options: **vLLM**, **Transformers (+Torch Compile)**, **Transformers**. Quantization dropdown swaps items per backend: vLLM → Native / AWQ / GPTQ; Transformers → None / 8-bit / 4-bit (NF4) / 4-bit (FP4). Attention: Flash Attention 2 / SDPA / Eager (auto-fallback chain: FA2 → SDPA → Eager). Resolution: draft / standard / high / ultra (default: **high**). Context: Auto / 8K / 4K / 2K (vLLM only). FP8 KV Cache checkbox (vLLM only): halves KV VRAM, doubles auto context. Batch size (vLLM only, default 16). `_AutoFitLabel` rescales the preview on resize; atomic `_atomic_write_json()` for `model_store.json` + `window.state` (project-root data path). |
| **Changelog** | See below. |

### `shared.py`

| Item | Notes |
|------|-------|
| `IMAGE_FILETYPE` / `VIDEO_FILETYPE` | Extension tuples. |
| `DEFAULT_PROMPT` | `"Describe this image using natural language."` |
| `RESOLUTION_PIXEL_BUDGET` | `draft` (128·28·28 … 196·28·28), `standard` (256·28·28 … 676·28·28), `high` (256·28·28 … 896·28·28), `ultra` (256·28·28 … 1280·28·28). |
| `OOM_RETRY_RESOLUTION` | `"draft"` — per-item OOM retry falls back to this mode. |
| `ModelInfo` dataclass | `name, device, vram, dtype, config, attn`. Returned by both backends' `get_info()`. |
| `CaptionResult` dataclass | `media_path, caption, tokens, seconds, error, aborted`. Property `tps` = tokens/sec. |
| `CaptionParams` dataclass | `folder, prompt, skip_existing, max_tokens, resolution_mode, batch_size`. |
| `auto_max_model_len()` | Reads `config.json` + weight sizes; solves for safe `max_model_len` given VRAM, batch, `dtype_size` (2 = BF16, 1 = FP8). Returns multiple of 16 or `None`. |
| `run_with_timeout()` | Daemon-thread wrapper; returns `(completed, box, thread)`. Detection only — cannot cancel a blocked CUDA call. |
| `configure_logging()` | Routes `pxcaption` logger into a `queue.Queue` for the GUI log pane. `DIAG` env var gates DEBUG level. |
| `LoadRequest` dataclass | `model_id, quant, attn_impl, actual_path, max_model_len, batch_size, kv_fp8, use_torch_compile`. Passed to `can_reuse()`. |
| `is_oom_error()` | Walks the exception chain for CUDA OOM markers (type + message). Shared by both backends. |
| `load_and_resize()` | Opens an image, forces full decode, converts to RGB, caps at `PRE_RESIZE_MAX_DIM`. Raises `RuntimeError` on corrupt files. |
| `vram_string()` | Returns `"used / total GB"` for the primary CUDA device, or `"N/A"`. |
| `resolution_mm_kwargs()` | Per-call `mm_processor_kwargs` dict for vLLM: `min_pixels`, `max_pixels`, `max_frames`, `cap_pixels_per_frame`. |

### `backends/base.py`

`BaseBackend` ABC. Abstract methods: `model` (property), `load()`, `can_reuse()`, `unload()`, `generate()`, `generate_batch()`, `get_info() → ModelInfo`.

`can_reuse(desired: LoadRequest) -> bool`.

### `backends/vllm_backend.py`

- `VLLMBackend(BaseBackend)`. `name = "vllm"`.
- **Load**: `LLM(model, dtype="bfloat16", max_model_len, gpu_memory_utilization=0.90, enforce_eager, mm_processor_kwargs)`. Optional `kv_cache_dtype="fp8"` when FP8 KV is enabled. Attention fallback chain: CUDA graphs (FA2) → `enforce_eager=True`.
- **AWQ / GPTQ**: auto-detected by vLLM from the model's `config.json`. No code change needed; user imports a pre-quantised folder.
- **FP8 KV cache**: `kv_cache_dtype="fp8"` passed to `LLM()`. Halves per-token KV size. `auto_max_model_len()` uses `dtype_size=1` to compensate, roughly doubling the auto context window.
- **`can_reuse(desired: LoadRequest)`**: checks model_id, quant, attn, kv_fp8, and resolved `max_model_len`. `use_torch_compile` is ignored.
- **Media encoding** (vLLM chat interface): `_image_data_url()` (load_and_resize → JPEG q90 → base64), `_video_data_url()` (raw base64, 2 GB cap), `media_content_block()`.
- **OOM handling**: uses `shared.is_oom_error()`. Single-item OOM → retry at `draft` resolution. Batch OOM → fall back to per-item.
- **VRAM cleanup on unload**: scoped tensor-ID reference severing, FX/dynamo hook removal, 3-retry sweep with `torch.cuda.mem_get_info()` check.

### `backends/transformers_backend.py`

- `TransformersBackend(BaseBackend)`. `name = "transformers"`.
- **Load**: `AutoModelForImageTextToText.from_pretrained(path, dtype=bfloat16, device_map="auto", attn_implementation, local_files_only=True)`. Optional `BitsAndBytesConfig` for 8-bit / NF4 / FP4.
- **Attention fallback chain**: `flash_attention_2` → `sdpa` → `eager`. Each failure logs, frees partial GPU allocations, tries next.
- **torch.compile**: when `use_torch_compile=True`, wraps the model with `torch.compile(model, dynamic=True)` after `.eval()`. `dynamic=True` handles variable vision-token counts without recompiling per shape. Inductor filesystem cache (`fx_graph_cache=True`) persists compiled graphs to `~/.cache/torch/inductor/` so subsequent launches skip the compile phase.
- **`_bnb_config()`**: factory method. NF4 (default 4-bit) uses `bnb_4bit_quant_type="nf4"` + `bnb_4bit_use_double_quant=True`. FP4 uses `bnb_4bit_quant_type="fp4"`.
- **`can_reuse(desired: LoadRequest)`**: checks model_id, quant, attn, and `use_torch_compile`. `max_model_len`, `batch_size`, `kv_fp8` are vLLM concepts and ignored.
- **OOM retry**: on CUDA OOM, retries once at `draft` resolution (mirrors vLLM backend). Uses `shared.is_oom_error()`.
- **Vision processing**: `_build_vision_block()` calls `shared.load_and_resize()` then applies `RESOLUTION_PIXEL_BUDGET` (min/max pixels, preserves aspect ratio). `_unpack_vision_info()` normalises `process_vision_info` tuple (2 or 3 elements).
- **Inference**: `torch.inference_mode()`, `do_sample=False`, `temperature=None`.

## Changelog

### Current
- Backend dropdown now has three options: **vLLM**, **Transformers (+Torch Compile)**, **Transformers**.
- `torch.compile` (Inductor) with disk cache (`~/.cache/torch/inductor/`) for the Transformers backend. `dynamic=True` for variable input shapes. Toggling compile state forces a reload.
- AWQ / GPTQ quantization via vLLM (auto-detected from model `config.json`). Quantization dropdown swaps items per backend.
- FP8 KV cache checkbox (vLLM only): `kv_cache_dtype="fp8"`, halves KV VRAM, doubles auto context. Included in `can_reuse()` check.
- Resolution default changed to **high**. Modes renamed: draft / standard / high / ultra (all use min/max pixels, preserving aspect ratio).
- Attention: three options (FA2 / SDPA / Eager) with auto-fallback chain FA2 → SDPA → Eager.
- `device_map="auto"` for multi-GPU support (Transformers backend).
- `ModelInfo` dataclass replaces the 6-tuple returned by `get_info()`.
- `IMAGE_EXTENSIONS` → `IMAGE_FILETYPE`; `VIDEO_EXTENSIONS` → `VIDEO_FILETYPE`.
- `DEFAULT_MODEL_ID` removed; model list starts empty until a model is imported.
- Dead code removed (prompt summary/one-sentence modes, `generate_caption()` — its OOM retry ported into the backends, unused imports).
- Batch size defaults to 16 when vLLM backend is selected.

### Earlier (historical)
- `Requirements.txt` deduped (191→161 pkg lines); `torchaudio` pinned `+cu130`.
- `prompt.txt` + `v1_llm_engine.txt` moved to `docs/`; removed `_audit_check.py`.
- `faulthandler` doc corrected (native crashes → console window, not the GUI pane).
- Context change forces reload (`can_reuse` `!=`), `cmb_ctx` disabled during load, max-tokens default 512, transformers path in bfloat16, all `print()`→module logger, videos never batched.
