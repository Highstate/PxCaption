"""Shared constants, types, and utilities for PxCaption.

Imported by both PxCaption.py and the backend modules to avoid circular imports.
"""
import json
import logging
import os
import threading
from dataclasses import dataclass

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

DIAG = os.environ.get("PXCAPTION_DIAG", "").strip().lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# A single module-level logger. The GUI installs a queue handler (see
# configure_logging) so every line lands in the log pane on the GUI thread.
# Until that is wired (or if there is no queue) a NullHandler keeps logging
# silent and side-effect free. This replaced ad-hoc print() calls so that log
# level, DIAG gating and timestamps are handled in one place.

log = logging.getLogger("pxcaption")
log.addHandler(logging.NullHandler())

_log_queue = None


class _QueueHandler(logging.Handler):
    def emit(self, record):
        q = _log_queue
        if q is not None:
            try:
                q.put(self.format(record))
            except Exception:
                pass


def configure_logging(queue, verbose: bool = False):
    """Route the pxcaption logger into a thread-safe ``queue.Queue`` (the GUI
    log pane). ``verbose`` (i.e. DIAG) switches the effective level to DEBUG.

    Idempotent: safe to call more than once.
    """
    global _log_queue
    _log_queue = queue
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(log.handlers):
        if isinstance(h, (logging.NullHandler, _QueueHandler)):
            log.removeHandler(h)
    handler = _QueueHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(handler)
    log.propagate = False

# Pillow's C-level codecs are NOT thread-safe. Both the caption worker
# (load_and_resize) and the GUI thread (_show_preview) decode images via
# PIL. A module-level lock serialises every PIL decode across threads.
_PIL_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_FILETYPE = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
VIDEO_FILETYPE = ('.mp4', '.mov', '.avi', '.webm', '.mkv', ".flv")
DEFAULT_PROMPT = "Describe this image using natural language."
MAX_VIDEO_FRAMES = 16
PRE_RESIZE_MAX_DIM = 2240
DEFAULT_BATCH_SIZE = 16
BATCH_SIZE_OPTIONS = [1, 2, 4, 8, 16]
CTX_PRESETS = ["Auto", "8K", "4K", "2K"]
CTX_GPU_UTIL = 0.90
CTX_RESERVE_GB = 2.0
CTX_AUTO_TARGET = 8192
CTX_FALLBACK = 32768
VLLM_MAX_IMAGE_PIXELS = 896 * 28 * 28
OOM_RETRY_RESOLUTION = "draft"

# ---------------------------------------------------------------------------
# OOM detection (shared by both backends)
# ---------------------------------------------------------------------------

# Substrings that indicate a CUDA out-of-memory, regardless of which exception
# class wraps it. In-process vLLM may surface an OOM as the raw
# torch.cuda.OutOfMemoryError or as a generic RuntimeError (vLLM's own error
# handling), so we detect it by message as well as by type.
_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "cublas_status_alloc",
    "std::bad_alloc",
    "cannot allocate",
)


def is_oom_error(e: BaseException) -> bool:
    """True if ``e`` (or its cause/context chain) looks like a CUDA OOM.

    We walk the exception chain because the OOM is often the root cause while
    the raised exception is a wrapper.
    """
    seen = set()
    cur = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, torch.cuda.OutOfMemoryError):
            return True
        msg = str(cur).lower()
        if any(marker in msg for marker in _OOM_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# Per-resolution-mode vision pixel budget (min_pixels, max_pixels) for the image
# processor. All modes use min/max pixel bounds (not fixed resize) so the
# processor picks an aspect-ratio-preserving resolution that is a multiple of
# 28×28. This is what the Resolution dropdown actually changes (passed per-call
# as mm_processor_kwargs for vLLM, or inline in the content block for
# Transformers); "high" is the default and its max equals VLLM_MAX_IMAGE_PIXELS.
RESOLUTION_PIXEL_BUDGET = {
    "draft":    (128  * 28 * 28,  196 * 28 * 28),   #  100,352 ..  153,664 px
    "standard": (256  * 28 * 28,  676 * 28 * 28),   #  200,704 ..  529,984 px
    "high":     (256  * 28 * 28, VLLM_MAX_IMAGE_PIXELS),  # 200,704 .. 702,464 px
    "ultra":    (256  * 28 * 28, 1280 * 28 * 28),   #  200,704 .. 1,003,520 px
}
BATCH_TIMEOUT_SECONDS = 600  # per inference batch; a stuck call is detected and surfaced


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CaptionResult:
    media_path: str
    caption: str
    tokens: int
    seconds: float
    error: str | None = None
    aborted: bool = False

    @property
    def tps(self) -> float | None:
        if self.error or self.seconds <= 0:
            return None
        return self.tokens / self.seconds


@dataclass
class CaptionParams:
    folder: str
    prompt: str
    skip_existing: bool
    max_tokens: int
    resolution_mode: str
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass
class ModelInfo:
    """Snapshot of a loaded model's runtime state."""
    name: str
    device: str
    vram: str
    dtype: str
    config: str
    attn: str


@dataclass
class LoadRequest:
    """Parameters describing a model load request.

    Used by ``ModelSession`` to communicate desired load parameters to the
    backend's ``can_reuse()`` check, replacing the previous 13-positional-arg
    signature.
    """
    model_id: str
    quant: str
    attn_impl: str
    actual_path: str
    max_model_len: int | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    kv_fp8: bool = False
    use_torch_compile: bool = False


# ---------------------------------------------------------------------------
# Concurrency helpers
# ---------------------------------------------------------------------------

def run_with_timeout(fn, timeout, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` in a daemon thread with a hard timeout.

    Returns ``(completed, box, thread)`` where ``completed`` is False if the
    thread was still running after ``timeout`` seconds. The call is *abandoned,
    not cancelled* — Python cannot interrupt a thread blocked in a C/CUDA
    call (in-process vLLM included). ``box`` holds ``result`` or ``exc`` once
    the thread has actually returned.

    This is a *detection* primitive: it lets the caller notice a hung batch
    and stop the run cleanly instead of freezing the UI forever.
    """
    box = {}

    def _target():
        try:
            box["result"] = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            box["exc"] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)
    return (not thread.is_alive()), box, thread


# ---------------------------------------------------------------------------
# Image / VRAM helpers
# ---------------------------------------------------------------------------

def load_and_resize(image_path: str) -> Image.Image:
    """Open an image, convert to RGB, and resize if it exceeds PRE_RESIZE_MAX_DIM.

    Why we re-encode instead of using the raw file:
      * ``Image.load()`` forces a *full* decode up front. A truncated/corrupt or
        unusual image (CMYK, 16-bit, palette, animated GIF) raises a Python
        exception here instead of segfaulting inside vLLM/PIL/CUDA later — a
        native crash the Python try/except in the worker cannot catch, which is
        exactly what makes the whole app die with no log line.
      * ``convert("RGB")`` + capping the longest edge give the model a
        predictable, bounded input (limits both memory and the number of vision
        tokens), and drops anything an odd codec could choke on.
    """
    try:
        with _PIL_LOCK:
            with Image.open(image_path) as img:
                img.load()
                img = img.convert("RGB")
                w, h = img.size
                if max(w, h) > PRE_RESIZE_MAX_DIM:
                    scale = PRE_RESIZE_MAX_DIM / max(w, h)
                    img = img.resize(
                        (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
                    )
                return img.copy()
    except Exception as e:
        raise RuntimeError(f"could not read image {os.path.basename(image_path)}: {e}") from e


def vram_string() -> str:
    """Return a human-readable 'used / total GB' string for the primary GPU."""
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            return f"{(total - free) / 1024**3:.2f} / {total / 1024**3:.2f} GB"
        except Exception:
            return "N/A"
    return "N/A"


def resolution_mm_kwargs(resolution_mode: str) -> dict:
    """Per-call ``mm_processor_kwargs`` (min/max pixels) for a resolution mode.

    The vLLM image processor resizes each image so its pixel count lands in
    [min_pixels, max_pixels]; that pixel count is what sets the number of vision
    tokens, so this is the real knob behind the Resolution dropdown. ``max_frames``
    and ``cap_pixels_per_frame`` are included so the dict is self-contained
    (vLLM merges per-call kwargs over the model-level defaults, but being explicit
    keeps the behaviour independent of those defaults).
    """
    lo, hi = RESOLUTION_PIXEL_BUDGET.get(
        resolution_mode, RESOLUTION_PIXEL_BUDGET["high"])
    return {
        "min_pixels": lo,
        "max_pixels": hi,
        "max_frames": MAX_VIDEO_FRAMES,
        "cap_pixels_per_frame": True,
    }


# ---------------------------------------------------------------------------
# File type helpers
# ---------------------------------------------------------------------------

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_FILETYPE)


def is_video_file(filename: str) -> bool:
    return filename.lower().endswith(VIDEO_FILETYPE)


# ---------------------------------------------------------------------------
# Context window auto-sizing
# ---------------------------------------------------------------------------

def auto_max_model_len(model_path: str, dtype_size: int = 2,
                       gpu_util: float = CTX_GPU_UTIL,
                       batch: int = DEFAULT_BATCH_SIZE,
                       reserve_gb: float = CTX_RESERVE_GB,
                       workload_tokens: int = CTX_AUTO_TARGET) -> int | None:
    """Compute a safe vLLM ``max_model_len`` for this model on this GPU.

    Inverts vLLM's sizing: instead of fixing the context and letting vLLM size
    the KV cache, we fix the VRAM budget and solve for the context length.

    The hard constraint (the one behind the original load-time OOM) is that a
    single sequence must fit in the KV pool: ``max_model_len <= pool_tokens``.
    ``batch_cap`` sizes for a full batch of full-length sequences, but the
    ``workload_tokens`` (8K) target can push the result above it, so a
    worst-case full batch may still preempt (real short-caption inputs don't).

    Returns the recommended context length (a multiple of 16) or ``None`` if it
    can't be determined (e.g. no CUDA device, unreadable config) so the caller
    can fall back to ``CTX_FALLBACK``.
    """
    # --- model architecture (read the real values; GQA head count matters) ---
    cfg = {}
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            cfg = {}
    # Some VL models nest the LM config under text_config.
    if "num_hidden_layers" not in cfg and isinstance(cfg.get("text_config"), dict):
        cfg = cfg["text_config"]

    num_layers = cfg.get("num_hidden_layers")
    num_heads = cfg.get("num_attention_heads")
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    hidden = cfg.get("hidden_size")
    head_dim = cfg.get("head_dim") or (hidden // num_heads if hidden and num_heads else None)
    native_max = cfg.get("max_position_embeddings", cfg.get("max_sequence_length"))

    if not (num_layers and num_kv_heads and head_dim):
        log.warning("auto_max_model_len: could not read architecture from config.json")
        return None

    # KV cache bytes per token: K+V, all layers, KV heads, element size.
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_size

    # --- VRAM budget ---
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    weight_bytes = 0
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                weight_bytes = int(json.load(f).get("total_size", 0))
        except (json.JSONDecodeError, OSError, ValueError):
            weight_bytes = 0
    if weight_bytes == 0:
        weight_bytes = sum(
            os.path.getsize(os.path.join(model_path, fn))
            for fn in os.listdir(model_path)
            if fn.endswith(".safetensors")
        )
    usable = total * gpu_util - weight_bytes - (reserve_gb * 1024 ** 3)
    if usable <= 0:
        log.warning("auto_max_model_len: not enough VRAM "
                    f"(usable={usable / 1024 ** 3:.1f} GB) for this model")
        return None

    pool_tokens = usable // kv_per_token
    single_seq_cap = pool_tokens
    batch_cap = pool_tokens // max(batch, 1)

    candidate = min(single_seq_cap, max(batch_cap, workload_tokens))
    if native_max:
        candidate = min(candidate, int(native_max))
    candidate -= candidate % 16
    candidate = max(candidate, 16)

    log.info(f"auto_max_model_len: kv/token={kv_per_token / 1024:.1f}KiB "
             f"weights={weight_bytes / 1024 ** 3:.1f}GB usable={usable / 1024 ** 3:.1f}GB "
             f"pool={pool_tokens} batch={batch} -> {candidate}")
    return candidate
