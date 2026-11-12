"""Transformers backend for PxCaption (fallback path)."""
import gc
import threading
import time

import torch

# Enable the Inductor filesystem cache so that torch.compile artefacts are
# written to ~/.cache/torch/inductor/ and reused on the next process launch.
# Without this, every cold start pays the full trace-and-compile penalty.
try:
    import torch._inductor.config as _inductor_cfg
    _inductor_cfg.fx_graph_cache = True
except (ImportError, AttributeError):
    pass
from PIL import Image
from qwen_vl_utils import process_vision_info

from shared import (
    MAX_VIDEO_FRAMES,
    OOM_RETRY_RESOLUTION,
    RESOLUTION_PIXEL_BUDGET,
    DEFAULT_BATCH_SIZE,
    CaptionResult,
    LoadRequest,
    ModelInfo,
    is_oom_error,
    load_and_resize,
    vram_string,
    is_video_file,
    log,
)
from .base import BaseBackend


class TransformersBackend(BaseBackend):
    """Transformers backend (AutoModelForImageTextToText; supports bitsandbytes)."""

    name = "transformers"

    def __init__(self):
        self._model = None
        self.processor = None
        self.effective_attn: str | None = None
        self._compiled = False

    @property
    def model(self):
        return self._model

    @staticmethod
    def _bnb_config(quant: str):
        """Map a UI quant label to a ``BitsAndBytesConfig`` or ``None``.

        Labels are the backend-qualified strings built by the UI, e.g.
        ``"8-bit (Transformers)"``.  Two 4-bit flavours are supported:
        NF4 (default, best quality/size trade-off) and FP4 (slightly less
        accurate, marginal speed gain).  Double quantization is enabled for
        the NF4 variant to save ~0.4 bits per weight on the quant constants.
        """
        from transformers import BitsAndBytesConfig
        if quant == "8-bit (Transformers)":
            return BitsAndBytesConfig(load_in_8bit=True)
        if quant == "4-bit (Transformers)":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        if quant == "4-bit FP4 (Transformers)":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        return None

    def load(self, path: str, quant: str, attn_impl: str,
             use_torch_compile: bool = False):
        from transformers.models.auto import AutoModelForImageTextToText, AutoProcessor

        torch.backends.cuda.matmul.allow_tf32 = True

        # bfloat16: same 8-bit mantissa as fp16 but a wider exponent, so it is
        # more stable for LLM inference and avoids the fp16 overflow issues.
        # device_map="auto" lets accelerate split layers across all visible GPUs.
        load_kwargs: dict = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": attn_impl,
            "local_files_only": True,
        }

        bnb = self._bnb_config(quant)
        if bnb is not None:
            load_kwargs["quantization_config"] = bnb

        # Fallback chain: each entry is tried in order starting from the
        # user's choice; the first one that loads successfully wins.
        _chain = ["flash_attention_2", "sdpa", "eager"]
        _start = _chain.index(attn_impl) if attn_impl in _chain else 2

        for _candidate in _chain[_start:]:
            load_kwargs["attn_implementation"] = _candidate
            try:
                self._model = AutoModelForImageTextToText.from_pretrained(path, **load_kwargs)
                effective_attn = _candidate
                break
            except Exception as exc:
                if _candidate == "eager":
                    raise
                log.warning(
                    "%s failed (%s: %s); trying next backend\u2026",
                    _candidate, exc.__class__.__name__, exc,
                )
                # Free partial GPU allocations from the failed attempt before
                # retrying, or the next load can OOM on top of them.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.effective_attn = effective_attn
        self._model.eval()

        if use_torch_compile:
            # dynamic=True lets the compiled graph handle variable input shapes
            # (different image resolutions → different vision-token counts)
            # without triggering a full recompile for each new shape.
            self._model = torch.compile(self._model, dynamic=True)
            self._compiled = True
            log.info("torch.compile applied (disk cache enabled).")

        log.info("Transformers model loaded (device_map=auto, attn=%s, compiled=%s).",
                 effective_attn, use_torch_compile)

    def can_reuse(self, desired: LoadRequest) -> bool:
        """True if the currently loaded model already satisfies this request.

        The Transformers backend has no engine-level context, batch state, or
        KV-cache dtype control (it runs per-item at batch 1), so only the
        load-time params matter: model + quantization + attention + compile
        state. ``max_model_len``, ``batch_size``, ``kv_fp8`` and
        ``use_torch_compile`` beyond the current compiled state are vLLM
        concepts and are ignored here.
        """
        if self._model is None:
            return False
        if (desired.model_id != self._get_model_id()
                or desired.quant != self._get_quant()
                or desired.attn_impl != self.effective_attn
                or self._compiled != desired.use_torch_compile):
            return False
        log.info("Reusing existing Transformers model; skipping reload.")
        return True

    def _get_model_id(self) -> str | None:
        return getattr(self._model.config, "_name_or_path", None) if self._model else None

    def _get_quant(self) -> str | None:
        """Recover the quant label from the model's config (best-effort)."""
        if self._model is None:
            return None
        cfg = self._model.config
        if getattr(cfg, "load_in_8bit", False):
            return "8-bit (Transformers)"
        bnb_cfg = getattr(cfg, "quantization_config", None)
        if bnb_cfg is not None:
            if getattr(bnb_cfg, "load_in_4bit", False):
                qtype = getattr(bnb_cfg, "bnb_4bit_quant_type", "nf4")
                if qtype == "fp4":
                    return "4-bit FP4 (Transformers)"
                return "4-bit (Transformers)"
        return None

    def unload(self):
        """Transformers teardown: just drop the reference (no compile hooks to clean)."""
        self._model = None
        self.processor = None
        self.effective_attn = None
        self._compiled = False

    def _build_vision_block(self, media_path: str, resolution_mode: str) -> dict:
        """Construct the Qwen3-VL content block for one media file.

        Images are pre-resized (to cap VRAM on very large files) and then
        constrained to a pixel budget via ``min_pixels`` / ``max_pixels`` so
        the processor picks an aspect-ratio-preserving resolution that is a
        multiple of 28×28.  Videos are passed through as a file path with a
        frame-count cap.
        """
        if is_video_file(media_path):
            return {"type": "video", "video": media_path,
                    "max_frames": MAX_VIDEO_FRAMES}
        img = load_and_resize(media_path)
        lo, hi = RESOLUTION_PIXEL_BUDGET.get(
            resolution_mode, RESOLUTION_PIXEL_BUDGET["high"])
        return {"type": "image", "image": img,
                "min_pixels": lo, "max_pixels": hi}

    @staticmethod
    def _unpack_vision_info(vision_info) -> tuple:
        """Normalise the ``(images, videos, [fps])`` tuple returned by
        ``process_vision_info`` into a plain ``(images, videos)`` pair."""
        if not isinstance(vision_info, tuple) or len(vision_info) not in (2, 3):
            raise ValueError(
                f"process_vision_info: expected a 2- or 3-tuple, "
                f"got {type(vision_info).__name__} of len {len(vision_info)}"
            )
        return vision_info[0], vision_info[1]

    def _generate_once(self, media_path: str, prompt: str,
                       max_tokens: int, res_mode: str):
        """Single inference attempt at the given resolution mode."""
        vision_block = self._build_vision_block(media_path, res_mode)
        chat = [{"role": "user",
                 "content": [vision_block, {"type": "text", "text": prompt}]}]

        rendered = self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)
        imgs, vids = self._unpack_vision_info(process_vision_info(chat))

        tensors = self.processor(
            text=[rendered],
            images=imgs or None,
            videos=vids or None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = self._model.generate(
                **tensors, max_new_tokens=max_tokens, do_sample=False)
        elapsed = time.perf_counter() - t0

        # Slice off the prompt tokens to keep only the generated span.
        new_tokens = [o[len(p):] for p, o in zip(tensors.input_ids, out_ids)]
        token_count = sum(len(t) for t in new_tokens)
        caption = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()

        return CaptionResult(media_path, caption, token_count, elapsed)

    def generate(self, media_path: str, prompt: str, max_tokens: int,
                 resolution_mode: str) -> CaptionResult:
        """Run one image or video through the loaded Qwen3-VL model and
        return the decoded caption wrapped in a ``CaptionResult``.

        On CUDA OOM the item is retried once at ``OOM_RETRY_RESOLUTION``
        (draft) to recover, mirroring the vLLM backend behaviour.
        """
        if self.processor is None:
            raise RuntimeError("Processor is not loaded; call load() first.")

        try:
            return self._generate_once(media_path, prompt, max_tokens, resolution_mode)
        except Exception as e:
            if is_oom_error(e) and resolution_mode != OOM_RETRY_RESOLUTION:
                log.warning(
                    "OOM on %s (%s); retrying at %s\u2026",
                    media_path, resolution_mode, OOM_RETRY_RESOLUTION)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._generate_once(
                    media_path, prompt, max_tokens, OOM_RETRY_RESOLUTION)
            raise

    def generate_batch(self, items: list[tuple[str, str, int, str]],
                       abort_event: threading.Event) -> list[CaptionResult]:
        """Transformers backend processes items individually (variable-size
        images make true batching complex)."""
        results = []
        for i, (media_path, prompt, max_tokens, resolution_mode) in enumerate(items):
            if abort_event.is_set():
                results.append(CaptionResult(media_path, "", 0, 0.0, aborted=True))
                continue
            try:
                results.append(self.generate(media_path, prompt, max_tokens, resolution_mode))
            except Exception as e:
                results.append(CaptionResult(media_path, "", 0, 0.0, error=str(e)))
        return results

    def get_info(self, model_id: str | None) -> ModelInfo:
        if self._model is None:
            return ModelInfo("Model not loaded.", "N/A", "N/A", "N/A", "N/A", "N/A")
        return ModelInfo(
            name=getattr(self._model.config, "_name_or_path", model_id or "Unknown"),
            device="CUDA" if torch.cuda.is_available() else "CPU",
            vram=vram_string(),
            dtype=str(next(self._model.parameters()).dtype),
            config=str(self._model.config),
            attn=self.effective_attn or "N/A",
        )
