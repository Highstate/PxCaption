"""vLLM backend for PxCaption.

Owns the vLLM LLM object and all vLLM-specific logic including the VRAM
cleanup paths (1–4) that prevent torch.compile/inductor artifacts from
leaking GPU memory on model switch.
"""
import base64
import gc
import io
import os
import sys
import threading
import time
import traceback
import weakref as _wr

import torch

from shared import (
    DIAG,
    CTX_GPU_UTIL,
    CTX_FALLBACK,
    DEFAULT_BATCH_SIZE,
    OOM_RETRY_RESOLUTION,
    CaptionResult,
    LoadRequest,
    ModelInfo,
    auto_max_model_len,
    is_oom_error,
    load_and_resize,
    resolution_mm_kwargs,
    vram_string,
    log,
)
from .base import BaseBackend
from .diagnostics import diagnose_gpu_leak

# ---------------------------------------------------------------------------
# Media encoding (base64 data URLs for vLLM chat interface)
# ---------------------------------------------------------------------------

MAX_VIDEO_SIZE_GB = 2.0
MEDIA_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".flv": "video/x-flv",
}


def _image_data_url(image_path: str) -> str:
    """Open an image, normalize it, and return a base64 data: URL."""
    img = load_and_resize(image_path)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _video_data_url(video_path: str) -> str:
    """Return a base64 data: URL for a video file."""
    file_size = os.path.getsize(video_path)
    if file_size > MAX_VIDEO_SIZE_GB * 1024 ** 3:
        raise RuntimeError(
            f"video {os.path.basename(video_path)} is {file_size / 1024 ** 3:.1f} GB, "
            f"exceeding the {MAX_VIDEO_SIZE_GB:.0f} GB limit"
        )
    try:
        with open(video_path, "rb") as f:
            payload = base64.b64encode(f.read()).decode("ascii")
    except OSError as e:
        raise RuntimeError(f"could not read video {os.path.basename(video_path)}: {e}") from e
    mime = MEDIA_MIME.get(os.path.splitext(video_path)[-1].lower(), "video/mp4")
    return f"data:{mime};base64,{payload}"


def media_content_block(media_path: str) -> dict:
    """Build an OpenAI-style chat content block for a local image or video."""
    ext = os.path.splitext(media_path)[-1].lower()
    if ext in (".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv"):
        url = _video_data_url(media_path)
        return {"type": "video_url", "video_url": {"url": url}}
    url = _image_data_url(media_path)
    return {"type": "image_url", "image_url": {"url": url}}


class VLLMBackend(BaseBackend):
    """vLLM LLM backend (preferred for Native quant, continuous batching)."""

    name = "vllm"

    def __init__(self):
        self._model = None
        self.model_id: str | None = None
        self.quant: str | None = None
        self.max_model_len: int | None = None
        self.effective_attn: str | None = None
        self.kv_fp8: bool = False

    @property
    def model(self):
        return self._model

    @staticmethod
    def available() -> bool:
        try:
            import vllm  # noqa: F401
            return True
        except ImportError:
            return False

    def can_reuse(self, desired: LoadRequest) -> bool:
        """True if the currently loaded engine already satisfies this request.

        ``use_torch_compile`` is a Transformers backend concept and is ignored here.
        """
        if self._model is None:
            return False
        if (self.model_id != desired.model_id or self.quant != desired.quant
                or self.effective_attn != desired.attn_impl
                or self.kv_fp8 != desired.kv_fp8):
            return False
        capacity = self.max_model_len
        if not capacity:
            return False
        if desired.max_model_len:
            desired_len = int(desired.max_model_len)
        else:
            desired_len = auto_max_model_len(
                desired.actual_path, batch=desired.batch_size,
                dtype_size=1 if desired.kv_fp8 else 2) or CTX_FALLBACK
        if desired_len != capacity:
            return False
        log.info(f"Reusing existing vLLM engine "
              f"(context {desired_len}, kv_fp8={desired.kv_fp8}); skipping reload.")
        return True

    def load(self, path: str, attn_impl: str, max_model_len: int | None = None,
             kv_fp8: bool = False):
        from vllm import LLM
        self.max_model_len = max_model_len or CTX_FALLBACK
        self.kv_fp8 = kv_fp8
        log.info(f"Loading via vLLM backend: {path} "
              f"(max_model_len={self.max_model_len}, kv_fp8={kv_fp8})")

        # vLLM always uses its own paged-attention kernel internally; the
        # only user-facing distinction is enforce_eager (CUDA-graph capture
        # on vs off).  flash_attention_2 → CUDA graphs (fastest decode);
        # sdpa / eager → no CUDA graphs (more robust, slightly slower).
        # Fallback chain: try CUDA graphs first when the user asked for
        # flash_attention_2, then retry without them if that fails.
        _eager_chain = [False, True] if attn_impl == "flash_attention_2" else [True]

        # Build the LLM kwargs once; the eager chain only toggles enforce_eager.
        _llm_kwargs: dict = dict(
            model=path,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=CTX_GPU_UTIL,
            # Default vision budget ("high"); each generate call overrides
            # min/max_pixels to match the selected Resolution mode.
            mm_processor_kwargs=resolution_mm_kwargs("high"),
        )
        if kv_fp8:
            _llm_kwargs["kv_cache_dtype"] = "fp8"

        for _enforce_eager in _eager_chain:
            _llm_kwargs["enforce_eager"] = _enforce_eager
            try:
                self._model = LLM(**_llm_kwargs)
                break
            except Exception as exc:
                if _enforce_eager:
                    raise
                log.warning(
                    "vLLM CUDA-graph load failed (%s: %s); "
                    "retrying with enforce_eager=True\u2026",
                    exc.__class__.__name__, exc,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.effective_attn = "vllm_paged"
        log.info("vLLM model loaded.")

    def unload(self):
        """vLLM-specific teardown: drop hooks, shutdown engine, run cleanup paths."""
        model, self._model = self._model, None
        self.kv_fp8 = False
        if model is not None:
            # 1) Grab a weakref to the actual nn.Module for diagnostics.
            model_weakref = self._get_model_weakref(model)
            # 2) Snapshot this model's CUDA tensors so the reference-severing
            #    below can be scoped to *this* model (see
            #    _snapshot_model_tensor_ids). Done while the model is intact.
            model_tensor_ids = self._snapshot_model_tensor_ids(model)
            # 3) Remove the torch.compile bytecode hooks vLLM registered on
            #    the model (see _drop_vllm_compile_hooks). These live in a
            #    *global* registry and keep the GPU weights reachable; they
            #    must be removed before the engine is torn down.
            self._drop_vllm_compile_hooks(model)
            # 4) Shut down the engine core: unfreezes the frozen heap and
            #    tears down the KV cache / executor / distributed state.
            self._shutdown_vllm_engine(model)
        else:
            model_weakref = None
            model_tensor_ids = None
        # CRITICAL: drop our last reference BEFORE collecting. While any
        # frame still holds the LLM object (e.g. a local/parameter named
        # 'model'), gc.collect() sees it as reachable and frees nothing.
        del model
        self._free_gpu_memory(model_tensor_ids)
        diagnose_gpu_leak(model_weakref)

    def generate(self, media_path: str, prompt: str, max_tokens: int,
                 resolution_mode: str) -> CaptionResult:
        from vllm import SamplingParams

        def _chat_once(res_mode: str) -> CaptionResult:
            messages = [{
                "role": "user",
                "content": [
                    media_content_block(media_path),
                    {"type": "text", "text": prompt},
                ],
            }]
            sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
            t0 = time.perf_counter()
            try:
                outputs = self._model.chat(
                    [messages],
                    sampling_params=sampling_params,
                    mm_processor_kwargs=resolution_mm_kwargs(res_mode),
                )
                elapsed = time.perf_counter() - t0
                output = outputs[0]
                caption = output.outputs[0].text.strip()
                tokens = len(output.outputs[0].token_ids)
                return CaptionResult(media_path, caption, tokens, elapsed)
            except Exception as e:  # noqa: BLE001
                elapsed = time.perf_counter() - t0
                if is_oom_error(e):
                    raise  # let the OOM-retry handler below deal with it
                return CaptionResult(media_path, "", 0, elapsed, error=str(e))

        try:
            return _chat_once(resolution_mode)
        except Exception as e:  # noqa: BLE001
            # OOM at the requested resolution: free cached CUDA memory and retry
            # once at the lower "fast" resolution. (Per-item OOM recovery that
            # used to live in the removed ModelSession.generate_caption wrapper.)
            if not is_oom_error(e):
                return CaptionResult(media_path, "", 0, 0.0, error=str(e))
            log.warning(f"OOM on {os.path.basename(media_path)}, retrying at lower res...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                return _chat_once(OOM_RETRY_RESOLUTION)
            except Exception as e2:  # noqa: BLE001
                torch.cuda.empty_cache()
                if is_oom_error(e2):
                    return CaptionResult(media_path, "", 0, 0.0, error="OOM (even at low resolution)")
                return CaptionResult(media_path, "", 0, 0.0, error=str(e2))

    def generate_batch(self, items: list[tuple[str, str, int, str]],
                       abort_event: threading.Event) -> list[CaptionResult]:
        """Batch generate via vLLM chat() with multiple conversations.

        The caller (CaptionWorker) is the batching authority: it controls how
        many items are passed per call. This method submits all items in a
        single chat() call.
        """
        from vllm import SamplingParams

        results = []

        if abort_event.is_set():
            for media_path, _, _, _ in items:
                results.append(CaptionResult(media_path, "", 0, 0.0, aborted=True))
            return results

        max_tokens = max(t[2] for t in items)
        # All items in a batch share the run's resolution mode.
        resolution_mode = items[0][3]

        conversations = []
        for media_path, item_prompt, _, _ in items:
            conversations.append([{
                "role": "user",
                "content": [
                    media_content_block(media_path),
                    {"type": "text", "text": item_prompt},
                ],
            }])

        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        t0 = time.perf_counter()
        try:
            outputs = self._model.chat(
                conversations,
                sampling_params=sampling_params,
                mm_processor_kwargs=resolution_mm_kwargs(resolution_mode),
            )
            elapsed = time.perf_counter() - t0
            for j, (media_path, _, _, _) in enumerate(items):
                output = outputs[j]
                caption = output.outputs[0].text.strip()
                tokens = len(output.outputs[0].token_ids)
                results.append(CaptionResult(media_path, caption, tokens, elapsed / len(items)))
        except Exception as e:  # noqa: BLE001
            if is_oom_error(e):
                # OOM (torch or vLLM-wrapped): drop the batch and retry each item
                # individually so one oversized item cannot sink the whole batch.
                log.warning("OOM in vLLM batch, falling back to per-item...")
                torch.cuda.empty_cache()
                for media_path, p, mt, res in items:
                    try:
                        results.append(self.generate(media_path, p, mt, res))
                    except Exception as e2:  # noqa: BLE001
                        results.append(CaptionResult(media_path, "", 0, 0.0, error=str(e2)))
            else:
                for media_path, _, _, _ in items:
                    results.append(CaptionResult(media_path, "", 0, 0.0, error=str(e)))

        return results

    def get_info(self, model_id: str | None) -> ModelInfo:
        if self._model is None:
            return ModelInfo("Model not loaded.", "N/A", "N/A", "N/A", "N/A", "N/A")
        mc = getattr(self._model, "model_config", None)
        return ModelInfo(
            name=model_id or getattr(mc, "model", "vLLM model"),
            device="CUDA",
            vram=vram_string(),
            dtype=str(getattr(mc, "dtype", "N/A")),
            config=str(getattr(mc, "hf_config", None) or "N/A"),
            attn=self.effective_attn or "N/A",
        )

    # -----------------------------------------------------------------------
    # VRAM cleanup paths (moved verbatim from ModelSession)
    # -----------------------------------------------------------------------

    def _free_gpu_memory(self, tensor_ids=None):
        """Collect the old model's objects and return their VRAM to the driver.

        ``tensor_ids`` (a set of CUDA-tensor ids for the model being unloaded)
        scopes the reference-severing so it only touches *this* model's
        tensors — it will never null another model's parameters. If the scoped
        pass is not enough to release the VRAM, we fall back to the historical
        global sweep, so this is never worse than the old behaviour.
        """
        self._clear_fx_node_tensor_refs(tensor_ids)
        self._clear_dynamo_installed_globals()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if tensor_ids is not None:
                free, total = torch.cuda.mem_get_info()
                if free < CTX_GPU_UTIL * total:
                    log.warning("Scoped VRAM cleanup was insufficient; running global fallback sweep.")
                    tensor_ids = None  # widen to the historical global pass
                    self._clear_fx_node_tensor_refs(None)
                    self._clear_dynamo_installed_globals()
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            for _ in range(3):
                free, total = torch.cuda.mem_get_info()
                if free >= CTX_GPU_UTIL * total:
                    break
                self._clear_fx_node_tensor_refs(tensor_ids)
                self._clear_dynamo_installed_globals()
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                time.sleep(0.5)
            free, total = torch.cuda.mem_get_info()
            log.info(f"VRAM after unload: {free / 2**30:.1f} / {total / 2**30:.1f} GiB free")
            if free < CTX_GPU_UTIL * total:
                log.warning("VRAM was not fully released after unload; the next "
                            "model load may fail or fall back to the Transformers backend.")

    @staticmethod
    def _snapshot_model_tensor_ids(llm):
        """Return the set of ``id()``s of every CUDA tensor reachable from the
        model's module tree (parameters, buffers, and any tensor attributes).

        Used to scope the unload-time reference severing (see
        _clear_fx_node_tensor_refs) so we only touch tensors belonging to
        *this* model. Returns an empty set if the module can't be reached — the
        caller then falls back to the global (historical) sweep.
        """
        module = None
        try:
            llm_engine = getattr(llm, "llm_engine", None)
            if llm_engine is not None:
                getter = getattr(llm_engine, "_get_driver_model_for_cleanup", None)
                if callable(getter):
                    try:
                        module = getter()
                    except Exception:
                        module = None
                if module is None:
                    executor = getattr(llm_engine, "model_executor", None)
                    if executor is None:
                        inproc = getattr(llm_engine, "engine_core", None)
                        core = getattr(inproc, "engine_core", None) if inproc else None
                        executor = getattr(core, "model_executor", None) if core else None
                    driver_worker = getattr(executor, "driver_worker", None) if executor else None
                    model_runner = getattr(driver_worker, "model_runner", None) if driver_worker else None
                    module = getattr(model_runner, "model", None) if model_runner else None
        except Exception:
            module = None
        ids = set()

        def add_if_tensor(x):
            if isinstance(x, torch.Tensor):
                if x.is_cuda:
                    ids.add(id(x))
            elif isinstance(x, dict):
                for v in x.values():
                    add_if_tensor(v)
            elif isinstance(x, (list, tuple, set)):
                for v in x:
                    add_if_tensor(v)

        try:
            for m in module.modules():
                add_if_tensor(getattr(m, "__dict__", None))
        except Exception:
            pass
        return ids

    @staticmethod
    def _get_model_weakref(llm):
        """Get a weakref to the actual nn.Module (unwrapped) for diagnostics."""
        try:
            llm_engine = getattr(llm, "llm_engine", None)
            if llm_engine is None:
                return None
            inproc = getattr(llm_engine, "engine_core", None)
            core = getattr(inproc, "engine_core", None)
            executor = getattr(core, "model_executor", None)
            driver_worker = getattr(executor, "driver_worker", None) if executor else None
            model_runner = getattr(driver_worker, "model_runner", None) if driver_worker else None
            model = getattr(model_runner, "model", None) if model_runner else None
            if model is None:
                return None
            while hasattr(model, "runnable"):
                model = model.runnable
            while hasattr(model, "unwrap") and callable(model.unwrap):
                try:
                    inner = model.unwrap()
                    if inner is model:
                        break
                    model = inner
                except Exception:
                    break
            return _wr.ref(model)
        except Exception:
            return None

    @staticmethod
    def _drop_vllm_compile_hooks(llm):
        """Remove vLLM's torch.compile bytecode hooks from the global registry.

        This is the actual root cause of the VRAM leak on in-process reload.

        When vLLM compiles the model, each wrapped submodule
        (``TorchCompileWithNoGuardsWrapper``) registers a *bytecode hook* in
        torch's GLOBAL compile state (``torch._dynamo`` / inductor). That
        global registry then holds a reference to the wrapper and, through
        it, to the model's CUDA tensors. So the model stays reachable -- and
        its ~16 GiB of weights stay resident -- even after the ``LLM`` object
        is deleted and ``gc.collect()`` runs.

        vLLM's own ``LLMEngine`` installs a ``weakref.finalize`` whose job is
        to call ``module.cleanup()`` on these wrappers on deletion, but that
        finalizer only fires once the engine becomes collectable -- which it
        cannot while the very hooks we need it to remove still pin it (a
        chicken-and-egg). ``EngineCore.shutdown()``'s ``gc.unfreeze()`` fixes
        the *freeze* half of the problem but does not remove the hooks, which
        is why the previous unload still leaked. We remove the hooks
        explicitly here, while the model is still intact, then let the GC do
        the rest.
        """
        try:
            from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
        except Exception as e:
            if DIAG:
                log.debug(f"_drop_vllm_compile_hooks: import failed: {e}")
            return
        model = None
        llm_engine = getattr(llm, "llm_engine", None)
        if llm_engine is None:
            if DIAG:
                log.debug("_drop_vllm_compile_hooks: llm_engine is None")
            return
        getter = getattr(llm_engine, "_get_driver_model_for_cleanup", None)
        if callable(getter):
            try:
                model = getter()
            except Exception as e:
                if DIAG:
                    log.debug(f"_drop_vllm_compile_hooks: getter() raised: {e}")
        if model is None:
            executor = getattr(llm_engine, "model_executor", None)
            if executor is None:
                inproc = getattr(llm_engine, "engine_core", None)
                core = getattr(inproc, "engine_core", None)
                executor = getattr(core, "model_executor", None)
            driver_worker = getattr(executor, "driver_worker", None) if executor else None
            model_runner = getattr(driver_worker, "model_runner", None) if driver_worker else None
            model = getattr(model_runner, "model", None) if model_runner else None
            if model is None:
                if DIAG:
                    log.debug(f"_drop_vllm_compile_hooks: model not found "
                          f"(executor={executor is not None}, "
                          f"driver_worker={driver_worker is not None}, "
                          f"model_runner={model_runner is not None})")
                return
        removed = 0
        for module in model.modules():
            if isinstance(module, TorchCompileWithNoGuardsWrapper):
                try:
                    module.cleanup()
                    removed += 1
                except Exception as e:
                    if DIAG:
                        log.debug(f"vLLM compile-hook cleanup failed: {e}")
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        try:
            from vllm.compilation.cuda_graph import CUDAGraphWrapper
            CUDAGraphWrapper.clear_all_graphs()
        except Exception as e:
            if DIAG:
                log.debug(f"CUDAGraphWrapper.clear_all_graphs() failed: {e}")
        try:
            from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
            BreakableCUDAGraphWrapper.clear_all_graphs()
        except Exception as e:
            if DIAG:
                log.debug(f"BreakableCUDAGraphWrapper.clear_all_graphs() failed: {e}")
        log.info(f"Dropped {removed} vLLM torch.compile hook(s), cleared CUDA graphs")

        aot_cleared = 0
        if model is not None:
            for mod in [model] + list(model.modules()):
                fn = getattr(mod, 'aot_compiled_fn', None)
                if fn is not None:
                    mod.aot_compiled_fn = None
                    aot_cleared += 1
            if aot_cleared:
                log.info(f"Cleared {aot_cleared} aot_compiled_fn reference(s)")

    @staticmethod
    def _clear_fx_node_tensor_refs(tensor_ids=None):
        """Break all reference paths from compilation artifacts to GPU tensors.

        After vLLM's piecewise torch.compile, THREE paths keep weight tensors
        alive even after the top-level model is collected:

        1. fx.Node.meta['example_value'] → tensor (FX graph metadata)
        2. fx.GraphModule._modules → nn.Module children → _parameters → tensor
           (GraphModule is an nn.Module that copies submodules into itself)
        3. FakeTensor.real_tensor → real weight tensor
           (dynamo's fake tensor mode stores a reference to the original
            tensor for cross-checking; these outlive the compilation)

        This method walks the GC object list and severs all three paths.
        Safe here because we're in the unload path: the old model's
        submodules are orphaned and the new model hasn't been loaded yet.

        ``tensor_ids`` optionally scopes the severing of nn.Module parameters
        (path 2) and FakeTensor refs (path 3) to the tensors in that set —
        i.e. the model being unloaded — so another model's parameters are never
        touched. Paths 1 and 4 (FX metadata / vLLM serializable fns) are always
        swept globally: they are compile artifacts that exist only because of
        this model's compilation (and may hold *views* not in the module-tree
        snapshot), so scoping them would risk leaving live refs behind.
        ``tensor_ids=None`` reproduces the original global behaviour.
        """
        try:
            import torch.fx as fx
        except ImportError:
            return

        NodeCls = fx.Node
        try:
            from torch.fx.experimental.symbolic_shapes import TrackedFake as _TrackedFakeCls
        except ImportError:
            _TrackedFakeCls = None

        fx_cleared = 0
        mod_cleared = 0
        fake_cleared = 0
        freed_bytes = 0.0

        for obj in gc.get_objects():
            # --- Path 1: fx.Node.meta['example_value'] and meta['val'] ---
            if type(obj) is NodeCls:
                meta = obj.meta
                if isinstance(meta, dict):
                    for key in ('example_value', 'val'):
                        if key not in meta:
                            continue
                        v = meta[key]
                        t = v[1] if isinstance(v, tuple) and len(v) > 1 else v
                        if v is not None and hasattr(t, 'is_cuda') and t.is_cuda:
                            try:
                                freed_bytes += float(t.numel()) * float(t.element_size())
                            except Exception:
                                pass
                        meta[key] = None
                        fx_cleared += 1
            # --- Path 2: nn.Module._parameters holding CUDA tensors ---
            elif isinstance(obj, torch.nn.Module):
                params = obj.__dict__.get('_parameters')
                if params:
                    for key in list(params.keys()):
                        p = params[key]
                        if p is not None and p.is_cuda and (tensor_ids is None or id(p) in tensor_ids):
                            try:
                                freed_bytes += float(p.numel()) * float(p.element_size())
                            except Exception:
                                pass
                            params[key] = None
                            mod_cleared += 1
            # --- Path 3a: TrackedFake (slots dataclass) holding FakeTensor in .fake ---
            elif _TrackedFakeCls is not None and type(obj) is _TrackedFakeCls:
                try:
                    fake = obj.fake
                    if fake is not None and isinstance(fake, torch.Tensor) and fake.is_cuda \
                            and (tensor_ids is None or id(fake) in tensor_ids):
                        freed_bytes += float(fake.numel()) * float(fake.element_size())
                        obj.fake = None
                        fake_cleared += 1
                except Exception:
                    pass
            # --- Path 3b: standalone FakeTensor with .real_tensor in __dict__ ---
            else:
                # obj.__dict__ can raise for some exotic objects (e.g. a pydantic
                # mock intercepts attribute access and raises PydanticUserError).
                # getattr(..., None) only guards AttributeError, so use a real
                # try/except to keep the whole GC sweep from crashing.
                try:
                    d = obj.__dict__
                except Exception:
                    d = None
                rt = d.get('real_tensor') if d is not None else None
                if rt is not None and isinstance(rt, torch.Tensor) and rt.is_cuda \
                        and (tensor_ids is None or id(rt) in tensor_ids):
                    try:
                        freed_bytes += float(rt.numel()) * float(rt.element_size())
                    except Exception:
                        pass
                    obj.__dict__['real_tensor'] = None
                    fake_cleared += 1

        # --- Path 4: VllmSerializableFunction.example_inputs ---
        vllm_fn_cleared = 0
        try:
            from vllm.compilation.caching import VllmSerializableFunction as _VllmSerFn
            for obj in gc.get_objects():
                if type(obj) is _VllmSerFn:
                    ei = getattr(obj, 'example_inputs', None)
                    if ei is not None:
                        obj.example_inputs = None
                        vllm_fn_cleared += 1
            if vllm_fn_cleared and DIAG:
                log.debug(f"_clear_fx_node_tensor_refs: cleared example_inputs "
                      f"from {vllm_fn_cleared} VllmSerializableFunction(s)")
        except ImportError:
            pass

        total = fx_cleared + mod_cleared + fake_cleared
        if (total or vllm_fn_cleared) and DIAG:
            log.debug(f"_clear_fx_node_tensor_refs: cleared {fx_cleared} fx.Node + "
                  f"{mod_cleared} module param + {fake_cleared} FakeTensor.real_tensor refs, "
                  f"est. {freed_bytes / 2**30:.2f} GiB of CUDA tensor refs broken")

    @staticmethod
    def _clear_dynamo_installed_globals():
        """Remove torch.compile-installed globals (comp_fn_*) from all modules.

        When torch.compile compiles a function, dynamo installs the compiled
        callable into the user module's __globals__ dict under a name like
        'comp_fn_0_c3'. A CleanupHook is supposed to remove it when the
        compiled code object is GC'd, but in practice the global itself keeps
        the code object alive (circular reference), so the hook never fires.
        This leaves the FX graph (with node.meta['example_value'] referencing
        the model's weight tensors) permanently alive, preventing VRAM release.

        We scan the torch/vllm/triton/functorch modules for keys matching the
        pattern and delete them, breaking the reference chain. We deliberately
        do NOT sweep every module: comp_fn_* globals are only ever installed
        by torch.compile into the torch/inductor (and downstream vllm/triton)
        namespace, never into user code, so restricting the sweep avoids
        touching any unrelated module (or any user global that happens to
        share the name).
        """
        removed = 0
        _allowed = ("torch", "vllm", "triton", "functorch")

        def _in_allowed(name):
            return any(name == p or name.startswith(p + ".") for p in _allowed)

        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            if not _in_allowed(mod_name):
                continue
            mod_dict = getattr(mod, '__dict__', None)
            if mod_dict is None:
                continue
            keys_to_remove = [
                k for k in mod_dict
                if isinstance(k, str) and k.startswith('comp_fn_')
            ]
            for k in keys_to_remove:
                try:
                    del mod_dict[k]
                    removed += 1
                except (KeyError, TypeError):
                    pass
        if removed and DIAG:
            log.debug(f"_clear_dynamo_installed_globals: removed {removed} comp_fn_* global(s)")

        try:
            from torch._inductor.utils import clear_caches as _ind_clear
            _ind_clear()
        except Exception:
            pass
        if DIAG:
            log.debug("_clear_dynamo_installed_globals: inductor caches cleared")

        try:
            from torch._dynamo.utils import CleanupManager
            cm = CleanupManager.instance
            if cm is not None:
                cm.clear()
        except Exception:
            pass
        if DIAG:
            log.debug("_clear_dynamo_installed_globals: CleanupManager cleared")

    @staticmethod
    def _shutdown_vllm_engine(llm):
        """Best-effort explicit VRAM release of vLLM's engine core.

        Calls the supported V1 release API ``LLMEngine.sleep(level=2)``.
        ``level=2`` offloads the model weights to CPU and discards the KV
        cache, returning the model's VRAM to the driver through vLLM's own
        bookkeeping. This is complementary to, not a replacement for, the
        manual GC sweep in ``_free_gpu_memory``: if ``sleep`` is missing or
        raises on some build, the manual machinery still frees the VRAM.
        """
        llm_engine = getattr(llm, "llm_engine", None)
        sleep_fn = getattr(llm_engine, "sleep", None)
        if callable(sleep_fn):
            try:
                if DIAG:
                    log.debug("Releasing VRAM via LLMEngine.sleep(level=2)")
                sleep_fn(level=2)
            except Exception as e:
                if DIAG:
                    log.debug(f"LLMEngine.sleep(level=2) failed: {e}")
        elif DIAG:
            log.debug("LLMEngine.sleep not found; relying on the manual GC path")
