"""PxCaption — batch-caption images and videos using local Qwen3-VL models.

This is the main entry-point script. It contains the UI (PyQt6), model store
helpers, and a thin ModelSession coordinator that delegates to backends in
the ``backends/`` package.
"""
import faulthandler
import gc
import io
import json
import os
import queue
import sys
import threading
import time
import traceback
import warnings

import torch

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Suppress vLLM's deprecation warning about torch.distributed.reduce_op.
# MUST appear before vllm is imported (which happens lazily inside backends).
warnings.filterwarnings("ignore", message=r"torch\.distributed\.reduce_op", category=FutureWarning)

from shared import (
    DIAG,
    CTX_PRESETS,
    CTX_FALLBACK,
    DEFAULT_PROMPT,
    DEFAULT_BATCH_SIZE,
    BATCH_SIZE_OPTIONS,
    CaptionResult,
    CaptionParams,
    LoadRequest,
    ModelInfo,
    auto_max_model_len,
    is_image_file,
    is_video_file,
    log,
    configure_logging,
    run_with_timeout,
    vram_string,
    BATCH_TIMEOUT_SECONDS,
)
from backends import VLLMBackend, TransformersBackend, BaseBackend


# ---------------------------------------------------------------------------
# Model store
# ---------------------------------------------------------------------------

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_STORE_PATH = os.path.join(_APP_DIR, "model_store.json")
WINDOW_STATE_PATH = os.path.join(_APP_DIR, "window.state")
ICON_PATH = os.path.join(_APP_DIR, "PxCaption.ico")
DEFAULT_WINDOW_W = 1600
DEFAULT_WINDOW_H = 900


def load_model_store():
    if not os.path.exists(MODEL_STORE_PATH):
        return []
    try:
        with open(MODEL_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error(f"Could not read model store {MODEL_STORE_PATH}: {e}")
        return []
    if not isinstance(data, list):
        return []
    models = []
    for entry in data:
        if isinstance(entry, dict) and "name" in entry and "path" in entry:
            models.append({"name": str(entry["name"]), "path": str(entry["path"])})
    return models


def _atomic_write_json(path: str, data):
    """Atomic JSON write: temp file + fsync + os.replace() over the target.

    A crash/power-loss mid-write can no longer leave a truncated/partial file at
    ``path`` (which would corrupt the model list or window geometry). os.replace
    is atomic on both Windows and POSIX. Used for model_store.json and window.state."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_model_store(models):
    """Persist the model store atomically (see _atomic_write_json)."""
    _atomic_write_json(MODEL_STORE_PATH, models)


MODEL_STORE = load_model_store()


def validate_model_folder(path):
    if not path or not os.path.exists(path):
        return False, "The selected path does not exist."
    if not os.path.isdir(path):
        return False, "The selected path is not a folder."

    has_model_index = os.path.isfile(os.path.join(path, "model_index.json"))
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    if not (has_model_index or has_config):
        return False, ("No config.json or model_index.json found \u2014 this does not look like a model folder.")

    config_file = "config.json" if has_config else "model_index.json"
    try:
        with open(os.path.join(path, config_file), "r", encoding="utf-8") as f:
            json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"{config_file} is not valid JSON: {e}"

    weight_exts = (".safetensors", ".bin")
    found_weights = False
    for _root, _dirs, files in os.walk(path):
        if any(fn.endswith(weight_exts) for fn in files):
            found_weights = True
            break
    if not found_weights:
        return False, "No weight files (*.safetensors / *.bin) found in the folder."

    if has_model_index:
        return True, "Valid diffusers pipeline model."
    return True, "Valid model folder (config.json + weight files)."


def preflight_check_weights(model_path):
    """Verify all safetensors shards referenced in the index file exist."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return True, ""
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
    except (json.JSONDecodeError, OSError) as e:
        return False, f"Could not parse {index_path}: {e}"

    missing = sorted({
        shard for shard in weight_map.values()
        if not os.path.isfile(os.path.join(model_path, shard))
    })
    if missing:
        return False, (
            f"Missing {len(missing)} weight shard(s) referenced in the index:\n"
            + "\n".join(f"  - {m}" for m in missing[:10])
            + (f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else "")
        )
    return True, ""


def resolve_model_path(model_name: str) -> str:
    for entry in MODEL_STORE:
        if entry["name"] == model_name:
            return entry["path"]
    return model_name


# ---------------------------------------------------------------------------
# ModelSession (thin coordinator)
# ---------------------------------------------------------------------------

class ModelSession:
    """Coordinates model lifecycle and generation by delegating to a backend.

    Public interface is unchanged from the original single-file design, so
    the UI workers need no modifications.
    """

    def __init__(self):
        self._impl: BaseBackend | None = None
        self.model_id: str | None = None
        self.quant: str | None = None
        self.attn_impl: str | None = None
        self.kv_fp8: bool = False
        self.use_torch_compile: bool = False
        self._abort_event = threading.Event()
        self._busy = threading.Event()

    # ----- properties delegating to the active backend -----
    # (Only the two below are read by the UI; the original single-file design also
    #  exposed is_busy / processor / max_model_len / effective_attn, but the workers
    #  access model / backend / model_id / quant / attn_impl directly, so those
    #  delegators became dead and were removed.)

    @property
    def model(self):
        return self._impl.model if self._impl else None

    @property
    def backend(self) -> str | None:
        return self._impl.name if self._impl else None

    # ----- lifecycle -----

    def load(self, model_id: str, quant: str, attn_impl: str,
             max_model_len: int | None = None,
             batch_size: int = DEFAULT_BATCH_SIZE,
             kv_fp8: bool = False,
             use_torch_compile: bool = False) -> ModelInfo:
        if self._busy.is_set():
            raise RuntimeError("Another model operation is in progress.")
        self._busy.set()
        try:
            return self._load_impl(model_id, quant, attn_impl,
                                   max_model_len, batch_size, kv_fp8,
                                   use_torch_compile)
        finally:
            self._busy.clear()

    def _load_impl(self, model_id: str, quant: str, attn_impl: str,
             max_model_len: int | None = None,
             batch_size: int = DEFAULT_BATCH_SIZE,
             kv_fp8: bool = False,
             use_torch_compile: bool = False) -> ModelInfo:
        actual_path = resolve_model_path(model_id)
        log.info(f"Loading model: {model_id}")
        log.info(f"Resolved path: {actual_path}")

        if not os.path.isdir(actual_path):
            raise FileNotFoundError(
                f"Model folder not found at:\n{actual_path}\n\n"
                "Check the path in the model store (and that the drive is mounted)."
            )

        ok, msg = preflight_check_weights(actual_path)
        if not ok:
            raise FileNotFoundError(f"Model folder is incomplete.\n{msg}")

        # Any quant string ending in "(vLLM)" routes to the vLLM backend;
        # this covers None, AWQ, and GPTQ in one check.
        use_vllm = (quant.endswith("(vLLM)") and VLLMBackend.available())
        requested_backend = "vllm" if use_vllm else "transformers"

        # Reuse the currently loaded model when a reload is not needed. Reuse is
        # only possible within the same backend: switching vLLM <-> Transformers
        # (or a vLLM request that fell back to Transformers) is a reload.
        if (self._impl is not None and self._impl.name == requested_backend
                and self._impl.can_reuse(LoadRequest(
                    model_id, quant, attn_impl, actual_path,
                    max_model_len=max_model_len,
                    batch_size=batch_size,
                    kv_fp8=kv_fp8,
                    use_torch_compile=use_torch_compile,
                ))):
            self._abort_event.clear()
            self.model_id = model_id
            self.quant = quant
            self.attn_impl = attn_impl
            self.kv_fp8 = kv_fp8
            self.use_torch_compile = use_torch_compile
            return self.get_info()

        self.unload()
        self._abort_event.clear()

        if use_vllm:
            # Resolve context window: explicit override, auto-calc, or fallback.
            # FP8 KV cache halves the per-token KV size, so dtype_size=1.
            _kv_dtype_size = 1 if kv_fp8 else 2
            if max_model_len:
                chosen = int(max_model_len)
            else:
                chosen = auto_max_model_len(
                    actual_path, batch=batch_size,
                    dtype_size=_kv_dtype_size) or CTX_FALLBACK
            log.info(f"vLLM max_model_len = {chosen} (kv_fp8={kv_fp8})")
            self._impl = VLLMBackend()
            try:
                self._impl.load(actual_path, attn_impl, chosen, kv_fp8=kv_fp8)
            except Exception as e:
                log.warning(f"vLLM load failed ({e}), falling back to Transformers backend.")
                self._impl = None
                # Clean up partial GPU allocations from the failed vLLM load.
                VLLMBackend._clear_fx_node_tensor_refs()
                VLLMBackend._clear_dynamo_installed_globals()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._impl = TransformersBackend()
                self._impl.load(actual_path, quant, attn_impl,
                                use_torch_compile=use_torch_compile)
        else:
            self._impl = TransformersBackend()
            self._impl.load(actual_path, quant, attn_impl,
                            use_torch_compile=use_torch_compile)

        self.model_id = model_id
        self.quant = quant
        self.attn_impl = attn_impl
        self.kv_fp8 = kv_fp8
        self.use_torch_compile = use_torch_compile
        return self.get_info()

    def unload(self):
        log.info("Unloading model")
        if self._impl:
            self._impl.unload()
            self._impl = None

    # ----- info -----

    def get_info(self) -> ModelInfo:
        if self._impl is None:
            return ModelInfo("Model not loaded.", "N/A", "N/A", "N/A", "N/A", "N/A")
        return self._impl.get_info(self.model_id)

    def get_vram_string(self) -> str:
        return vram_string()

    # ----- generation -----

    def abort(self):
        self._abort_event.set()

    def is_aborted(self) -> bool:
        return self._abort_event.is_set()

    def reset_abort(self):
        self._abort_event.clear()

    def generate_batch(self, items: list[tuple[str, str, int, str]]) -> list[CaptionResult]:
        """Generate captions for a batch of (media_path, prompt, max_tokens, res_mode)."""
        if self._busy.is_set():
            raise RuntimeError("Model is loading or unloading.")
        if self._impl is None or self._impl.model is None:
            raise RuntimeError("Model must be loaded before generating captions.")
        return self._impl.generate_batch(items, self._abort_event)


# ---------------------------------------------------------------------------
# Console capture
# ---------------------------------------------------------------------------

class _PrintRedirect(io.TextIOBase):
    """Wraps stdout/stderr, forwarding lines to a thread-safe queue.

    The GUI thread drains the queue via a QTimer, so no Qt widget is ever
    touched from a worker / C-level thread (which was causing 0xc0000374
    stack-overflow crashes under vLLM's internal threads).
    """

    def __init__(self, original, log_queue: queue.Queue):
        super().__init__()
        self.original = original
        self.log_queue = log_queue
        self._buf = ""
        # The GIL does not make `self._buf += text` atomic: multiple worker
        # threads write to stdout/stderr concurrently (an abandoned, still-
        # alive inference daemon after a batch timeout included). A lost
        # update would corrupt the line buffer and drop/garble log lines.
        self._buf_lock = threading.Lock()

    def writable(self):
        return True

    def write(self, text):
        if not text:
            return 0
        with self._buf_lock:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self.log_queue.put(line)
        try:
            return self.original.write(text)
        except Exception:
            return len(text)

    def flush(self):
        with self._buf_lock:
            if self._buf:
                self.log_queue.put(self._buf)
                self._buf = ""
        try:
            self.original.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self.original.isatty()
        except Exception:
            return False

    def fileno(self):
        # Delegate to the real underlying stream so native/ctypes/extension code
        # that asks the redirect for a file descriptor (faulthandler, progress
        # bars, logging, C extensions) gets the actual fd instead of io.TextIOBase
        # defaulting to io.UnsupportedOperation. If the underlying stream has no
        # real fd, surface that cleanly.
        try:
            return self.original.fileno()
        except (AttributeError, ValueError, io.UnsupportedOperation):
            raise io.UnsupportedOperation(
                "redirected stdout/stderr has no file descriptor"
            ) from None


# ---------------------------------------------------------------------------
# Qt helpers
# ---------------------------------------------------------------------------




class CollapsibleGroup(QWidget):
    """A group box whose content can be collapsed/expanded."""

    def __init__(self, title: str, default_collapsed: bool = False, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._title = title
        self._header = QPushButton(self)
        self._header.setCheckable(True)
        self._header.setFlat(True)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._header.setMinimumHeight(28)
        self._header.setContentsMargins(10, 4, 10, 4)
        self._header.setStyleSheet(
            "QPushButton {"
            "  text-align: left; font-weight: bold; background-color: transparent;"
            "  border: none; border-radius: 4px; padding: 2px 6px;}"
            "QPushButton:hover   { background-color: rgba(255,255,255,0.1); }"
            "QPushButton:pressed { background-color: rgba(255,255,255,0.15); }"
        )
        self._header.toggled.connect(self._sync_header)
        outer.addWidget(self._header)

        self._content = QWidget(self)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 8, 10, 10)
        self._content_layout.setSpacing(6)
        self._content.setVisible(not default_collapsed)
        outer.addWidget(self._content)

        self._header.blockSignals(True)
        self._header.setChecked(not default_collapsed)
        self._header.blockSignals(False)
        self._sync_header()

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def _sync_header(self):
        expanded = self._header.isChecked()
        self._header.setText(f"{'\u25be' if expanded else '\u25b8'}  {self._title}")
        self._content.setVisible(expanded)

    def set_expanded(self, expanded: bool):
        self._header.blockSignals(True)
        self._header.setChecked(expanded)
        self._header.blockSignals(False)
        self._sync_header()


class _AutoFitLabel(QLabel):
    """QLabel that invokes a callback whenever it is resized, so the current
    preview can be rescaled to the new size. QLabel exposes no ``sizeChanged``
    signal, so we override ``resizeEvent`` (the previous
    ``sizeChanged.connect`` was a silent no-op that never refit the preview)."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._on_resize = None

    def set_resize_callback(self, cb):
        self._on_resize = cb

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._on_resize is not None:
            try:
                self._on_resize()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class ModelLoadWorker(QThread):
    sig_done = pyqtSignal(str, str, str, str, str, str, str)
    sig_error = pyqtSignal(str)

    LOAD_TIMEOUT = 300  # seconds

    def __init__(self, session: ModelSession, model_id: str, quant: str, attn: str,
                 max_model_len: int | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 kv_fp8: bool = False,
                 use_torch_compile: bool = False,
                 parent=None):
        super().__init__(parent)
        self.session = session
        self.model_id = model_id
        self.quant = quant
        self.attn = attn
        self.max_model_len = max_model_len
        self.batch_size = batch_size
        self.kv_fp8 = kv_fp8
        self.use_torch_compile = use_torch_compile
        self._load_thread = None

    def load_in_flight(self) -> bool:
        """True while the load daemon thread is still running.

        This can outlive the QThread itself: on a LOAD_TIMEOUT the QThread
        returns (isRunning() -> False) but the daemon keeps loading in the
        background and keeps session._busy set. The UI uses this to refuse a
        second, concurrent load (which would otherwise raise "Another model
        operation is in progress" and run two CUDA ops at once).
        """
        return self._load_thread is not None and self._load_thread.is_alive()

    def run(self):
        """Load the model in a daemon thread so a hung CUDA init cannot
        freeze the UI forever. The worker emits an error after LOAD_TIMEOUT
        seconds; the load thread continues in the background as a daemon."""
        result: dict = {}
        error: dict = {}

        def _do_load():
            try:
                result['info'] = self.session.load(
                    self.model_id, self.quant, self.attn,
                    self.max_model_len, self.batch_size,
                    self.kv_fp8, self.use_torch_compile,
                )
            except Exception as e:
                error['exc'] = e

        load_thread = threading.Thread(target=_do_load, daemon=True)
        self._load_thread = load_thread
        load_thread.start()
        load_thread.join(self.LOAD_TIMEOUT)

        if load_thread.is_alive():
            self.sig_error.emit(
                f"\u274c Model load timed out after {self.LOAD_TIMEOUT // 60} min. "
                "The load may still be running in the background."
            )
            return

        if 'exc' in error:
            exc = error['exc']
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            self.sig_error.emit(f"\u274c Failed to load model: {exc}")
            return

        info: ModelInfo = result['info']
        backend = self.session.backend or "unknown"
        if info.attn != self.attn and backend == "transformers":
            status = (f"\u26a0\ufe0f Loaded ({backend}) with {self.quant} "
                      f"(attention fell back to {info.attn}). See Log.")
        else:
            status = f"\u2705 Loaded ({backend}) \u2014 {self.quant}, {info.attn}."
        self.sig_done.emit(status, info.name, info.device, info.vram,
                           info.dtype, info.config, backend)


class CaptionWorker(QThread):
    sig_item = pyqtSignal(object)

    def __init__(self, session: ModelSession, params: CaptionParams, parent=None):
        super().__init__(parent)
        self.session = session
        self.params = params
        self._inflight_thread = None

    def inference_in_flight(self) -> bool:
        """True while an abandoned inference thread (from a timed-out batch)
        is still running on the GPU. Mirrors ModelLoadWorker.load_in_flight():
        in-process vLLM cannot be interrupted mid-call, so a batch that trips
        run_with_timeout leaves its daemon thread alive. The UI uses this to
        refuse a second caption run / model load (which would otherwise race
        two concurrent CUDA operations)."""
        return (self._inflight_thread is not None
                and self._inflight_thread.is_alive())

    def run(self):
        p = self.params
        folder_path = p.folder.strip()
        prompt = p.prompt.strip()
        skip_existing = p.skip_existing
        max_tokens = p.max_tokens
        resolution_mode = p.resolution_mode

        log.info(f"Starting folder processing:")
        log.info(f"  model: {self.session.model_id} ({self.session.backend})")
        log.info(f"  quant: {self.session.quant}")
        log.info(f"  folder: {folder_path}")
        log.info(f"  prompt: {prompt}")
        log.info(f"  max_tokens: {max_tokens}")
        log.info(f"  resolution: {resolution_mode}")

        if not folder_path:
            self.sig_item.emit({"kind": "no_media", "status": "\u26a0\ufe0f Please enter a valid folder path."})
            return
        if not os.path.exists(folder_path):
            self.sig_item.emit({"kind": "no_media", "status": f"\u274c Folder not found: {folder_path}"})
            return

        media_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if is_image_file(file) or is_video_file(file):
                    media_files.append(os.path.join(root, file))

        total = len(media_files)
        if total == 0:
            self.sig_item.emit({"kind": "no_media", "status": "\U0001f4c2 No media found."})
            return

        # Pre-compute skip flags so we can batch non-skipped files.
        skip_flags = []
        for mp in media_files:
            txt = os.path.splitext(os.path.basename(mp))[0] + ".txt"
            skip_flags.append(skip_existing and os.path.exists(os.path.join(os.path.dirname(mp), txt)))

        # vLLM benefits from true batching; transformers is per-item.
        batch_size = p.batch_size if self.session.backend == "vllm" else 1

        start_time = time.time()
        self.session.reset_abort()

        state = {"processed": 0, "skipped": 0, "failed": 0, "aborted": 0, "last_caption": ""}
        pending: list[tuple[int, str]] = []  # (file_index, media_path)

        def _elapsed_str():
            e = int(time.time() - start_time)
            return f"{e // 60:02d}:{e % 60:02d}"

        def _flush():
            """Submit the pending batch to the backend and emit per-item results."""
            if not pending:
                return
            if self.session.is_aborted():
                # Aborted (user Stop, or an earlier batch timed out): do not submit
                # any more work to the GPU.
                pending.clear()
                return
            items = [(mp, prompt, max_tokens, resolution_mode) for _, mp in pending]
            log.debug(f"-> generating batch of {len(items)} "
                  f"(first: {os.path.basename(pending[0][1])})")

            # Run in a daemon thread with a hard timeout so a hung inference
            # call cannot freeze the run forever. In-process vLLM cannot be
            # interrupted mid-call, so on timeout we only *detect* it: mark the
            # batch failed and abort the run (we can't safely continue while a
            # background inference is still hammering the GPU).
            completed, box, thread = run_with_timeout(
                self.session.generate_batch, BATCH_TIMEOUT_SECONDS, items)
            self._inflight_thread = thread

            if not completed:
                log.error(f"Batch of {len(items)} did not finish within "
                          f"{BATCH_TIMEOUT_SECONDS // 60} min; inference appears stuck. Stopping.")
                self.session.abort()
                for idx, mp in pending:
                    state["failed"] += 1
                    rel = os.path.relpath(mp, folder_path)
                    self.sig_item.emit({
                        "kind": "error",
                        "status": (f"\u26d4 Stuck: {rel} \u2014 no response after "
                                   f"{BATCH_TIMEOUT_SECONDS // 60} min"),
                        "media_path": None, "is_image": False, "media_name": rel,
                        "caption": (f"Timed out after {BATCH_TIMEOUT_SECONDS // 60} min "
                                    f"(inference stuck). Restart if needed."),
                        "progress": int(((idx + 1) / total) * 100),
                        "elapsed": _elapsed_str(),
                        "vram": self.session.get_vram_string(), "tps": None, "tokens": 0,
                    })
                pending.clear()
                return

            if "exc" in box:
                e = box["exc"]
                log.error(f"Batch of {len(items)} failed: {e}")
                traceback.print_exception(type(e), e, e.__traceback__)
                for idx, mp in pending:
                    state["failed"] += 1
                    rel = os.path.relpath(mp, folder_path)
                    self.sig_item.emit({
                        "kind": "error",
                        "status": f"\u26a0\ufe0f Error: {rel} \u2014 {e}",
                        "media_path": None, "is_image": False, "media_name": rel,
                        "caption": f"Error: {e}",
                        "progress": int(((idx + 1) / total) * 100),
                        "elapsed": _elapsed_str(),
                        "vram": self.session.get_vram_string(), "tps": None, "tokens": 0,
                    })
                pending.clear()
                return

            results = box["result"]
            log.debug(f"<= got batch of {len(results)}")

            for (idx, mp), result in zip(pending, results):
                txt = os.path.splitext(os.path.basename(mp))[0] + ".txt"
                txt_path = os.path.join(os.path.dirname(mp), txt)
                rel = os.path.relpath(mp, folder_path)

                if result.aborted:
                    state["aborted"] += 1
                    self.sig_item.emit({
                        "kind": "aborted",
                        "status": f"\u23f9 Stopped: {rel}",
                        "media_path": None, "is_image": False, "media_name": rel,
                        "caption": "Stopped by user.",
                        "progress": int(((idx + 1) / total) * 100),
                        "elapsed": _elapsed_str(),
                        "vram": self.session.get_vram_string(), "tps": None, "tokens": 0,
                    })
                    continue

                if result.error:
                    state["failed"] += 1
                    self.sig_item.emit({
                        "kind": "error",
                        "status": f"\u26a0\ufe0f Error: {rel} \u2014 {result.error}",
                        "media_path": None, "is_image": False, "media_name": rel,
                        "caption": f"Error: {result.error}",
                        "progress": int(((idx + 1) / total) * 100),
                        "elapsed": _elapsed_str(),
                        "vram": self.session.get_vram_string(), "tps": None, "tokens": 0,
                    })
                    continue

                try:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(result.caption)
                except OSError as e:
                    state["failed"] += 1
                    self.sig_item.emit({
                        "kind": "error",
                        "status": f"\u26a0\ufe0f Write error: {rel} \u2014 {e}",
                        "media_path": None, "is_image": False, "media_name": rel,
                        "caption": f"Write error: {e}",
                        "progress": int(((idx + 1) / total) * 100),
                        "elapsed": _elapsed_str(),
                        "vram": self.session.get_vram_string(), "tps": None, "tokens": 0,
                    })
                    continue

                state["processed"] += 1
                state["last_caption"] = result.caption

                self.sig_item.emit({
                    "kind": "done",
                    "status": f"\U0001f589 {idx + 1}/{total}: {rel}",
                    "media_path": mp, "is_image": is_image_file(mp),
                    "media_name": rel,
                    "caption": result.caption,
                    "progress": int(((idx + 1) / total) * 100),
                    "elapsed": _elapsed_str(),
                    "vram": self.session.get_vram_string(),
                    "tps": result.tps, "tokens": result.tokens,
                })
            pending.clear()

        for idx, media_path in enumerate(media_files):
            if self.session.is_aborted():
                self.sig_item.emit({"kind": "abort", "status": "\u26d4 Aborted by user."})
                return

            if skip_flags[idx]:
                _flush()
                state["skipped"] += 1
                rel = os.path.relpath(media_path, folder_path)
                self.sig_item.emit({
                    "kind": "skip",
                    "status": f"\u23ed\ufe0f Skipped {idx + 1}/{total}: {rel}",
                    "media_path": None, "is_image": False, "media_name": None,
                    "caption": "Skipped (already captioned)",
                    "progress": int(((idx + 1) / total) * 100),
                    "elapsed": _elapsed_str(),
                    "vram": self.session.get_vram_string(), "tps": None, "tokens": 0,
                })
                continue

            if is_video_file(media_path):
                # Never share a batch with a video: a 16-item batch of up-to-2GB
                # videos would embed ~43GB of base64 in RAM before any call.
                # Flush accumulated images, then process the video alone.
                _flush()
                pending.append((idx, media_path))
                _flush()
            else:
                pending.append((idx, media_path))
                if len(pending) >= batch_size:
                    _flush()

        if not self.session.is_aborted():
            _flush()

        total_time = time.time() - start_time

        def _summary():
            parts = [f"{state['processed']} processed",
                     f"{state['skipped']} skipped",
                     f"{state['failed']} failed"]
            if state["aborted"]:
                parts.append(f"{state['aborted']} stopped")
            return ", ".join(parts)

        if self.session.is_aborted():
            # A batch timed out (or the user pressed Stop) at/after the final flush:
            # report a terminal "abort" instead of "Done" so a stuck final batch is
            # not claimed as a success.
            self.sig_item.emit({
                "kind": "abort",
                "status": f"\u26d4 Stopped: {_summary()} [{total_time:.1f}s]",
                "media_path": None, "is_image": False, "media_name": None,
                "caption": "Stopped (a batch timed out, or you pressed Stop). Restart if needed.",
                "progress": None, "elapsed": None,
                "vram": self.session.get_vram_string(), "tps": None, "tokens": 0,
            })
            return

        self.sig_item.emit({
            "kind": "complete",
            "status": f"\u2705 Done: {_summary()} [{total_time:.1f}s]",
            "media_path": None, "is_image": False, "media_name": None,
            "caption": state["last_caption"], "progress": None, "elapsed": None,
            "vram": self.session.get_vram_string(), "tps": None, "tokens": 0,
        })


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, app: QApplication, log_queue: queue.Queue | None = None):
        super().__init__()
        self.setWindowTitle("PxCaption v0.1.1")
        self.setWindowIcon(QIcon(ICON_PATH))
        self.setMinimumSize(1180, 660)
        self.resize(DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
        self._load_window_state()

        self.app = app
        self.session = ModelSession()
        self.model_worker = None
        self.caption_worker = None
        self._current_pixmap = None
        self._log_queue = log_queue if log_queue is not None else queue.Queue()

        self._build_ui()

        # Drain the thread-safe log queue on the GUI thread.
        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._drain_log_queue)
        self._log_timer.start(50)  # 50 ms polling

        self._show_model_info()
        self._set_processing(False)

    def _drain_log_queue(self):
        """Process buffered log lines on the GUI thread."""
        try:
            while True:
                line = self._log_queue.get_nowait()
                self.txt_log.appendPlainText(line)
        except queue.Empty:
            pass
        # Bound the log by dropping the oldest blocks *in place* rather than
        # rebuilding the whole text (the old toPlainText/setPlainText rebuilt the
        # pane and reset scroll + cursor on every trim). Removing from the top
        # keeps the view anchored near the newest lines.
        doc = self.txt_log.document()
        if doc.blockCount() > 5000:
            n_drop = doc.blockCount() - 2000
            cursor = QTextCursor(doc)
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.KeepAnchor, n_drop)
            cursor.removeSelectedText()

    def _section_header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        f = lbl.font()
        f.setBold(True)
        f.setPointSize(10)
        lbl.setFont(f)
        return lbl

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        # TOP BAR
        top = QWidget()
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(10)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMinimumHeight(24)
        top_lay.addWidget(self.progress, 1)

        time_box = QWidget()
        tb_lay = QHBoxLayout(time_box)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.setSpacing(4)
        tb_lay.addWidget(self._lbl("\u23f1\ufe0f"))
        self.txt_time = QLineEdit()
        self.txt_time.setReadOnly(True)
        self.txt_time.setFixedWidth(56)
        tb_lay.addWidget(self.txt_time)
        top_lay.addWidget(time_box)

        self.lbl_vram = QLabel("VRAM: \u2014")
        self.lbl_vram.setMinimumWidth(160)
        self.lbl_vram.setToolTip("GPU memory used / total.")
        top_lay.addWidget(self.lbl_vram)

        self.lbl_tps = QLabel("\U0001f525 \u2014 tok/s")
        self.lbl_tps.setMinimumWidth(150)
        self.lbl_tps.setToolTip("Tokens per second for the last generated caption.")
        top_lay.addWidget(self.lbl_tps)

        root.addWidget(top)

        # BODY
        body = QSplitter(Qt.Orientation.Horizontal)
        body.setChildrenCollapsible(False)

        # LEFT
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 4, 6, 4)
        ll.setSpacing(8)

        # Model settings
        settings = QGroupBox("\u2699\ufe0f Model")
        s_lay = QVBoxLayout(settings)
        s_lay.setSpacing(6)
        grid = QGridLayout()
        grid.setSpacing(8)

        self.cmb_model = QComboBox()
        self.cmb_model.setToolTip("Pick a model to use for captioning.")
        self._populate_model_combo()
        grid.addWidget(self._lbl("Model:"), 0, 0)
        grid.addWidget(self.cmb_model, 0, 1, 1, 2)

        self.cmb_backend = QComboBox()
        self.cmb_backend.addItems(["vLLM", "Transformers (+Torch Compile)", "Transformers"])
        self.cmb_backend.setCurrentText("vLLM")
        self.cmb_backend.setToolTip(
            "vLLM - Fastest inference (continuous batching, paged attention).\n"
            "Transformers (+Torch Compile) - PyTorch compile with disk cache.\n"
            "  First load is slower; subsequent loads reuse cached graphs.\n"
            "Transformers - Standard HuggingFace inference.")
        grid.addWidget(self._lbl("Backend:"), 1, 0)
        grid.addWidget(self.cmb_backend, 1, 1, 1, 2)

        self.cmb_quant = QComboBox()
        self.cmb_quant.addItems(["None", "8-bit", "4-bit (NF4)", "4-bit (FP4)"])
        self.cmb_quant.setCurrentText("None")
        self.cmb_quant.setToolTip(
            "BitsAndBytes quantization for the Transformers backend.\n"
            "NF4 = normalised float4 (best quality, default).\n"
            "FP4 = fixed-point 4-bit (slightly smaller, marginal speed gain).")
        grid.addWidget(self._lbl("Quantization:"), 2, 0)
        grid.addWidget(self.cmb_quant, 2, 1, 1, 2)

        self.cmb_attn = QComboBox()
        self.cmb_attn.addItems(["Flash Attention 2", "SDPA", "Eager"])
        self.cmb_attn.setCurrentText("Flash Attention 2")
        self.cmb_attn.setToolTip(
            "FA2 = fastest (requires flash-attn package).\n"
            "SDPA = PyTorch native, no extra package, faster than Eager.\n"
            "Eager = always works, slowest.\n"
            "Auto-fallback to Eager if the chosen backend is unavailable.")
        grid.addWidget(self._lbl("Attention:"), 3, 0)
        grid.addWidget(self.cmb_attn, 3, 1, 1, 2)

        self.cmb_ctx = QComboBox()
        self.cmb_ctx.addItems(CTX_PRESETS)
        self.cmb_ctx.setCurrentText(CTX_PRESETS[0])
        self.cmb_ctx.setToolTip(
            "Max context window (vLLM only).\n"
            "\n"
            "Auto = solved from your GPU: (VRAM \u00d7 0.90 \u2212 model weights \u2212 2 GB reserve) "
            "\u00f7 per-token KV-cache size gives the KV pool in tokens.\n"
            "The context is sized so a full batch of full-length sequences fits, but never below 8K.")
        grid.addWidget(self._lbl("Context:"), 4, 0)
        grid.addWidget(self.cmb_ctx, 4, 1, 1, 2)

        self.chk_kv_fp8 = QCheckBox("FP8 KV Cache")
        self.chk_kv_fp8.setToolTip(
            "Store the KV cache in FP8 instead of BF16 (vLLM only).\n"
            "\u2022 Halves KV-cache VRAM usage\n"
            "\u2022 Doubles the auto-computed context window\n"
            "\u2022 Negligible quality impact on caption-length outputs")
        grid.addWidget(self.chk_kv_fp8, 5, 0, 1, 3)

        self.cmb_backend.currentTextChanged.connect(self._update_backend_gated_controls)
        self._update_backend_gated_controls()

        btn_row = QWidget()
        btn_row_lay = QHBoxLayout(btn_row)
        btn_row_lay.setContentsMargins(0, 0, 0, 0)
        btn_row_lay.setSpacing(6)
        self.btn_load = QPushButton("\U0001f4e6 Load")
        self.btn_load.setToolTip("Load the selected model into the GPU")
        self.btn_load.clicked.connect(self.on_load_model)
        self.btn_import = QPushButton("\U0001f4e5 Import")
        self.btn_import.setToolTip("Import a local model folder into the store.\n"
            "\n"
            "A valid model is a folder containing config.json (or model_index.json)\n"
            "and at least one weight file (*.safetensors / *.bin) that passes the index check.")
        self.btn_import.clicked.connect(self.on_import_model)
        self.btn_remove = QPushButton("🗑 Remove")
        self.btn_remove.setToolTip("Remove the selected model from the store.")
        self.btn_remove.clicked.connect(self.on_remove_model)
        for _b in (self.btn_load, self.btn_import, self.btn_remove):
            _b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            _b.setMinimumWidth(110)
        btn_row_lay.addStretch(1)
        btn_row_lay.addWidget(self.btn_load, 1)
        btn_row_lay.addWidget(self.btn_import, 1)
        btn_row_lay.addWidget(self.btn_remove, 1)
        btn_row_lay.addStretch(1)
        grid.addWidget(btn_row, 6, 0, 1, 3)

        s_lay.addLayout(grid)
        ll.addWidget(settings)

        # Dataset
        ds = QGroupBox("\U0001f4c1 Dataset")
        ds_lay = QVBoxLayout(ds)
        ds_lay.setSpacing(6)
        self.btn_browse = QPushButton("\U0001f4c2 Browse\u2026")
        self.btn_browse.clicked.connect(self.on_browse_folder)
        self.txt_folder = QLineEdit()
        self.txt_folder.setPlaceholderText("e.g. C:\\Dataset")
        browse_row = QWidget()
        browse_row_lay = QHBoxLayout(browse_row)
        browse_row_lay.setContentsMargins(0, 0, 0, 0)
        browse_row_lay.setSpacing(6)
        browse_row_lay.addWidget(self.btn_browse)
        browse_row_lay.addWidget(self.txt_folder, 1)
        ds_lay.addWidget(browse_row)
        self.chk_skip_existing = QCheckBox("Skip existing captions")
        ds_lay.addWidget(self.chk_skip_existing)
        self.chk_retain_preview = QCheckBox("Retain preview on skip")
        self.chk_retain_preview.setChecked(True)
        self.chk_retain_preview.setToolTip("Keep the last shown image/caption when a file is skipped.")
        ds_lay.addWidget(self.chk_retain_preview)
        ll.addWidget(ds)

        # Prompt
        pt = QGroupBox("\U0001f4dd Prompt")
        pt_lay = QVBoxLayout(pt)
        pt_lay.setSpacing(6)
        self.txt_prompt = QPlainTextEdit()
        self.txt_prompt.setPlainText(DEFAULT_PROMPT)
        self.txt_prompt.setMinimumHeight(88)
        pt_lay.addWidget(self.txt_prompt)
        ll.addWidget(pt)

        # Processing
        proc = QGroupBox("\U0001f39b\ufe0f Processing")
        proc_lay = QVBoxLayout(proc)
        proc_lay.setSpacing(6)

        tr = QWidget()
        tr_lay = QHBoxLayout(tr)
        tr_lay.setContentsMargins(0, 0, 0, 0)
        self.sld_max_tokens = QSlider(Qt.Orientation.Horizontal)
        self.sld_max_tokens.setRange(1, 64)
        self.sld_max_tokens.setSingleStep(1)
        self.sld_max_tokens.setPageStep(1)
        self.sld_max_tokens.setValue(16)
        self.lbl_max_tokens = QLabel("512")
        self.lbl_max_tokens.setMinimumWidth(36)
        self.sld_max_tokens.valueChanged.connect(
            lambda v: self.lbl_max_tokens.setText(str(v * 32))
        )
        tr_lay.addWidget(self._lbl("Max Output Tokens:"))
        tr_lay.addWidget(self.sld_max_tokens, 1)
        tr_lay.addWidget(self.lbl_max_tokens)
        proc_lay.addWidget(tr)

        bs_row = QWidget()
        bs_lay = QHBoxLayout(bs_row)
        bs_lay.setContentsMargins(0, 0, 0, 0)
        bs_lay.setSpacing(6)
        self.cmb_batch = QComboBox()
        for bs in BATCH_SIZE_OPTIONS:
            self.cmb_batch.addItem(str(bs), bs)
        self.cmb_batch.setCurrentIndex(len(BATCH_SIZE_OPTIONS) - 1)
        self.cmb_batch.setToolTip(
            "vLLM batch size\n"
            "How many items to submit in one inference call.\n"
            "Higher = faster throughput but more VRAM per batch.\n"
            "Also affects Auto context sizing (larger batch → shorter context)."
        )
        bs_lay.addWidget(self._lbl("Batch Size:"))
        bs_lay.addWidget(self.cmb_batch)
        bs_lay.addStretch(1)
        proc_lay.addWidget(bs_row)

        res_row = QWidget()
        res_lay = QHBoxLayout(res_row)
        res_lay.setContentsMargins(0, 0, 0, 0)
        self.cmb_resolution = QComboBox()
        self.cmb_resolution.addItems(["draft", "standard", "high", "ultra"])
        self.cmb_resolution.setCurrentText("high")
        self.cmb_resolution.setToolTip(
            "Sets the image's pixel budget, which sets the number of vision tokens.\n"
            "Lower = faster and less VRAM; higher = more detail.\n"
            "All modes preserve aspect ratio.")
        _res_tips = {
            "draft":    "Smallest. \u2248153,664 px (\u2248392\u00d7392) cap. "
                        "Fewest tokens, fastest, least VRAM; some detail lost.",
            "standard": "Medium. \u2248529,984 px (\u2248728\u00d7728) cap. "
                        "Good for general images.",
            "high":     "Large. \u2248702,464 px (\u2248838\u00d7838) cap. "
                        "Balanced quality and speed.",
            "ultra":    "Default. \u22481,003,520 px (\u22481002\u00d71002) cap. "
                        "Most detail; best for text, faces, fine detail. "
                        "Uses the most VRAM.",
        }
        _res_model = self.cmb_resolution.model()
        for _i, _name in enumerate(["draft", "standard", "high", "ultra"]):
            _res_model.setData(_res_model.index(_i, 0), _res_tips[_name],
                               Qt.ItemDataRole.ToolTipRole)
        res_lay.addWidget(self._lbl("Resolution:"))
        res_lay.addWidget(self.cmb_resolution)
        res_lay.addStretch(1)
        proc_lay.addWidget(res_row)

        br = QWidget()
        br_lay = QHBoxLayout(br)
        br_lay.setContentsMargins(0, 0, 0, 0)
        br_lay.setSpacing(6)
        self.btn_start = QPushButton("\U0001f7e2 Start")
        self.btn_start.clicked.connect(self.on_start)
        self.btn_abort = QPushButton("\u26d4 Stop")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self.on_abort)
        for _b in (self.btn_start, self.btn_abort):
            _b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            _b.setMinimumWidth(90)
        br_lay.addStretch(1)
        br_lay.addWidget(self.btn_start, 1)
        br_lay.addWidget(self.btn_abort, 1)
        br_lay.addStretch(1)
        proc_lay.addWidget(br)
        ll.addWidget(proc)

        # Status
        status_box = QGroupBox("Status")
        status_lay = QVBoxLayout(status_box)
        status_lay.setSpacing(4)
        self.txt_status = QLineEdit()
        self.txt_status.setReadOnly(True)
        status_lay.addWidget(self.txt_status)
        ll.addWidget(status_box)

        # Model info
        info = CollapsibleGroup("\U0001f4ca Model Information", default_collapsed=True)
        i_lay = info.content_layout()
        i_grid = QGridLayout()
        self.txt_model_name = self._readonly_line()
        self.txt_device = self._readonly_line()
        self.txt_dtype = self._readonly_line()
        self.txt_config = QPlainTextEdit()
        self.txt_config.setReadOnly(True)
        self.txt_config.setFixedHeight(90)
        i_grid.addWidget(self._lbl("Model:"), 0, 0)
        i_grid.addWidget(self.txt_model_name, 0, 1)
        i_grid.addWidget(self._lbl("Device:"), 1, 0)
        i_grid.addWidget(self.txt_device, 1, 1)
        i_grid.addWidget(self._lbl("Dtype:"), 2, 0)
        i_grid.addWidget(self.txt_dtype, 2, 1)
        i_grid.addWidget(self._lbl("Config:"), 3, 0)
        i_grid.addWidget(self.txt_config, 3, 1)
        i_lay.addLayout(i_grid)
        i_lay.addStretch(1)
        ll.addWidget(info)

        # Log
        log = CollapsibleGroup("\U0001f4dc Log", default_collapsed=True)
        log_lay = log.content_layout()
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(160)
        lf = self.txt_log.font()
        lf.setFamily("Consolas")
        lf.setPointSize(9)
        self.txt_log.setFont(lf)
        log_lay.addWidget(self.txt_log)
        ll.addWidget(log)

        ll.addStretch(1)
        left.setMinimumWidth(300)
        self.scroll.setWidget(left)
        body.addWidget(self.scroll)

        # CENTER
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)
        cl.addWidget(self._section_header("Current Image"))

        self.lbl_image = _AutoFitLabel("\u2014")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(320, 240)
        self.lbl_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_image.setFrameShape(QLabel.Shape.Box)
        self.lbl_image.setStyleSheet("background: transparent; color: #999; font-size: 24px;")
        self.lbl_image.set_resize_callback(self._fit_preview)
        cl.addWidget(self.lbl_image, 1)

        self.lbl_media_name = QLabel("")
        self.lbl_media_name.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_media_name.setWordWrap(True)
        self.lbl_media_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(self.lbl_media_name)

        body.addWidget(center)

        # RIGHT
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        rl.addWidget(self._section_header("Generated Caption"))
        self.txt_caption = QPlainTextEdit()
        self.txt_caption.setReadOnly(True)
        self.txt_caption.setMinimumWidth(220)
        rl.addWidget(self.txt_caption, 1)
        body.addWidget(right)

        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 5)
        body.setStretchFactor(2, 3)
        body.setSizes([320, 680, 380])

        root.addWidget(body, 1)

    @staticmethod
    def _lbl(text: str) -> QLabel:
        return QLabel(text)

    @staticmethod
    def _readonly_line() -> QLineEdit:
        e = QLineEdit()
        e.setReadOnly(True)
        return e

    def _populate_model_combo(self):
        names = sorted((m["name"] for m in MODEL_STORE), key=str.lower)
        current = self.cmb_model.currentText()
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        self.cmb_model.addItems(names)
        if current in names:
            self.cmb_model.setCurrentText(current)
        self.cmb_model.blockSignals(False)

    def _update_backend_gated_controls(self):
        """Swap backend-specific controls when the backend selection changes.

        Quantization items change per backend:
          vLLM           \u2192 None / AWQ / GPTQ  (checkpoint-level, auto-detected)
          Transformers*  \u2192 None / 8-bit / 4-bit (NF4) / 4-bit (FP4)  (runtime BnB)
        Context, FP8 KV cache, and batch size are vLLM-only.
        """
        use_vllm = self.cmb_backend.currentText() == "vLLM"

        # --- Quantization: swap items to show only compatible options ---
        self.cmb_quant.blockSignals(True)
        self.cmb_quant.clear()
        if use_vllm:
            self.cmb_quant.addItems(["None", "AWQ", "GPTQ"])
            self.cmb_quant.setCurrentText("None")
            self.cmb_quant.setToolTip(
                "Checkpoint quantization \u2014 auto-detected from the model\u2019s config.json.\n"
                "AWQ and GPTQ use optimised vLLM CUDA kernels.\n"
                "Import a pre-quantised model folder to use AWQ or GPTQ.")
        else:
            self.cmb_quant.addItems(["None", "8-bit", "4-bit (NF4)", "4-bit (FP4)"])
            self.cmb_quant.setCurrentText("None")
            self.cmb_quant.setToolTip(
                "BitsAndBytes runtime quantization (Transformers backend).\n"
                "NF4 = normalised float4 (best quality, default).\n"
                "FP4 = fixed-point 4-bit (slightly smaller, marginal speed gain).")
        self.cmb_quant.blockSignals(False)

        # --- vLLM-only controls ---
        self.cmb_ctx.setEnabled(use_vllm)
        self.chk_kv_fp8.setEnabled(use_vllm)
        if hasattr(self, "cmb_batch"):
            self.cmb_batch.setEnabled(use_vllm)
            if use_vllm:
                self.cmb_batch.setCurrentIndex(
                    BATCH_SIZE_OPTIONS.index(DEFAULT_BATCH_SIZE))  # default to 16
            else:
                self.cmb_batch.setCurrentIndex(0)  # force to 1

    def _set_processing(self, processing: bool):
        use_vllm = self.cmb_backend.currentText() == "vLLM"
        controls = (
            self.cmb_model, self.cmb_backend, self.cmb_attn, self.cmb_quant,
            self.btn_load, self.btn_import, self.btn_remove, self.btn_start,
            self.txt_folder, self.btn_browse, self.txt_prompt,
            self.chk_skip_existing, self.sld_max_tokens,
            self.chk_retain_preview, self.cmb_resolution,
        )
        for w in controls:
            w.setEnabled(not processing)
        # Backend-gated controls: additionally gated by backend selection.
        self.cmb_batch.setEnabled((not processing) and use_vllm)
        self.cmb_ctx.setEnabled((not processing) and use_vllm)
        self.chk_kv_fp8.setEnabled((not processing) and use_vllm)
        self.btn_abort.setEnabled(processing)

    def _show_model_info(self):
        info = self.session.get_info()
        self.txt_model_name.setText(info.name)
        self.txt_device.setText(info.device)
        self.txt_dtype.setText(info.dtype)
        self.txt_config.setPlainText(info.config)
        self.lbl_vram.setText(f"VRAM: {self.session.get_vram_string()}")

    def _reset_stats_bar(self):
        self.progress.setValue(0)
        self.txt_time.clear()
        self.lbl_vram.setText("VRAM: \u2014")
        self.lbl_tps.setText("\U0001f525 \u2014 tok/s")

    def _clear_preview(self):
        self._current_pixmap = None
        self.lbl_image.clear()
        self.lbl_image.setText("\u2014")
        self.lbl_media_name.setText("")

    def _show_preview(self, media_path: str, rel_path: str):
        try:
            # Use Qt's native image loader — it reads the file, decodes, and
            # allocates its own pixel buffer in one step. Avoids the PIL→QImage
            # pointer handoff that caused 0xC0000005 crashes (the QImage
            # (bytes,…) ctor stores a raw pointer into Python memory that can
            # dangle).
            pixmap = QPixmap(media_path)
            if not pixmap.isNull():
                self._current_pixmap = pixmap
                self._fit_preview()
            else:
                log.error(f"Qt could not decode preview: {media_path}")
        except Exception as e:
            log.error(f"Could not load preview for {media_path}: {e}")
        self.lbl_media_name.setText(f"<b>File:</b> <code>{rel_path}</code>")

    def _fit_preview(self):
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        target = self.lbl_image.size()
        if target.width() <= 1 or target.height() <= 1:
            return
        scaled = self._current_pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_image.setPixmap(scaled)

    def _load_window_state(self):
        try:
            if os.path.exists(WINDOW_STATE_PATH):
                with open(WINDOW_STATE_PATH, "r") as f:
                    data = json.load(f)
                x, y = data.get("x", 0), data.get("y", 0)
                w, h = data.get("w", 0), data.get("h", 0)
                if w > 0 and h > 0:
                    self.setGeometry(x, y, w, h)
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            geo = self.geometry()
            _atomic_write_json(WINDOW_STATE_PATH, {
                "x": geo.x(), "y": geo.y(),
                "w": geo.width(), "h": geo.height(),
            })
        except Exception:
            pass
        super().closeEvent(event)

    # ----- slots -----

    def on_load_model(self):
        if self.caption_worker is not None and self.caption_worker.inference_in_flight():
            self.txt_status.setText(
                "\u23f3 A caption batch is still running in the background "
                "(it timed out). Wait for it to finish before loading another model."
            )
            return
        if self.model_worker is not None:
            if self.model_worker.isRunning():
                return
            if self.model_worker.load_in_flight():
                self.txt_status.setText(
                    "\u23f3 A previous load is still in flight (possibly stuck). "
                    "Wait for it to finish before loading another model."
                )
                return
        model_id = self.cmb_model.currentText()
        if not model_id:
            self.txt_status.setText("\u274c Please pick a model (import one first).")
            return
        backend_text = self.cmb_backend.currentText()
        use_vllm = backend_text == "vLLM"
        use_torch_compile = backend_text == "Transformers (+Torch Compile)"

        q = self.cmb_quant.currentText()
        if use_vllm:
            _QUANT_MAP = {
                "None": "None (vLLM)",
                "AWQ":    "AWQ (vLLM)",
                "GPTQ":   "GPTQ (vLLM)",
            }
        else:
            _QUANT_MAP = {
                "None":        "None (Transformers)",
                "8-bit":       "8-bit (Transformers)",
                "4-bit (NF4)": "4-bit (Transformers)",
                "4-bit (FP4)": "4-bit FP4 (Transformers)",
            }
        quant = _QUANT_MAP.get(q, "None (Transformers)")
        _ATTN_MAP = {
            "Flash Attention 2": "flash_attention_2",
            "SDPA":              "sdpa",
            "Eager":             "eager",
        }
        attn = _ATTN_MAP.get(self.cmb_attn.currentText(), "eager")
        ctx_text = self.cmb_ctx.currentText()
        max_model_len = None if ctx_text.startswith("Auto") else int(ctx_text.rstrip("K")) * 1024

        self.txt_status.setText(f"\u23f3 Loading '{model_id}'\u2026")
        self.btn_load.setEnabled(False)
        self._set_processing(True)

        batch_size = self.cmb_batch.currentData()
        self.model_worker = ModelLoadWorker(
            self.session, model_id, quant, attn, max_model_len,
            batch_size, self.chk_kv_fp8.isChecked(), use_torch_compile, self
        )
        self.model_worker.sig_done.connect(self._on_model_loaded)
        self.model_worker.sig_error.connect(self._on_model_error)
        self.model_worker.start()

    def on_import_model(self):
        if self.model_worker is not None and self.model_worker.isRunning():
            self.txt_status.setText("\u23f3 Please wait for the current model load to finish.")
            return

        start_dir = self.txt_folder.text().strip() or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select a model folder to import", start_dir)
        if not folder:
            return

        is_valid, message = validate_model_folder(folder)
        if not is_valid:
            QMessageBox.critical(self, "Invalid model folder", f"\u274c {message}\n\n{folder}")
            self.txt_status.setText(f"\u274c Import failed: {message}")
            return

        existing = next((m for m in MODEL_STORE if m["path"] == folder), None)
        if existing:
            self._populate_model_combo()
            self.cmb_model.setCurrentText(existing["name"])
            self.txt_status.setText(f"\u2139\ufe0f '{existing['name']}' is already in the model store.")
            return

        base = os.path.basename(os.path.normpath(folder)) or folder
        existing_names = {m["name"] for m in MODEL_STORE}
        name = base
        n = 2
        while name in existing_names:
            name = f"{base} ({n})"
            n += 1

        MODEL_STORE.append({"name": name, "path": folder})
        try:
            save_model_store(MODEL_STORE)
        except OSError as e:
            MODEL_STORE.pop()
            QMessageBox.critical(self, "Save failed", f"Could not save the model store:\n{e}")
            self.txt_status.setText(f"\u274c Could not save the model store: {e}")
            return

        self._populate_model_combo()
        self.cmb_model.setCurrentText(name)
        self.txt_status.setText(f"\u2705 Imported '{name}' \u2014 {message}")

    def on_remove_model(self):
        name = self.cmb_model.currentText()
        match = next((m for m in MODEL_STORE if m["name"] == name), None)
        if match is None:
            self.txt_status.setText("No model selected to remove.")
            return

        MODEL_STORE.remove(match)
        try:
            save_model_store(MODEL_STORE)
        except OSError as e:
            MODEL_STORE.append(match)
            self.txt_status.setText("Could not save the model store: " + str(e))
            return

        self._populate_model_combo()
        self.txt_status.setText("Removed '" + name + "' from the model store.")

    def on_browse_folder(self):
        start_dir = self.txt_folder.text().strip() or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select dataset folder", start_dir)
        if folder:
            self.txt_folder.setText(folder)

    def _on_model_loaded(self, status, name, device, vram, dtype, cfg, backend):
        self.txt_status.setText(status)
        self.txt_model_name.setText(name)
        self.txt_device.setText(device)
        self.txt_dtype.setText(dtype)
        self.txt_config.setPlainText(cfg)
        self.lbl_vram.setText(f"VRAM: {vram}")
        self._set_processing(False)

    def _on_model_error(self, status: str):
        self.txt_status.setText(status)
        self._set_processing(False)

    def on_start(self):
        if self.caption_worker is not None and self.caption_worker.isRunning():
            return
        if self.caption_worker is not None and self.caption_worker.inference_in_flight():
            self.txt_status.setText(
                "\u23f3 A previous batch is still running in the background "
                "(it timed out). Wait for it to finish before starting again."
            )
            return
        if self.session.model is None:
            self.txt_status.setText("\u274c Please load a model before processing.")
            return

        self._show_model_info()
        self._clear_preview()
        self.txt_caption.clear()
        self._reset_stats_bar()
        self._set_processing(True)

        params = CaptionParams(
            folder=self.txt_folder.text(),
            prompt=self.txt_prompt.toPlainText().strip(),
            skip_existing=self.chk_skip_existing.isChecked(),
            max_tokens=self.sld_max_tokens.value() * 32,
            batch_size=self.cmb_batch.currentData(),
            resolution_mode=self.cmb_resolution.currentText(),
        )

        self.caption_worker = CaptionWorker(self.session, params, self)
        self.caption_worker.sig_item.connect(self._on_item)
        self.caption_worker.start()

    def on_abort(self):
        self.session.abort()
        self.txt_status.setText("\u26d4 Aborting process...")

    def _update_stats(self, item: dict):
        if item.get("elapsed") is not None:
            self.txt_time.setText(item["elapsed"])
        if item.get("vram"):
            self.lbl_vram.setText(f"VRAM: {item['vram']}")
        tps = item.get("tps")
        tokens = item.get("tokens") or 0
        if tps is not None:
            self.lbl_tps.setText(f"\U0001f525 {tps:.1f} tok/s ({tokens})")
        if item.get("progress") is not None:
            self.progress.setValue(item["progress"])

    def _on_item(self, item: dict):
        kind = item["kind"]

        if kind == "skip":
            self.txt_status.setText(item["status"])
            if not self.chk_retain_preview.isChecked():
                self._clear_preview()
                self.txt_caption.setPlainText(item["caption"])
            self._update_stats(item)
            return

        if kind == "done":
            self.txt_status.setText(item["status"])
            if item.get("is_image") and item.get("media_path"):
                self._show_preview(item["media_path"], item["media_name"])
            self.txt_caption.setPlainText(item["caption"])
            self._update_stats(item)
            return

        if kind == "aborted":
            # Per-item "user stopped this one" -- not a failure. Update the bar and
            # keep processing state active (the run continues).
            self.txt_status.setText(item["status"])
            self._clear_preview()
            self.txt_caption.setPlainText(item["caption"])
            self._update_stats(item)
            return

        if kind == "error":
            self.txt_status.setText(item["status"])
            self._clear_preview()
            self.txt_caption.setPlainText(item["caption"])
            self._update_stats(item)
            return

        # Terminal events
        self.txt_status.setText(item["status"])
        if kind == "abort":
            self._clear_preview()
            self.txt_caption.setPlainText(item.get("caption", "Aborted."))
        elif kind == "no_media":
            self._clear_preview()
            self.txt_caption.setPlainText("No media to process.")

        if kind in ("abort", "no_media"):
            self._reset_stats_bar()
        else:
            self.progress.setValue(0)
            self.txt_time.clear()

        self._set_processing(False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Dump a C-level stack trace on a native crash (segfault / abort / fatal
    # signal). faulthandler is enabled *before* the stdout/stderr redirect is
    # installed below, so it writes to the real fd 2 -- i.e. the console window,
    # NOT the GUI Log pane. That is deliberate: PxCaption.bat keeps the console
    # open on a non-zero exit (via "pause"), so a native crash leaves the C stack
    # trace visible there, showing exactly which file / C call (PIL / CUDA /
    # ffmpeg) is responsible.
    faulthandler.enable()

    app = QApplication(sys.argv)
    app.setApplicationName("PxCaption v0.1")

    window = MainWindow(app)

    # Route the module logger into the log pane (DIAG -> DEBUG level).
    # Third-party prints (vLLM internals, etc.) still arrive via the stdout/
    # stderr redirect below, so the pane shows both our structured logs and
    # whatever the backends print.
    configure_logging(window._log_queue, DIAG)
    sys.stdout = _PrintRedirect(sys.stdout, window._log_queue)
    sys.stderr = _PrintRedirect(sys.stderr, window._log_queue)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
