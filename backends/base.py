"""Abstract base class for PxCaption backends."""
from abc import ABC, abstractmethod
import threading

from shared import CaptionResult, LoadRequest, ModelInfo


class BaseBackend(ABC):
    """Interface that all backends must implement."""

    name: str = "base"

    @property
    @abstractmethod
    def model(self):
        """The loaded model object (LLM, AutoModel, etc.). None if unloaded."""

    @abstractmethod
    def load(self, **kwargs):
        """Load the model. Raises on failure."""

    @abstractmethod
    def can_reuse(self, desired: LoadRequest) -> bool:
        """True if the currently loaded model already satisfies this request,
        so the caller can skip a reload. Reuse is backend-specific: vLLM also
        requires the same resolved context and KV-cache dtype, while the
        Transformers backend tracks model + quant + attention + compile state."""

    @abstractmethod
    def unload(self):
        """Unload the model and perform backend-specific teardown."""

    @abstractmethod
    def generate(self, media_path: str, prompt: str, max_tokens: int,
                 resolution_mode: str) -> CaptionResult:
        """Generate a caption for a single media file."""

    @abstractmethod
    def generate_batch(self, items: list[tuple[str, str, int, str]],
                       abort_event: threading.Event) -> list[CaptionResult]:
        """Generate captions for a batch of items."""

    @abstractmethod
    def get_info(self, model_id: str | None) -> ModelInfo:
        """Return a ``ModelInfo`` snapshot of the loaded model's runtime state."""
