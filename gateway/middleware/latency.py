import time
from dataclasses import dataclass, field


@dataclass
class LatencyTracker:
    """
    Tracks latency at each stage of a single request.
    Passed through the request lifecycle, populated at each step.

    WHY PER-STAGE:
    Total latency tells you the request was slow.
    Per-stage latency tells you WHERE it was slow.
    Cache slow? Embedding model. Provider slow? Network.
    This is what you need to actually debug production issues.
    """
    request_start: float = field(default_factory=time.monotonic)

    # Stage timings — set as each stage completes
    cache_lookup_ms: int = 0
    embedding_ms: int = 0
    provider_ms: int = 0
    cache_store_ms: int = 0

    # Stage flags
    cache_hit: bool = False
    cache_layer: str = ""  # "redis" | "qdrant" | ""

    def mark_cache_start(self):
        self._cache_start = time.monotonic()

    def mark_cache_end(self, hit: bool, layer: str = ""):
        self.cache_lookup_ms = int((time.monotonic() - self._cache_start) * 1000)
        self.cache_hit = hit
        self.cache_layer = layer

    def mark_provider_start(self):
        self._provider_start = time.monotonic()

    def mark_provider_end(self):
        self.provider_ms = int((time.monotonic() - self._provider_start) * 1000)

    def total_ms(self) -> int:
        return int((time.monotonic() - self.request_start) * 1000)

    def to_dict(self) -> dict:
        return {
            "total_ms":         self.total_ms(),
            "cache_lookup_ms":  self.cache_lookup_ms,
            "provider_ms":      self.provider_ms,
            "cache_hit":        self.cache_hit,
            "cache_layer":      self.cache_layer,
        }