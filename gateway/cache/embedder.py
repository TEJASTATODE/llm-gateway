import asyncio
import hashlib
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Local embedding model — runs on CPU, zero API calls, zero cost.

    WHY LOCAL:
    Cache must be the most reliable layer in the system.
    If an API-based embedder goes down, every request becomes
    a cache miss and costs spike immediately.
    Local model has zero external dependencies — works offline,
    no quota, no billing, consistent forever.

    MODEL: all-MiniLM-L6-v2
    - 90MB, runs on CPU
    - 384 dimensions
    - ~50ms per embedding
    - Industry standard for semantic similarity tasks
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print("[Embedder] Loading local model — one time only...")
        self._model = SentenceTransformer(self.MODEL_NAME)
        self._local_cache: dict[str, list[float]] = {}
        print("[Embedder] Ready")

    async def embed(self, text: str) -> list[float]:
        """
        Convert text to vector.
        Checks in-memory cache first — avoids re-embedding
        same text twice in one server session.
        """
        if text in self._local_cache:
            return self._local_cache[text]

        vector = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._model.encode(text).tolist()
        )

        self._local_cache[text] = vector
        return vector

    def cosine_similarity(
        self,
        vec_a: list[float],
        vec_b: list[float]
    ) -> float:
        """
        Measures similarity between two vectors.
        Returns 0.0 to 1.0 — higher means more similar meaning.

        WHY COSINE NOT EUCLIDEAN:
        Cosine measures the angle between vectors — direction matters.
        Euclidean measures distance between points — magnitude matters.
        For text meaning, two sentences can be different lengths
        but same meaning. Cosine handles this correctly.

        MATH:
        similarity = dot_product / (magnitude_a x magnitude_b)
        """
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return max(0.0, min(1.0, dot / (mag_a * mag_b)))

    def text_hash(self, text: str) -> str:
        """
        SHA256 hash for exact-match Redis layer.
        Checked before vector search — identical text
        returns instantly without embedding call.
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]