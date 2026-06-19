import asyncio
import json
import time
import redis.asyncio as aioredis
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from gateway.cache.domain_detector import DomainDetector
from gateway.cache.context_builder import ContextBuilder
from gateway.cache.embedder import Embedder
from gateway.config import settings


COLLECTION_NAME = "llm_cache"


class CacheEntry:
    """Represents a cache hit result"""
    def __init__(
        self,
        response: dict,
        domain: str,
        similarity: float,
        model: str,
        cache_layer: str = "",  # "redis" | "qdrant"
    ):
        self.response = response
        self.domain = domain
        self.similarity = similarity
        self.model = model
        self.cache_layer = cache_layer


class SemanticCache:
    """
    Two-layer semantic cache with context-awareness and
    domain-adaptive thresholds.

    LAYER 1 — Redis exact match (microseconds)
    Hash the context key. If same hash exists in Redis,
    return instantly. No embedding needed.
    Handles: repeated identical requests (very common in production)

    LAYER 2 — Qdrant vector search (milliseconds)
    Embed the context key. Search for similar vectors.
    If similarity > domain threshold, return cached response.
    Handles: same question rephrased or in similar context

    IMPROVEMENT 1 — Context-aware keys
    Cache key includes last 3 messages, not just current message.
    Same question in different conversation context = different key.

    IMPROVEMENT 2 — Domain-adaptive thresholds
    Code questions need 0.96 similarity to hit cache.
    Conversational questions only need 0.70.
    Prevents wrong answers for precise domains.
    """

    def __init__(self):
        self.domain_detector = DomainDetector()
        self.context_builder = ContextBuilder(context_window=3)
        self.embedder = Embedder()
        self.redis = None
        self.qdrant = None
        self._stats = {
            "total_lookups": 0,
            "redis_hits": 0,
            "qdrant_hits": 0,
            "misses": 0,
        }

    async def init(self):
        """
        Initialize Redis and Qdrant connections.
        Called once at server startup.
        Creates Qdrant collection if it doesn't exist.
        """
        self.redis = await aioredis.from_url(
            settings.redis_url,
            decode_responses=True
        )
        await self.redis.ping()
        print("[Cache] Redis connected")

        self.qdrant = QdrantClient(url=settings.qdrant_url)

        existing = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[Cache] Qdrant collection '{COLLECTION_NAME}' created")
        else:
            print(f"[Cache] Qdrant collection '{COLLECTION_NAME}' ready")

    async def lookup(
        self,
        messages: list,
        user_id: str = "default",
    ) -> CacheEntry | None:
        """
        Main cache lookup — two layers.
        Returns CacheEntry if hit, None if miss.
        CacheEntry now includes cache_layer field for logging.
        """
        self._stats["total_lookups"] += 1

        # Step 1 — build context key (Improvement 1)
        context_key = self.context_builder.build_key(messages)
        if not context_key:
            return None

        # Step 2 — detect domain + get threshold (Improvement 2)
        last_message = self.context_builder.get_last_user_message(messages)
        domain = self.domain_detector.detect(last_message)
        threshold = self.domain_detector.get_threshold(domain)

        print(f"[Cache] Domain: {domain.value} | Threshold: {threshold}")
        print(f"[Cache] Context key: {context_key[:80]}...")

        # Step 3 — Redis exact match (Layer 1)
        text_hash = self.embedder.text_hash(context_key)
        redis_key = f"llm_cache:{text_hash}"

        cached_json = await self.redis.get(redis_key)
        if cached_json:
            self._stats["redis_hits"] += 1
            print("[Cache] REDIS HIT — exact match")
            cached = json.loads(cached_json)
            return CacheEntry(
                response=cached["response"],
                domain=domain.value,
                similarity=1.0,
                model=cached.get("model", "unknown"),
                cache_layer="redis",      # ← tracked
            )

        # Step 4 — Qdrant vector search (Layer 2)
        vector = await self.embedder.embed(context_key)

        result = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=1,
        )

        points = result.points

        if points and points[0].score >= threshold:
            self._stats["qdrant_hits"] += 1
            print(
                f"[Cache] QDRANT HIT — similarity: "
                f"{points[0].score:.4f} >= {threshold}"
            )
            payload = points[0].payload
            return CacheEntry(
                response=payload["response"],
                domain=domain.value,
                similarity=points[0].score,
                model=payload.get("model", "unknown"),
                cache_layer="qdrant",     # ← tracked
            )

        # Step 5 — miss
        self._stats["misses"] += 1
        print("[Cache] MISS — calling provider")
        return None

    async def store(
        self,
        messages: list,
        response: dict,
        domain: str,
        user_id: str = "default",
    ):
        """
        Store response in both cache layers async.
        Called after provider response — never blocks.
        """
        context_key = self.context_builder.build_key(messages)
        if not context_key:
            return

        last_message = self.context_builder.get_last_user_message(messages)
        domain_enum = self.domain_detector.detect(last_message)

        ttl_map = {
            "code":           86400 * 7,  # 7 days
            "factual":        86400,      # 1 day
            "conceptual":     86400,      # 1 day
            "conversational": 1800,       # 30 min
        }
        ttl = ttl_map.get(domain_enum.value, 3600)

        cache_payload = {
            "response": response,
            "model": response.get("model", "unknown"),
            "domain": domain_enum.value,
            "stored_at": time.time(),
        }

        # Layer 1 — Redis exact match
        text_hash = self.embedder.text_hash(context_key)
        redis_key = f"llm_cache:{text_hash}"
        await self.redis.setex(
            redis_key,
            ttl,
            json.dumps(cache_payload)
        )

        # Layer 2 — Qdrant vector search
        vector = await self.embedder.embed(context_key)
        point_id = abs(hash(context_key)) % (2 ** 53)

        self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=cache_payload,
                )
            ]
        )

        print(f"[Cache] Stored — domain: {domain_enum.value} | TTL: {ttl}s")

    def get_stats(self) -> dict:
        total = self._stats["total_lookups"]
        hits = self._stats["redis_hits"] + self._stats["qdrant_hits"]

        return {
            "total_lookups":   total,
            "redis_hits":      self._stats["redis_hits"],
            "qdrant_hits":     self._stats["qdrant_hits"],
            "misses":          self._stats["misses"],
            "hit_rate":        round(hits / max(total, 1) * 100, 1),
            "redis_hit_rate":  round(
                self._stats["redis_hits"] / max(total, 1) * 100, 1
            ),
            "qdrant_hit_rate": round(
                self._stats["qdrant_hits"] / max(total, 1) * 100, 1
            ),
        }