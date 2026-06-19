import asyncio
import time
from dataclasses import dataclass
import asyncpg
from gateway.config import settings


@dataclass
class RequestLog:
    user_id: str
    provider: str
    model: str
    domain: str
    tier: str
    complexity_score: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    savings_vs_gpt4o: float
    cache_hit: bool
    cache_layer: str        # redis | qdrant | ""
    similarity_score: float
    total_latency_ms: int
    cache_latency_ms: int
    provider_latency_ms: int
    classified_by: str
    routing_reasoning: str


class RequestLogger:
    """
    Async Postgres logger — writes after response is sent.
    Uses fire-and-forget pattern so logging never adds latency.

    WHY POSTGRES NOT REDIS:
    Redis is fast but volatile — data lost on restart.
    Postgres is permanent — historical analytics, cost reports,
    trend analysis all need persistent storage.
    We write async so it never blocks the response.
    """

    def __init__(self):
        self._pool = None

    async def init(self):
        """Create connection pool and ensure table exists"""
        try:
            self._pool = await asyncpg.create_pool(
                settings.postgres_url.replace("+asyncpg", ""),
                min_size=2,
                max_size=10,
            )
            await self._create_table()
            print("[Logger] Postgres connected")
        except Exception as e:
            print(f"[Logger] Postgres unavailable: {e}")
            self._pool = None

    async def _create_table(self):
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id                  SERIAL PRIMARY KEY,
                    created_at          TIMESTAMPTZ DEFAULT NOW(),
                    user_id             TEXT DEFAULT 'default',
                    provider            TEXT NOT NULL,
                    model               TEXT NOT NULL,
                    domain              TEXT DEFAULT '',
                    tier                TEXT DEFAULT '',
                    complexity_score    INT DEFAULT 0,
                    prompt_tokens       INT DEFAULT 0,
                    completion_tokens   INT DEFAULT 0,
                    total_tokens        INT DEFAULT 0,
                    cost_usd            NUMERIC(12,8) DEFAULT 0,
                    savings_vs_gpt4o    NUMERIC(12,8) DEFAULT 0,
                    cache_hit           BOOLEAN DEFAULT FALSE,
                    cache_layer         TEXT DEFAULT '',
                    similarity_score    NUMERIC(6,4) DEFAULT 0,
                    total_latency_ms    INT DEFAULT 0,
                    cache_latency_ms    INT DEFAULT 0,
                    provider_latency_ms INT DEFAULT 0,
                    classified_by       TEXT DEFAULT '',
                    routing_reasoning   TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_logs_created
                    ON request_logs (created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_logs_provider
                    ON request_logs (provider);
                CREATE INDEX IF NOT EXISTS idx_logs_cache
                    ON request_logs (cache_hit);
                CREATE INDEX IF NOT EXISTS idx_logs_tier
                    ON request_logs (tier);
            """)

    async def log(self, entry: RequestLog):
        """
        Fire and forget — called with asyncio.create_task()
        Never awaited directly so it never blocks response.
        """
        if not self._pool:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO request_logs (
                        provider, model, domain, tier,
                        complexity_score, prompt_tokens,
                        completion_tokens, total_tokens,
                        cost_usd, savings_vs_gpt4o,
                        cache_hit, cache_layer, similarity_score,
                        total_latency_ms, cache_latency_ms,
                        provider_latency_ms, classified_by,
                        routing_reasoning
                    ) VALUES (
                        $1,$2,$3,$4,$5,$6,$7,$8,
                        $9,$10,$11,$12,$13,$14,$15,
                        $16,$17,$18
                    )
                """,
                    entry.provider,
                    entry.model,
                    entry.domain,
                    entry.tier,
                    entry.complexity_score,
                    entry.prompt_tokens,
                    entry.completion_tokens,
                    entry.prompt_tokens + entry.completion_tokens,
                    entry.cost_usd,
                    entry.savings_vs_gpt4o,
                    entry.cache_hit,
                    entry.cache_layer,
                    entry.similarity_score,
                    entry.total_latency_ms,
                    entry.cache_latency_ms,
                    entry.provider_latency_ms,
                    entry.classified_by,
                    entry.routing_reasoning,
                )
        except Exception as e:
            print(f"[Logger] Write failed: {e}")

    async def get_analytics(self) -> dict:
        """
        Aggregated analytics from Postgres.
        This powers the historical dashboard section.
        """
        if not self._pool:
            return {}

        async with self._pool.acquire() as conn:
            # Overall stats
            overall = await conn.fetchrow("""
                SELECT
                    COUNT(*)                            AS total_requests,
                    COUNT(*) FILTER (WHERE cache_hit)   AS cache_hits,
                    COUNT(*) FILTER (WHERE NOT cache_hit
                        AND cache_layer = '')           AS cache_misses,
                    SUM(cost_usd)                       AS total_cost,
                    SUM(savings_vs_gpt4o)               AS total_savings,
                    SUM(prompt_tokens + completion_tokens) AS total_tokens,
                    AVG(total_latency_ms)
                        FILTER (WHERE NOT cache_hit)    AS avg_provider_latency,
                    AVG(total_latency_ms)
                        FILTER (WHERE cache_hit)        AS avg_cache_latency,
                    AVG(complexity_score)
                        FILTER (WHERE complexity_score > 0) AS avg_complexity
                FROM request_logs
            """)

            # Per provider breakdown
            by_provider = await conn.fetch("""
                SELECT
                    provider,
                    COUNT(*)                AS requests,
                    SUM(cost_usd)           AS cost,
                    AVG(total_latency_ms)   AS avg_latency,
                    COUNT(*) FILTER (WHERE cache_hit) AS cache_hits
                FROM request_logs
                GROUP BY provider
                ORDER BY requests DESC
            """)

            # Per tier breakdown
            by_tier = await conn.fetch("""
                SELECT
                    tier,
                    COUNT(*)        AS requests,
                    SUM(cost_usd)   AS cost,
                    SUM(prompt_tokens + completion_tokens) AS tokens
                FROM request_logs
                WHERE tier != ''
                GROUP BY tier
                ORDER BY requests DESC
            """)

            # Per model breakdown
            by_model = await conn.fetch("""
                SELECT
                    model,
                    COUNT(*)        AS requests,
                    SUM(cost_usd)   AS cost,
                    AVG(total_latency_ms) AS avg_latency
                FROM request_logs
                WHERE NOT cache_hit
                GROUP BY model
                ORDER BY requests DESC
            """)

            # Hourly cost trend (last 24 hours)
            hourly = await conn.fetch("""
                SELECT
                    DATE_TRUNC('hour', created_at) AS hour,
                    COUNT(*)        AS requests,
                    SUM(cost_usd)   AS cost,
                    COUNT(*) FILTER (WHERE cache_hit) AS hits
                FROM request_logs
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY hour
                ORDER BY hour ASC
            """)

            # Cache layer breakdown
            cache_layers = await conn.fetch("""
                SELECT
                    cache_layer,
                    COUNT(*) AS count,
                    AVG(total_latency_ms) AS avg_latency
                FROM request_logs
                WHERE cache_hit = TRUE
                GROUP BY cache_layer
            """)

            # Latency percentiles
            latency = await conn.fetchrow("""
                SELECT
                    PERCENTILE_CONT(0.50) WITHIN GROUP
                        (ORDER BY total_latency_ms)
                        FILTER (WHERE NOT cache_hit) AS p50,
                    PERCENTILE_CONT(0.95) WITHIN GROUP
                        (ORDER BY total_latency_ms)
                        FILTER (WHERE NOT cache_hit) AS p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP
                        (ORDER BY total_latency_ms)
                        FILTER (WHERE NOT cache_hit) AS p99,
                    PERCENTILE_CONT(0.50) WITHIN GROUP
                        (ORDER BY total_latency_ms)
                        FILTER (WHERE cache_hit)     AS cache_p50
                FROM request_logs
            """)

            # Recent 20 requests
            recent = await conn.fetch("""
                SELECT
                    id, created_at, provider, model,
                    tier, complexity_score, cache_hit,
                    cache_layer, cost_usd, total_latency_ms,
                    prompt_tokens, completion_tokens
                FROM request_logs
                ORDER BY created_at DESC
                LIMIT 20
            """)

        total = overall["total_requests"] or 0
        hits = overall["cache_hits"] or 0

        return {
            "overall": {
                "total_requests":      total,
                "cache_hits":          hits,
                "cache_misses":        overall["cache_misses"] or 0,
                "hit_rate_pct":        round((hits / total * 100) if total else 0, 1),
                "total_cost_usd":      float(overall["total_cost"] or 0),
                "total_savings_usd":   float(overall["total_savings"] or 0),
                "total_tokens":        int(overall["total_tokens"] or 0),
                "avg_provider_latency_ms": round(float(overall["avg_provider_latency"] or 0)),
                "avg_cache_latency_ms":    round(float(overall["avg_cache_latency"] or 0)),
                "avg_complexity":          round(float(overall["avg_complexity"] or 0), 1),
            },
            "by_provider": [
                {
                    "provider":    r["provider"],
                    "requests":    r["requests"],
                    "cost":        float(r["cost"] or 0),
                    "avg_latency": round(float(r["avg_latency"] or 0)),
                    "cache_hits":  r["cache_hits"],
                }
                for r in by_provider
            ],
            "by_tier": [
                {
                    "tier":     r["tier"],
                    "requests": r["requests"],
                    "cost":     float(r["cost"] or 0),
                    "tokens":   int(r["tokens"] or 0),
                }
                for r in by_tier
            ],
            "by_model": [
                {
                    "model":       r["model"],
                    "requests":    r["requests"],
                    "cost":        float(r["cost"] or 0),
                    "avg_latency": round(float(r["avg_latency"] or 0)),
                }
                for r in by_model
            ],
            "latency_percentiles": {
                "provider_p50": round(float(latency["p50"] or 0)),
                "provider_p95": round(float(latency["p95"] or 0)),
                "provider_p99": round(float(latency["p99"] or 0)),
                "cache_p50":    round(float(latency["cache_p50"] or 0)),
            },
            "hourly_trend": [
                {
                    "hour":     r["hour"].strftime("%H:%M"),
                    "requests": r["requests"],
                    "cost":     float(r["cost"] or 0),
                    "hits":     r["hits"],
                }
                for r in hourly
            ],
            "cache_layers": [
                {
                    "layer":       r["cache_layer"] or "miss",
                    "count":       r["count"],
                    "avg_latency": round(float(r["avg_latency"] or 0)),
                }
                for r in cache_layers
            ],
            "recent_requests": [
                {
                    "id":               r["id"],
                    "created_at":       r["created_at"].isoformat(),
                    "provider":         r["provider"],
                    "model":            r["model"],
                    "tier":             r["tier"],
                    "complexity_score": r["complexity_score"],
                    "cache_hit":        r["cache_hit"],
                    "cache_layer":      r["cache_layer"],
                    "cost_usd":         float(r["cost_usd"] or 0),
                    "total_latency_ms": r["total_latency_ms"],
                    "prompt_tokens":    r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
                }
                for r in recent
            ],
        }