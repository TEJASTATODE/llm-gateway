import asyncio
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

from gateway.config import settings
from gateway.providers.registry import ProviderRegistry
from gateway.providers.base import Message
from gateway.routing.circuit_breaker_registry import CircuitBreakerRegistry
from gateway.routing.cost_router import CostRouter, MODEL_COSTS, PROVIDER_MODELS
from gateway.cache.semantic_cache import SemanticCache
from gateway.logger import RequestLogger, RequestLog
from gateway.middleware.latency import LatencyTracker


provider_registry = ProviderRegistry()
cb_registry = CircuitBreakerRegistry()
cache = SemanticCache()
cost_router = CostRouter()
logger = RequestLogger()


@asynccontextmanager
async def lifespan(app):
    await cache.init()
    await logger.init()
    print("[Gateway] All systems ready")
    yield


app = FastAPI(title="LLM Gateway", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[dict]
    preferred_provider: str = "gemini"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "available_providers": provider_registry.available_providers(),
        "circuit_breakers": cb_registry.all_stats(),
        "cache_stats": cache.get_stats(),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    tracker = LatencyTracker()

    msg_objects = [
        Message(role=m["role"], content=m["content"])
        for m in request.messages
    ]

    # Step 1 — cache lookup with latency tracking
    tracker.mark_cache_start()
    cache_result = await cache.lookup(messages=msg_objects)
    tracker.mark_cache_end(
        hit=cache_result is not None,
        layer=cache_result.cache_layer if cache_result else "",
    )

    if cache_result:
        latency = tracker.to_dict()

        asyncio.create_task(logger.log(RequestLog(
            user_id="default",
            provider="cache",
            model=cache_result.model,
            domain=cache_result.domain,
            tier="",
            complexity_score=0,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            savings_vs_gpt4o=0.0,
            cache_hit=True,
            cache_layer=cache_result.cache_layer,
            similarity_score=cache_result.similarity,
            total_latency_ms=latency["total_ms"],
            cache_latency_ms=latency["cache_lookup_ms"],
            provider_latency_ms=0,
            classified_by="cache",
            routing_reasoning="cache hit",
        )))

        return {
            "provider_used": "cache",
            "model": cache_result.model,
            "content": cache_result.response.get("content", ""),
            "domain": cache_result.domain,
            "similarity": cache_result.similarity,
            "cache_hit": True,
            "cache_layer": cache_result.cache_layer,
            "routing": None,
            "cost": {"estimated_usd": 0.0, "savings_vs_gpt4o_usd": 0.0},
            "tokens": {"prompt": 0, "completion": 0},
            "latency": latency,
        }

    # Step 2 — cost routing
    routing = cost_router.decide(
        messages=msg_objects,
        preferred_provider=request.preferred_provider,
    )

    print(f"[Router] Score: {routing.complexity_score} | "
          f"Tier: {routing.tier} | By: {routing.classified_by}")

    # Step 3 — provider call with circuit breaker
    provider_order = cb_registry.get_healthy_providers(
        preferred=request.preferred_provider
    )
    last_error = None

    for provider_name in provider_order:
        cb = cb_registry.get(provider_name)
        if not cb.is_closed():
            continue

        provider = provider_registry.get(provider_name)
        if not provider.is_available():
            continue

        try:
            tracker.mark_provider_start()
            response = await provider.complete(
                model=routing.tier,
                messages=msg_objects,
            )
            tracker.mark_provider_end()
            cb.record_success()

            estimated_cost = cost_router.estimate_cost(
                response.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            savings = cost_router.savings_vs_powerful(
                response.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

            latency = tracker.to_dict()

            response_dict = {
                "content": response.content,
                "model": response.model,
                "provider": provider_name,
            }

            # Store in cache async
            asyncio.create_task(
                cache.store(
                    messages=msg_objects,
                    response=response_dict,
                    domain=routing.tier,
                )
            )

            # Log to Postgres async
            asyncio.create_task(logger.log(RequestLog(
                user_id="default",
                provider=provider_name,
                model=response.model,
                domain=routing.tier,
                tier=routing.tier,
                complexity_score=routing.complexity_score,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cost_usd=estimated_cost,
                savings_vs_gpt4o=savings,
                cache_hit=False,
                cache_layer="",
                similarity_score=0.0,
                total_latency_ms=latency["total_ms"],
                cache_latency_ms=latency["cache_lookup_ms"],
                provider_latency_ms=latency["provider_ms"],
                classified_by=routing.classified_by,
                routing_reasoning=routing.reasoning,
            )))

            return {
                "provider_used": provider_name,
                "model": response.model,
                "content": response.content,
                "cache_hit": False,
                "cache_layer": "",
                "routing": {
                    "complexity_score": routing.complexity_score,
                    "tier": routing.tier,
                    "classified_by": routing.classified_by,
                    "reasoning": routing.reasoning,
                },
                "cost": {
                    "estimated_usd": estimated_cost,
                    "savings_vs_gpt4o_usd": savings,
                },
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                },
                "latency": latency,
            }

        except Exception as e:
            cb.record_failure()
            last_error = str(e)
            print(f"[Gateway] Provider {provider_name} failed: {e}")
            continue

    raise HTTPException(
        status_code=503,
        detail=f"All providers unavailable. Last error: {last_error}"
    )


@app.get("/analytics")
async def analytics():
    """Full analytics from Postgres — powers historical dashboard"""
    return await logger.get_analytics()


@app.get("/cache/stats")
async def cache_stats():
    return cache.get_stats()


@app.get("/providers/health")
async def provider_health():
    return cb_registry.all_stats()


@app.get("/router/stats")
async def router_stats():
    return {
        "llm_classifier_available": cost_router._llm_available,
        "model_costs": MODEL_COSTS,
        "provider_tiers": PROVIDER_MODELS,
    }


@app.post("/test/simulate-failure/{provider_name}")
async def simulate_failure(provider_name: str, failures: int = 5):
    cb = cb_registry.get(provider_name)
    for _ in range(failures):
        cb.record_failure()
    return {
        "message": f"Simulated {failures} failures for {provider_name}",
        "circuit_state": cb.state,
        "failure_count": cb.failure_count,
    }


@app.post("/test/break-provider/{provider_name}")
async def break_provider(provider_name: str):
    provider = provider_registry.get(provider_name)
    cb = cb_registry.get(provider_name)
    failures = 0
    for _ in range(5):
        try:
            await provider.complete(
                model="auto",
                messages=[Message(role="user", content="")]
            )
        except Exception:
            cb.record_failure()
            failures += 1
    return {
        "provider": provider_name,
        "real_failures_triggered": failures,
        "circuit_state": cb.state,
        "failure_count": cb.failure_count,
    }


@app.post("/test/reset-provider/{provider_name}")
async def reset_provider(provider_name: str):
    from gateway.routing.circuit_breaker import CircuitState
    cb = cb_registry.get(provider_name)
    cb._state = CircuitState.CLOSED
    cb._failure_count = 0
    return {
        "provider": provider_name,
        "circuit_state": cb.state,
        "message": "Reset to closed",
    }


@app.post("/test/{provider_name}")
async def test_provider(
    provider_name: str,
    message: str = "What is 2+2? One sentence only."
):
    if provider_name not in ["gemini", "openai", "anthropic"]:
        return {"error": "Unknown provider"}
    provider = provider_registry.get(provider_name)
    if not provider.is_available():
        return {"error": f"{provider_name} key not configured"}
    messages = [Message(role="user", content=message)]
    response = await provider.complete(model="auto", messages=messages)
    return {
        "provider": response.provider,
        "model": response.model,
        "content": response.content,
    }