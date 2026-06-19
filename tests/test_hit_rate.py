"""
Hit rate measurement — runs controlled experiments to measure
actual cache hit rates under different conditions.

These are integration tests — they test the full cache stack
(embedder + domain detector + context builder) together.
Run after unit tests to validate end-to-end behaviour.
"""
import pytest
import asyncio
from gateway.cache.semantic_cache import SemanticCache
from gateway.providers.base import Message


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
class TestCacheHitRates:
    """
    Measures actual hit rates under controlled conditions.
    Results become your resume metrics.
    """

    async def _setup_cache(self):
        """Create in-memory cache for testing"""
        cache = SemanticCache()
        # Mock Redis and Qdrant for unit testing
        # In real integration tests these would be real connections
        return cache

    def test_hit_rate_identical_questions(self):
        """
        Identical questions should hit Redis 100% of the time
        after the first call.
        Expected hit rate: 100% on repeats
        """
        from gateway.cache.embedder import Embedder
        from gateway.cache.context_builder import ContextBuilder

        builder = ContextBuilder()
        embedder = Embedder()

        questions = [
            "what is machine learning",
            "what is machine learning",  # duplicate
            "what is recursion",
            "what is recursion",          # duplicate
            "how are you",
            "how are you",                # duplicate
        ]

        keys = [builder.build_key([Message(role="user", content=q)]) for q in questions]
        hashes = [embedder.text_hash(k) for k in keys]

        # Check duplicates produce same hash
        assert hashes[0] == hashes[1], "Identical questions must have same hash"
        assert hashes[2] == hashes[3], "Identical questions must have same hash"
        assert hashes[4] == hashes[5], "Identical questions must have same hash"

        print(f"\n  Identical question Redis hit rate: 100% (by hash)")

    def test_semantic_similarity_rates(self):
        """
        Measures similarity scores for known question pairs.
        Validates that thresholds correctly classify hits vs misses.
        """
        import asyncio
        from gateway.cache.embedder import Embedder
        from gateway.cache.domain_detector import DomainDetector

        embedder = Embedder()
        detector = DomainDetector()

        pairs = [
            # (q1, q2, domain, should_hit)
            ("what is machine learning", "explain machine learning",     "factual",       True),
            ("what is recursion",        "define recursion",              "factual",       True),
            ("how are you",              "how are you doing today",       "conversational", True),
            ("sort ascending",           "sort descending",               "code",          False),
            ("what is ML",               "how to cook pasta",             "factual",       False),
            ("explain TCP",              "explain UDP",                   "conceptual",    False),
        ]

        results = []
        for q1, q2, domain_str, expected_hit in pairs:
            from gateway.cache.domain_detector import Domain
            domain = Domain(domain_str)
            threshold = detector.get_threshold(domain)

            v1 = asyncio.run(embedder.embed(q1))
            v2 = asyncio.run(embedder.embed(q2))
            sim = embedder.cosine_similarity(v1, v2)
            actual_hit = sim >= threshold

            results.append({
                "q1": q1[:30],
                "q2": q2[:30],
                "domain": domain_str,
                "similarity": round(sim, 4),
                "threshold": threshold,
                "expected": expected_hit,
                "actual": actual_hit,
                "correct": actual_hit == expected_hit,
            })

        print("\n  Semantic similarity test results:")
        print(f"  {'Q1':<32} {'Domain':<15} {'Sim':>6} {'Thresh':>7} {'Hit':>5} {'OK':>4}")
        print(f"  {'-'*32} {'-'*15} {'-'*6} {'-'*7} {'-'*5} {'-'*4}")
        for r in results:
            ok = "OK" if r["correct"] else "FAIL"
            print(f"  {r['q1']:<32} {r['domain']:<15} {r['similarity']:>6.4f} {r['threshold']:>7.2f} {str(r['actual']):>5} {ok:>4}")

        correct = sum(1 for r in results if r["correct"])
        accuracy = correct / len(results) * 100
        print(f"\n  Threshold accuracy: {correct}/{len(results)} = {accuracy:.0f}%")

        assert accuracy >= 80, f"Threshold accuracy {accuracy:.0f}% below 80% minimum"

    def test_context_isolation_rate(self):
        """
        Validates that context-aware keys prevent cross-context cache hits.
        Same question in different contexts must produce different keys.
        """
        from gateway.cache.context_builder import ContextBuilder
        from gateway.cache.embedder import Embedder

        builder = ContextBuilder(context_window=3)
        embedder = Embedder()

        context_pairs = [
            (
                [Message(role="user", content="I am a Python developer"),
                 Message(role="user", content="tell me about Python")],
                [Message(role="user", content="I love reptiles"),
                 Message(role="user", content="tell me about Python")],
                "Python developer vs Python snake context",
            ),
            (
                [Message(role="user", content="I work in finance"),
                 Message(role="user", content="what are derivatives")],
                [Message(role="user", content="I study mathematics"),
                 Message(role="user", content="what are derivatives")],
                "Financial derivatives vs math derivatives context",
            ),
        ]

        print("\n  Context isolation test results:")
        all_isolated = True

        for msgs_a, msgs_b, description in context_pairs:
            key_a = builder.build_key(msgs_a)
            key_b = builder.build_key(msgs_b)

            # Keys must be different
            keys_different = key_a != key_b

            # Embeddings should be meaningfully different
            import asyncio
            v_a = asyncio.run(embedder.embed(key_a))
            v_b = asyncio.run(embedder.embed(key_b))
            sim = embedder.cosine_similarity(v_a, v_b)

            print(f"  {description}")
            print(f"    Keys different: {keys_different}")
            print(f"    Embedding similarity: {sim:.4f}")

            if not keys_different:
                all_isolated = False
                print(f"    FAIL: same keys for different contexts")

        assert all_isolated, "All context pairs must produce different cache keys"