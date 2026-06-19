"""
Cache test suite — measures actual hit rates and validates
context-awareness and domain-adaptive thresholds.
"""
import pytest
import asyncio
from gateway.cache.domain_detector import DomainDetector, Domain
from gateway.cache.context_builder import ContextBuilder
from gateway.cache.embedder import Embedder
from gateway.providers.base import Message


@pytest.fixture(scope="module")
def detector():
    return DomainDetector()


@pytest.fixture(scope="module")
def builder():
    return ContextBuilder(context_window=3)


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


# ─── domain detection tests ──────────────────────────────────────────────────

class TestDomainDetector:

    def test_code_domain(self, detector):
        cases = [
            "implement binary search in Python",
            "debug this function it has an error",
            "sort array in ascending order",
            "write a class for linked list",
        ]
        for text in cases:
            assert detector.detect(text) == Domain.CODE, f"Expected CODE for: {text}"

    def test_factual_domain(self, detector):
        cases = [
            "what is machine learning",
            "who is the president of India",
            "when did world war 2 end",
            "what is the capital of France",
        ]
        for text in cases:
            assert detector.detect(text) == Domain.FACTUAL, f"Expected FACTUAL for: {text}"

    def test_conversational_domain(self, detector):
        cases = [
            "how are you",
            "thanks for your help",
            "good morning",
        ]
        for text in cases:
            assert detector.detect(text) == Domain.CONVERSATIONAL, f"Expected CONVERSATIONAL for: {text}"

    def test_conceptual_domain(self, detector):
        cases = [
            "compare microservices vs monolith architecture",
            "explain the difference between TCP and UDP",
            "analyse the tradeoffs of distributed systems",
        ]
        for text in cases:
            assert detector.detect(text) == Domain.CONCEPTUAL, f"Expected CONCEPTUAL for: {text}"

    def test_thresholds(self, detector):
        assert detector.get_threshold(Domain.CODE) == 0.96
        assert detector.get_threshold(Domain.FACTUAL) == 0.75
        assert detector.get_threshold(Domain.CONCEPTUAL) == 0.78
        assert detector.get_threshold(Domain.CONVERSATIONAL) == 0.70

    def test_code_threshold_strictest(self, detector):
        """Code must have strictest threshold to prevent wrong cache hits"""
        code_t = detector.get_threshold(Domain.CODE)
        for domain in [Domain.FACTUAL, Domain.CONCEPTUAL, Domain.CONVERSATIONAL]:
            assert code_t > detector.get_threshold(domain), \
                f"Code threshold {code_t} should be > {domain.value} threshold"


# ─── context builder tests ────────────────────────────────────────────────────

class TestContextBuilder:

    def test_single_message_key(self, builder):
        messages = [Message(role="user", content="what is Python")]
        key = builder.build_key(messages)
        assert "user" in key
        assert "what is Python" in key

    def test_multi_message_key(self, builder):
        messages = [
            Message(role="user",      content="I love animals"),
            Message(role="assistant", content="That is great"),
            Message(role="user",      content="tell me about Python"),
        ]
        key = builder.build_key(messages)
        assert "I love animals" in key
        assert "tell me about Python" in key
        assert "|" in key  # separator present

    def test_context_window_limits(self, builder):
        """Only last 3 messages should be in the key"""
        messages = [
            Message(role="user", content="message one"),
            Message(role="user", content="message two"),
            Message(role="user", content="message three"),
            Message(role="user", content="message four"),
            Message(role="user", content="message five"),
        ]
        key = builder.build_key(messages)
        assert "message one" not in key    # outside window
        assert "message two" not in key    # outside window
        assert "message three" in key      # inside window
        assert "message five" in key       # inside window

    def test_different_context_different_key(self, builder):
        """Same last message + different history = different key"""
        messages_a = [
            Message(role="user", content="I am a developer"),
            Message(role="user", content="tell me about Python"),
        ]
        messages_b = [
            Message(role="user", content="I love reptiles"),
            Message(role="user", content="tell me about Python"),
        ]
        key_a = builder.build_key(messages_a)
        key_b = builder.build_key(messages_b)
        assert key_a != key_b, "Different contexts must produce different keys"

    def test_same_context_same_key(self, builder):
        """Identical conversation = identical key"""
        messages = [Message(role="user", content="what is recursion")]
        assert builder.build_key(messages) == builder.build_key(messages)

    def test_get_last_user_message(self, builder):
        messages = [
            Message(role="user",      content="first question"),
            Message(role="assistant", content="first answer"),
            Message(role="user",      content="second question"),
        ]
        assert builder.get_last_user_message(messages) == "second question"


# ─── embedder tests ───────────────────────────────────────────────────────────

class TestEmbedder:

    def test_embedding_dimensions(self, embedder):
        vector = asyncio.run(embedder.embed("test message"))
        assert len(vector) == 384, f"Expected 384 dims, got {len(vector)}"

    def test_similar_texts_high_similarity(self, embedder):
        v1 = asyncio.run(embedder.embed("what is machine learning"))
        v2 = asyncio.run(embedder.embed("explain machine learning"))
        sim = embedder.cosine_similarity(v1, v2)
        assert sim > 0.70, f"Similar texts should score > 0.70, got {sim:.4f}"

    def test_dissimilar_texts_low_similarity(self, embedder):
        v1 = asyncio.run(embedder.embed("what is machine learning"))
        v2 = asyncio.run(embedder.embed("how to cook biryani"))
        sim = embedder.cosine_similarity(v1, v2)
        assert sim < 0.40, f"Dissimilar texts should score < 0.40, got {sim:.4f}"

    def test_identical_texts_max_similarity(self, embedder):
        v1 = asyncio.run(embedder.embed("what is recursion"))
        v2 = asyncio.run(embedder.embed("what is recursion"))
        sim = embedder.cosine_similarity(v1, v2)
        assert sim > 0.999, f"Identical texts should score ~1.0, got {sim:.4f}"

    def test_code_domain_threshold_prevents_wrong_hit(self, embedder, detector):
        """
        Core test — validates Improvement 2.
        Ascending vs descending sort should NOT be a cache hit
        because they have opposite answers.
        """
        v1 = asyncio.run(embedder.embed("sort array in ascending order"))
        v2 = asyncio.run(embedder.embed("sort array in descending order"))
        sim = embedder.cosine_similarity(v1, v2)
        threshold = detector.get_threshold(Domain.CODE)
        
        print(f"\n  ascending vs descending similarity: {sim:.4f}")
        print(f"  code threshold: {threshold}")
        print(f"  would cache hit: {sim >= threshold}")
        
        assert sim < threshold, (
            f"CRITICAL: ascending vs descending scored {sim:.4f} >= "
            f"code threshold {threshold}. Would serve wrong cached answer!"
        )

    def test_factual_threshold_allows_rephrasing(self, embedder, detector):
        """
        Factual questions with different wording should hit cache.
        'what is ML' and 'explain ML' mean the same thing.
        """
        v1 = asyncio.run(embedder.embed("what is machine learning"))
        v2 = asyncio.run(embedder.embed("explain machine learning to me"))
        sim = embedder.cosine_similarity(v1, v2)
        threshold = detector.get_threshold(Domain.FACTUAL)
        
        print(f"\n  ML vs explain ML similarity: {sim:.4f}")
        print(f"  factual threshold: {threshold}")
        print(f"  would cache hit: {sim >= threshold}")
        
        assert sim >= threshold, (
            f"Factual rephrasing scored {sim:.4f} < "
            f"factual threshold {threshold}. Should be a cache hit!"
        )

    def test_context_changes_embedding(self, embedder, builder):
        """
        Improvement 1 — context-aware keys produce different embeddings.
        Same question, different conversation context = different vector.
        """
        messages_dev = [
            Message(role="user", content="I am a Python developer"),
            Message(role="user", content="tell me about Python"),
        ]
        messages_nature = [
            Message(role="user", content="I love reptiles and animals"),
            Message(role="user", content="tell me about Python"),
        ]

        key_dev    = builder.build_key(messages_dev)
        key_nature = builder.build_key(messages_nature)

        v_dev    = asyncio.run(embedder.embed(key_dev))
        v_nature = asyncio.run(embedder.embed(key_nature))

        sim = embedder.cosine_similarity(v_dev, v_nature)
        print(f"\n  developer ctx vs animal ctx similarity: {sim:.4f}")

        assert sim < 0.99, (
            f"Different contexts should produce different embeddings, "
            f"got similarity {sim:.4f}"
        )
        assert sim > 0.40, (
            f"Same question should keep some similarity even with "
            f"different context, got {sim:.4f}"
        )

    def test_hash_deterministic(self, embedder):
        """Same text always produces same hash"""
        text = "what is machine learning"
        assert embedder.text_hash(text) == embedder.text_hash(text)

    def test_hash_different_texts(self, embedder):
        """Different texts produce different hashes"""
        h1 = embedder.text_hash("what is machine learning")
        h2 = embedder.text_hash("what is deep learning")
        assert h1 != h2