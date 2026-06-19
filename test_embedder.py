import asyncio

from gateway.cache.embedder import Embedder
from gateway.cache.domain_detector import DomainDetector, Domain


async def test():
    embedder = Embedder()
    detector = DomainDetector()

    print("=" * 60)
    print("EMBEDDING TEST")
    print("=" * 60)

    q1 = "what is machine learning"
    q2 = "explain machine learning to me"
    q3 = "how to cook biryani"

    v1 = await embedder.embed(q1)
    v2 = await embedder.embed(q2)
    v3 = await embedder.embed(q3)

    print(f"Vector dimensions: {len(v1)}")
    print()

    sim_12 = embedder.cosine_similarity(v1, v2)
    sim_13 = embedder.cosine_similarity(v1, v3)

    print(f"ML vs Explain ML : {sim_12:.4f}")
    print(f"ML vs Biryani    : {sim_13:.4f}")

    print()
    print("=" * 60)
    print("CACHE THRESHOLD TEST")
    print("=" * 60)

    print(f"0.92 (conversation) : {sim_12 >= 0.92}")
    print(f"0.97 (conceptual)   : {sim_12 >= 0.97}")
    print(f"0.99 (code)         : {sim_12 >= 0.99}")

    print()
    print("=" * 60)
    print("CONTEXT AWARENESS TEST")
    print("=" * 60)

    key_developer = (
        "user: I am a Python developer | "
        "user: tell me about Python"
    )

    key_animals = (
        "user: I love animals | "
        "user: tell me about Python"
    )

    key_plain = "user: tell me about Python"

    vd = await embedder.embed(key_developer)
    va = await embedder.embed(key_animals)
    vp = await embedder.embed(key_plain)

    print(
        "Developer vs Animal Context : "
        f"{embedder.cosine_similarity(vd, va):.4f}"
    )

    print(
        "Developer vs No Context     : "
        f"{embedder.cosine_similarity(vd, vp):.4f}"
    )

    print()
    print(f"Hash: {embedder.text_hash(q1)}")

    print()
    print("=" * 60)
    print("DOMAIN THRESHOLD VALIDATION")
    print("=" * 60)

    test_cases = [
        # Positive matches
        (
            "what is machine learning",
            "explain machine learning",
            Domain.FACTUAL,
        ),
        (
            "compare TCP and UDP",
            "difference between TCP and UDP",
            Domain.CONCEPTUAL,
        ),
        (
            "how are you",
            "how are you doing",
            Domain.CONVERSATIONAL,
        ),
        (
            "sort array ascending",
            "sort array descending",
            Domain.CODE,
        ),

        # Negative matches
        (
            "what is machine learning",
            "how to cook biryani",
            Domain.FACTUAL,
        ),
        (
            "sort array ascending",
            "weather in mumbai",
            Domain.CODE,
        ),
        (
            "compare TCP and UDP",
            "best pizza near me",
            Domain.CONCEPTUAL,
        ),
        (
            "how are you",
            "quantum mechanics explained",
            Domain.CONVERSATIONAL,
        ),
    ]

    for idx, (a, b, domain) in enumerate(test_cases, start=1):
        vec_a = await embedder.embed(a)
        vec_b = await embedder.embed(b)

        similarity = embedder.cosine_similarity(vec_a, vec_b)
        threshold = detector.get_threshold(domain)

        cache_hit = similarity >= threshold

        print(f"\n[{idx}] {domain.value}")
        print(f"Q1: {a}")
        print(f"Q2: {b}")
        print(f"Similarity : {similarity:.4f}")
        print(f"Threshold  : {threshold:.4f}")
        print(f"Cache Hit  : {cache_hit}")

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test())