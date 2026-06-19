"""
Cost router test suite — validates complexity scoring
and model selection per tier.
"""
import pytest
from gateway.routing.cost_router import CostRouter
from gateway.providers.base import Message


@pytest.fixture(scope="module")
def router():
    return CostRouter()


def msg(content):
    return [Message(role="user", content=content)]


class TestCostRouter:

    def test_greeting_routes_cheap(self, router):
        decision = router.decide(msg("how are you"), "gemini")
        assert decision.tier == "cheap"
        assert decision.classified_by == "obvious"

    def test_single_word_routes_cheap(self, router):
        decision = router.decide(msg("hello"), "gemini")
        assert decision.tier == "cheap"

    def test_simple_factual_routes_cheap(self, router):
        decision = router.decide(msg("what is the capital of France"), "gemini")
        assert decision.tier == "cheap"
        assert decision.complexity_score <= 3

    def test_code_block_routes_powerful(self, router):
        decision = router.decide(
            msg("```python\ndef sort(arr): pass\n``` fix this"), "gemini"
        )
        assert decision.tier == "powerful"
        assert decision.classified_by == "obvious"

    def test_architecture_routes_powerful(self, router):
        decision = router.decide(
            msg("design a distributed caching architecture"), "gemini"
        )
        assert decision.tier == "powerful"

    def test_explain_routes_mid_or_above(self, router):
        decision = router.decide(
            msg("explain how neural networks learn from data"), "gemini"
        )
        assert decision.tier in ["mid", "powerful"]
        assert decision.complexity_score >= 4

    def test_complex_analysis_routes_powerful(self, router):
        decision = router.decide(
            msg("compare and analyse microservices vs monolith architecture in detail with tradeoffs"),
            "gemini"
        )
        assert decision.tier == "powerful"
        assert decision.complexity_score >= 7

    def test_deep_conversation_routes_mid_or_above(self, router):
        messages = [
            Message(role="user",      content="what is recursion"),
            Message(role="assistant", content="recursion is..."),
            Message(role="user",      content="can you give an example"),
            Message(role="assistant", content="sure..."),
            Message(role="user",      content="now explain dynamic programming"),
            Message(role="assistant", content="dp is..."),
            Message(role="user",      content="how does memoization relate"),
        ]
        decision = router.decide(messages, "gemini")
        assert decision.tier in ["mid", "powerful"]

    def test_score_range(self, router):
        """Complexity score always between 1 and 10"""
        test_cases = [
            "hi",
            "what is 2+2",
            "explain quantum entanglement in detail with mathematical derivations",
            "implement a distributed consensus algorithm from scratch",
        ]
        for text in test_cases:
            score, _ = router._keyword_score(msg(text))
            assert 1 <= score <= 10, f"Score {score} out of range for: {text}"

    def test_cost_estimation_cheap_less_than_powerful(self, router):
        """Cheap models should cost less than powerful models"""
        cheap   = router.estimate_cost("llama3-8b-8192",    100, 200)
        powerful = router.estimate_cost("gpt-4o",           100, 200)
        assert cheap < powerful, f"Cheap {cheap} should < powerful {powerful}"

    def test_savings_positive(self, router):
        """Routing to cheap model should always save money vs GPT-4o"""
        savings = router.savings_vs_powerful("llama3-8b-8192", 100, 200)
        assert savings > 0, "Routing to cheap model should save money"

    def test_savings_zero_for_powerful(self, router):
        """No savings when using the powerful model itself"""
        savings = router.savings_vs_powerful("gpt-4o", 100, 200)
        assert savings == 0.0


class TestObviousFilter:

    def test_obvious_cheap_greetings(self, router):
        greetings = ["hello", "hi", "thanks", "goodbye", "how are you"]
        for g in greetings:
            assert router._is_obvious_cheap(g), f"Should be obvious cheap: {g}"

    def test_obvious_cheap_single_word(self, router):
        assert router._is_obvious_cheap("yes")
        assert router._is_obvious_cheap("ok")
        assert router._is_obvious_cheap("no")

    def test_not_obvious_cheap_complex(self, router):
        complex_cases = [
            "explain microservices architecture",
            "implement binary search",
            "compare machine learning approaches",
        ]
        for text in complex_cases:
            assert not router._is_obvious_cheap(text), \
                f"Should NOT be obvious cheap: {text}"

    def test_obvious_powerful_code_block(self, router):
        assert router._is_obvious_powerful("```python\ncode here\n```")
        assert router._is_obvious_powerful("def my_function():")
        assert router._is_obvious_powerful("import numpy as np")

    def test_obvious_powerful_architecture(self, router):
        assert router._is_obvious_powerful("design a microservices architecture")
        assert router._is_obvious_powerful("build a distributed system")