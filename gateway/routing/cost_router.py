from dataclasses import dataclass
from gateway.providers.base import Message
from gateway.cache.domain_detector import DomainDetector, Domain


PROVIDER_MODELS = {
    "gemini": {
        "cheap":    "gemini-2.5-flash-lite",
        "mid":      "gemini-2.5-flash-lite",
        "powerful": "gemini-2.5-flash-lite",
    },
    "openai": {
        "cheap":    "llama-3.1-8b-instant",     # Groq — 8B fast
        "mid":      "llama-3.3-70b-versatile",  # Groq — 70B balanced
        "powerful": "llama-3.3-70b-versatile",  # Groq — 70B best
    },
}
MODEL_COSTS = {
    # Gemini
    "gemini-2.5-flash-lite": {"prompt": 0.0001,  "completion": 0.0004},
    "gemini-2.5-flash":      {"prompt": 0.0003,  "completion": 0.0025},

    # Groq (via OpenAI adapter)
    "llama-3.1-8b-instant":    {"prompt": 0.00005,  "completion": 0.00008},
    "llama-3.3-70b-versatile": {"prompt": 0.00059,  "completion": 0.00079},

    # OpenAI (if added later)
    "gpt-4o-mini":  {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o":       {"prompt": 0.0025,  "completion": 0.01},

    # Anthropic (if added later)
    "claude-sonnet-4-6":  {"prompt": 0.003,   "completion": 0.015},
    "claude-3-haiku":     {"prompt": 0.00025, "completion": 0.00125},
}

COMPLEX_KEYWORDS = [
    "explain", "compare", "analyse", "analyze", "design",
    "evaluate", "critique", "tradeoffs", "architecture",
    "implement", "optimize", "difference between", "pros and cons",
    "step by step", "in detail", "comprehensive", "thorough",
    "why does", "how does", "what would happen", "deep dive",
    "tell me about", "tell me more", "elaborate", "describe",
]

SIMPLE_KEYWORDS = [
    "what is", "define", "who is", "when did", "where is",
    "how many", "what year", "yes or no", "true or false",
    "name of", "capital of", "when was", "who was",
]

OBVIOUS_CHEAP = [
    "hello", "hi", "hey", "thanks", "thank you",
    "bye", "goodbye", "good morning", "good night",
    "how are you", "what's up", "ok", "okay", "yes", "no",
]

OBVIOUS_POWERFUL = [
    "implement a", "build a", "design a", "architecture",
]


@dataclass
class RoutingDecision:
    model: str
    provider: str
    tier: str
    complexity_score: int
    reasoning: str
    classified_by: str


class CostRouter:
    """
    Hybrid complexity router — two stage.

    Stage 1 — obvious filter (instant, free)
    Genuine greetings → cheap. Code blocks → powerful.

    Stage 2A — keyword scoring (instant, free)
    Score 1-10 based on signals. Clear cases decided here.

    Stage 2B — LLM classifier (future, Groq)
    Only fires for ambiguous 4-6 range.
    Placeholder ready — wire in when Groq key available.
    """

    def __init__(self):
        self.domain_detector = DomainDetector()
        self._llm_available = False
        self._classification_cache: dict[str, str] = {}

    # ─────────────────────────────────────────
    # Stage 1 — obvious filter
    # ─────────────────────────────────────────

    def _is_obvious_cheap(self, text: str) -> bool:
        """
        Only genuine greetings and single word replies.
        NOT triggered by length alone — that was the bug.
        """
        text_lower = text.lower().strip()

        # Single word response
        if len(text_lower.split()) == 1:
            return True

        # Starts with known greeting phrase
        if any(text_lower.startswith(kw) for kw in OBVIOUS_CHEAP):
            return True

        return False

    def _is_obvious_powerful(self, text: str) -> bool:
        """
        Code blocks in original text, or known complex patterns.
        Check original text for code — lowered text loses backticks.
        """
        # Code block signals — check original text
        if any(sig in text for sig in ["```", "def ", "class ", "import "]):
            return True

        # Complex patterns — check lowered
        text_lower = text.lower()
        if any(sig in text_lower for sig in OBVIOUS_POWERFUL):
            return True

        return False

    # ─────────────────────────────────────────
    # Stage 2A — keyword scoring
    # ─────────────────────────────────────────

    def _keyword_score(self, messages: list[Message]) -> tuple[int, str]:
        """
        Score complexity 1-10 from keyword signals.

        Key fix from v1:
        Complex keywords now guarantee minimum score of 5 (mid tier).
        Previously a short complex question could still score low
        because word count was the dominant signal.
        """
        last_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user = msg.content.lower().strip()
                break

        if not last_user:
            return 1, "no user message"

        score = 1
        reasons = []

        # Signal 1 — word count
        word_count = len(last_user.split())
        if word_count > 100:
            score += 3
            reasons.append(f"long ({word_count}w)")
        elif word_count > 50:
            score += 2
            reasons.append(f"medium ({word_count}w)")
        elif word_count > 20:
            score += 1
            reasons.append(f"short-med ({word_count}w)")

        # Signal 2 — complex keywords
        # Guarantee minimum score of 5 so complex keywords
        # always reach at least mid tier regardless of length
        matched_complex = [kw for kw in COMPLEX_KEYWORDS if kw in last_user]
        if matched_complex:
            score += min(len(matched_complex) * 2, 5)
            score = max(score, 5)  # floor at mid tier
            reasons.append(f"complex: {matched_complex[:2]}")

        # Signal 3 — simple keywords reduce score
        # Only applied when NO complex keywords present
        matched_simple = [kw for kw in SIMPLE_KEYWORDS if kw in last_user]
        if matched_simple and not matched_complex:
            score = max(1, score - 2)
            reasons.append(f"simple: {matched_simple[:1]}")

        # Signal 4 — domain
        domain = self.domain_detector.detect(last_user)
        if domain == Domain.CODE:
            score += 1
            reasons.append("code domain")
        elif domain == Domain.CONVERSATIONAL:
            score = max(1, score - 1)
            reasons.append("conversational")

        # Signal 5 — conversation depth
        user_turns = sum(1 for m in messages if m.role == "user")
        if user_turns > 5:
            score += 1
            reasons.append(f"deep convo ({user_turns} turns)")

        # Signal 6 — multi-part question
        multi = ["and also", "additionally", "furthermore", "as well as"]
        if any(m in last_user for m in multi):
            score += 1
            reasons.append("multi-part")

        score = max(1, min(10, score))
        return score, " | ".join(reasons) if reasons else "baseline"

    # ─────────────────────────────────────────
    # Stage 2B — LLM classifier (Groq, future)
    # ─────────────────────────────────────────

    async def _llm_classify(self, text: str) -> str | None:
        """
        Classify ambiguous prompts using fast cheap LLM.
        Returns "cheap" | "mid" | "powerful" | None

        Currently returns None — Groq not yet wired in.
        When Groq available:
        1. Set self._llm_available = True in __init__
        2. Add Groq async client
        3. Uncomment the API call below
        4. The rest of the router picks it up automatically
        """
        if not self._llm_available:
            return None

        cache_key = text[:100]
        if cache_key in self._classification_cache:
            return self._classification_cache[cache_key]

        # TODO — wire Groq here
        # prompt = f"""Classify complexity. Reply ONLY: SIMPLE, MEDIUM, or COMPLEX.
        # SIMPLE: greetings, factual lookups, yes/no questions
        # MEDIUM: explanations, summaries, basic analysis
        # COMPLEX: architecture, code implementation, deep analysis
        # Prompt: {text[:500]}"""
        # response = await groq_client.complete(prompt)
        # tier = self._parse_tier(response)
        # self._classification_cache[cache_key] = tier
        # return tier

        return None

    def _parse_tier(self, llm_response: str) -> str:
        r = llm_response.upper().strip()
        if "SIMPLE" in r:
            return "cheap"
        if "COMPLEX" in r:
            return "powerful"
        return "mid"

    # ─────────────────────────────────────────
    # Main routing decision
    # ─────────────────────────────────────────

    def decide(
        self,
        messages: list[Message],
        preferred_provider: str = "gemini",
    ) -> RoutingDecision:
        """
        Full routing flow:
        1. Obvious cheap → instant return
        2. Obvious powerful → instant return
        3. Keyword score → clear cases (≤3 or ≥7) decided here
        4. Ambiguous (4-6) → LLM classifier if available, else mid
        """
        last_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user = msg.content
                break

        provider_map = PROVIDER_MODELS.get(
            preferred_provider,
            PROVIDER_MODELS["gemini"]
        )

        # Stage 1 — obvious filter
        if self._is_obvious_cheap(last_user):
            return RoutingDecision(
                model=provider_map["cheap"],
                provider=preferred_provider,
                tier="cheap",
                complexity_score=1,
                reasoning="obvious cheap — greeting/single word",
                classified_by="obvious",
            )

        if self._is_obvious_powerful(last_user):
            return RoutingDecision(
                model=provider_map["powerful"],
                provider=preferred_provider,
                tier="powerful",
                complexity_score=9,
                reasoning="obvious powerful — code block/architecture",
                classified_by="obvious",
            )

        # Stage 2A — keyword scoring
        score, reasoning = self._keyword_score(messages)

        if score <= 3:
            tier = "cheap"
            classified_by = "keyword"
        elif score >= 7:
            tier = "powerful"
            classified_by = "keyword"
        else:
            # Ambiguous 4-6 range
            # LLM classifier fires here when Groq available
            # Falls back to mid tier until then
            tier = "mid"
            classified_by = "keyword"
            reasoning += " | ambiguous → mid (LLM classifier: pending Groq)"

        return RoutingDecision(
            model=provider_map[tier],
            provider=preferred_provider,
            tier=tier,
            complexity_score=score,
            reasoning=reasoning,
            classified_by=classified_by,
        )

    # ─────────────────────────────────────────
    # Cost tracking
    # ─────────────────────────────────────────

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        costs = MODEL_COSTS.get(
            model, {"prompt": 0.001, "completion": 0.002}
        )
        return round(
            (prompt_tokens / 1000) * costs["prompt"] +
            (completion_tokens / 1000) * costs["completion"],
            6
        )

    def savings_vs_powerful(
        self,
        model_used: str,
        prompt_tokens: int,
        completion_tokens: int,
        powerful_model: str = "gpt-4o",
    ) -> float:
        actual = self.estimate_cost(
            model_used, prompt_tokens, completion_tokens
        )
        powerful = self.estimate_cost(
            powerful_model, prompt_tokens, completion_tokens
        )
        return round(powerful - actual, 6)