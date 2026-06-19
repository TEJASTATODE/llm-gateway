from enum import Enum


class Domain(Enum):
    CODE = "code"
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    CONVERSATIONAL = "conversational"


# Keywords that signal each domain
# Order matters — code and factual are checked first
# because they're more precise than conceptual/conversational

CODE_KEYWORDS = [
    "code", "function", "bug", "error", "algorithm",
    "implement", "syntax", "debug", "class", "array",
    "loop", "variable", "import", "library",          # removed "api"
    "database", "sql", "query", "string", "integer",
    "recursion", "sort", "search", "compile", "runtime",
    "exception", "null", "object", "method", "return",
    "async", "await", "thread", "memory", "stack",
    "rest api", "api call", "api key", "api endpoint", # more specific
]

FACTUAL_KEYWORDS = [
    "what is", "who is", "when did", "where is",
    "how many", "define", "meaning of", "what are",
    "who was", "when was", "where was", "what does",
    "which is", "how much", "what year", "how old",
]

CONVERSATIONAL_KEYWORDS = [
    "how are you", "thank you", "thanks", "hello",
    "hi ", "hey ", "good morning", "good night",
    "what do you think", "can you help", "please help",
    "i need help", "nice to meet", "goodbye", "bye",
    "how do you", "what's up", "whats up",
]


class DomainDetector:
    """
    Classifies a message into one of 4 domains.
    Pure Python — no API calls, runs in microseconds.

    Why this matters:
    Each domain gets a different similarity threshold in the cache.
    Code needs 0.99 (very precise), conversational needs only 0.92.
    Wrong threshold = wrong cache hits = wrong answers served.
    """

    def detect(self, text: str) -> Domain:
        """
        Detect domain from message text.
        Checks in priority order — code first (most precise),
        conversational last (most general).
        """
        text_lower = text.lower().strip()

        # Check code first — highest precision needed
        if self._matches(text_lower, CODE_KEYWORDS):
            return Domain.CODE

        # Check conversational — short social exchanges
        if self._matches(text_lower, CONVERSATIONAL_KEYWORDS):
            return Domain.CONVERSATIONAL

        # Check factual — question words about specific facts
        if self._matches_factual(text_lower):
            return Domain.FACTUAL

        # Default — everything else is conceptual
        # Explanations, comparisons, analyses, opinions
        return Domain.CONCEPTUAL

    def _matches(self, text: str, keywords: list[str]) -> bool:
        """Returns True if any keyword found in text"""
        return any(kw in text for kw in keywords)

    def _matches_factual(self, text: str) -> bool:
        """
        Factual questions are trickier — we check keyword phrases
        not single words, so 'what is' matches but 'what' alone doesn't.
        Also checks if sentence ends with '?' and starts with question word.
        """
        if self._matches(text, FACTUAL_KEYWORDS):
            return True

        # Short questions ending in ? are usually factual
        question_starters = ["what", "who", "when", "where", "which", "how"]
        if text.endswith("?") and any(text.startswith(w) for w in question_starters):
            return True

        return False

    def get_threshold(self, domain: Domain) -> float:
        """
        Returns similarity threshold for this domain.
        This is the core of Improvement 2 — adaptive thresholds.

        CODE: 0.99 — sort ascending vs descending look similar but
                      have opposite answers. Must be very strict.

        FACTUAL: 0.95 — "what is ML" and "define ML" mean the same.
                         Can be more relaxed.

        CONCEPTUAL: 0.97 — explanations need to be relevant but
                            slight rephrasing is fine.

        CONVERSATIONAL: 0.92 — "how are you" and "how are you doing"
                                are the same. Very relaxed threshold.
        """
        thresholds = {
        Domain.CODE: 0.96,
        Domain.FACTUAL: 0.75,
        Domain.CONCEPTUAL: 0.78,
        Domain.CONVERSATIONAL: 0.70,
    }
        return thresholds[domain]