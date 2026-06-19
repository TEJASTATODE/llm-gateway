from gateway.providers.base import Message


class ContextBuilder:
    """
    Builds a context-aware cache key from conversation history.

    WHY THIS EXISTS:
    Basic cache keys use only the last message.
    Problem: same question, different context = different answer needed.

    Example:
    Conversation A: "I love animals" → "tell me about Python"
    Conversation B: "I am a developer" → "tell me about Python"

    Basic cache: both map to same key → wrong answer served
    Context cache: different history → different keys → correct answers

    HOW IT WORKS:
    Takes last N messages, joins them with a separator,
    producing a single string that captures both the question
    AND the conversation context it came from.
    """

    def __init__(self, context_window: int = 3):
        """
        context_window: how many previous messages to include.
        3 is the sweet spot —
        - Too few (1): misses context entirely
        - Too many (6+): unrelated old messages pollute the key,
          causing cache misses for conversations that are
          actually asking the same thing
        """
        self.context_window = context_window

    def build_key(self, messages: list[Message]) -> str:
        """
        Builds the context-aware string that gets embedded.

        Example input:
        [
            Message(role="user", content="I am a Python developer"),
            Message(role="assistant", content="Great! How can I help?"),
            Message(role="user", content="tell me about Python")
        ]

        Example output:
        "user: I am a Python developer | assistant: Great! How can I help? | user: tell me about Python"

        This full string gets embedded — not just the last message.
        The embedding captures the MEANING of the whole context.
        """
        if not messages:
            return ""

        # Take only the last N messages — context window
        recent = messages[-self.context_window:]

        # Build parts — "role: content" for each message
        parts = []
        for msg in recent:
            # Truncate very long messages to avoid embedding token limits
            # We only need the gist of previous messages for context
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"{msg.role}: {content}")

        # Join with pipe separator — clearly separates messages
        return " | ".join(parts)

    def get_last_user_message(self, messages: list[Message]) -> str:
        """
        Extracts the actual question being asked.
        Used for domain detection — we only classify the current question,
        not the whole conversation history.
        """
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""

    def build_display_key(self, messages: list[Message]) -> str:
        """
        Shorter version for logging and debugging.
        Shows what the cache key looks like without full content.
        """
        if not messages:
            return ""

        recent = messages[-self.context_window:]
        parts = []
        for msg in recent:
            preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            parts.append(f"{msg.role}: {preview}")

        return " | ".join(parts)