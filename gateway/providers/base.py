from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator


@dataclass
class Message:
    role: str      # "user", "assistant", "system"
    content: str


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class CompletionResponse:
    content: str
    model: str
    provider: str
    usage: Usage


class BaseProvider(ABC):
    """
    Every provider adapter must implement these two methods.
    The gateway never calls OpenAI/Anthropic/Gemini directly —
    it only ever calls these methods. That's the adapter pattern.
    """

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[Message],
    ) -> CompletionResponse:
        """Single response — returns full answer at once"""
        pass

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[Message],
    ) -> AsyncGenerator[str, None]:
        """Streaming response — yields tokens one by one"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns False if API key is missing/placeholder"""
        pass