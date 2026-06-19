from gateway.providers.base import BaseProvider
from gateway.providers.gemini import GeminiProvider
from gateway.providers.openai_provider import OpenAIProvider
from gateway.providers.anthropic_provider import AnthropicProvider


class ProviderRegistry:
    """
    Single place that knows about all providers.
    Gateway asks registry for a provider by name —
    never instantiates providers directly.
    """

    def __init__(self):
        self._providers: dict[str, BaseProvider] = {
            "gemini": GeminiProvider(),
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
        }

    def get(self, name: str) -> BaseProvider:
        provider = self._providers.get(name)
        if not provider:
            raise ValueError(f"Unknown provider: {name}")
        return provider

    def available_providers(self) -> list[str]:
        """Returns only providers that have real API keys configured"""
        return [
            name for name, p in self._providers.items()
            if p.is_available()
        ]