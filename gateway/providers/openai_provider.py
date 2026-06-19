from openai import AsyncOpenAI
from typing import AsyncGenerator

from gateway.providers.base import BaseProvider, Message, CompletionResponse, Usage
from gateway.config import settings


MODEL_MAP = {
    "auto":      "llama-3.1-8b-instant",
    "cheap":     "llama-3.1-8b-instant",
    "mid":       "llama-3.3-70b-versatile",
    "powerful":  "llama-3.3-70b-versatile",
    # Direct model names passthrough
    "llama-3.1-8b-instant":    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
}


class OpenAIProvider(BaseProvider):

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    def is_available(self) -> bool:
        return (
            bool(settings.openai_api_key)
            and not settings.openai_api_key.startswith("sk-placeholder")
        )

    def _resolve_model(self, model: str) -> str:
        return MODEL_MAP.get(model, "llama3-8b-8192")

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(
        self,
        model: str,
        messages: list[Message],
    ) -> CompletionResponse:
        resolved_model = self._resolve_model(model)

        response = await self.client.chat.completions.create(
            model=resolved_model,
            messages=self._format_messages(messages),
        )

        return CompletionResponse(
            content=response.choices[0].message.content,
            model=resolved_model,
            provider="groq",
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    async def stream(
        self,
        model: str,
        messages: list[Message],
    ) -> AsyncGenerator[str, None]:
        resolved_model = self._resolve_model(model)

        stream = await self.client.chat.completions.create(
            model=resolved_model,
            messages=self._format_messages(messages),
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta