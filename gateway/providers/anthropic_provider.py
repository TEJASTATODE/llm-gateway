import anthropic
from typing import AsyncGenerator

from gateway.providers.base import BaseProvider, Message, CompletionResponse, Usage
from gateway.config import settings


MODEL_MAP = {
    "claude-haiku": "claude-3-haiku-20240307",    # cheap + fast
    "claude-sonnet": "claude-sonnet-4-6",          # powerful
    "auto": "claude-3-haiku-20240307",             # default to cheap
}


class AnthropicProvider(BaseProvider):

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    def is_available(self) -> bool:
        return (
            bool(settings.anthropic_api_key)
            and not settings.anthropic_api_key.startswith("sk-ant-placeholder")
        )

    def _resolve_model(self, model: str) -> str:
        return MODEL_MAP.get(model, "claude-3-haiku-20240307")

    def _split_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """
        Anthropic is the only provider that requires the system prompt
        to be passed separately — not inside the messages array.
        This is what the adapter hides from the rest of the gateway.
        """
        system = ""
        conversation = []

        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                conversation.append({"role": m.role, "content": m.content})

        return system, conversation

    async def complete(
        self,
        model: str,
        messages: list[Message],
    ) -> CompletionResponse:
        resolved_model = self._resolve_model(model)
        system, conversation = self._split_messages(messages)

        kwargs = dict(
            model=resolved_model,
            max_tokens=1024,
            messages=conversation,
        )
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        return CompletionResponse(
            content=response.content[0].text,
            model=resolved_model,
            provider="anthropic",
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
        )

    async def stream(
        self,
        model: str,
        messages: list[Message],
    ) -> AsyncGenerator[str, None]:
        resolved_model = self._resolve_model(model)
        system, conversation = self._split_messages(messages)

        kwargs = dict(
            model=resolved_model,
            max_tokens=1024,
            messages=conversation,
        )
        if system:
            kwargs["system"] = system

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text